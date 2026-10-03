//! Streaming arbitrary-ratio resampler (Kaiser-windowed sinc, table lookup).

const ZERO_CROSSINGS: f64 = 24.0;
const TABLE_RES: usize = 256;
const KAISER_BETA: f64 = 8.0;

pub struct Resampler {
    /// Input samples advanced per output sample.
    step: f64,
    /// Kernel half-width in input samples.
    half: usize,
    /// Kernel sampled at `1 / TABLE_RES` input-sample spacing, from 0 to `half`.
    table: Vec<f32>,
    buf: Vec<f32>,
    /// Position of the next output sample, in `buf` coordinates.
    pos: f64,
    passthrough: bool,
}

impl Resampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        let step = in_rate as f64 / out_rate as f64;
        // Cut a bit below the lower Nyquist so the transition band does not alias.
        let cutoff = (out_rate as f64 / in_rate as f64).min(1.0) * 0.92;
        let half = (ZERO_CROSSINGS / cutoff).ceil() as usize;
        let table = (0..=half * TABLE_RES + 1)
            .map(|i| {
                let x = i as f64 / TABLE_RES as f64;
                (cutoff * sinc(cutoff * x) * kaiser(x / half as f64)) as f32
            })
            .collect();
        Self {
            step,
            half,
            table,
            buf: vec![0.0; half],
            pos: half as f64,
            passthrough: in_rate == out_rate,
        }
    }

    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if self.passthrough {
            out.extend_from_slice(input);
            return;
        }
        self.buf.extend_from_slice(input);
        while self.pos.floor() as usize + self.half < self.buf.len() {
            out.push(self.sample_at(self.pos));
            self.pos += self.step;
        }
        let consumed = (self.pos.floor() as usize).saturating_sub(self.half);
        self.buf.drain(..consumed);
        self.pos -= consumed as f64;
    }

    /// Push enough silence to emit every output sample that depends on real input.
    pub fn flush(&mut self, out: &mut Vec<f32>) {
        if !self.passthrough {
            self.process(&vec![0.0; self.half + 1], out);
        }
    }

    fn sample_at(&self, pos: f64) -> f32 {
        let center = pos.floor() as usize;
        let lo = center + 1 - self.half;
        let hi = center + self.half;
        (lo..=hi)
            .map(|j| self.buf[j] * self.kernel((pos - j as f64).abs()))
            .sum()
    }

    fn kernel(&self, d: f64) -> f32 {
        let x = d * TABLE_RES as f64;
        let i = x as usize;
        if i + 1 >= self.table.len() {
            return 0.0;
        }
        let frac = (x - i as f64) as f32;
        self.table[i] + (self.table[i + 1] - self.table[i]) * frac
    }
}

/// Resample a whole buffer, keeping the output length at `len * out / in`.
pub fn resample_all(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    let mut r = Resampler::new(in_rate, out_rate);
    let mut out = Vec::with_capacity(input.len() * out_rate as usize / in_rate as usize + 1);
    r.process(input, &mut out);
    r.flush(&mut out);
    out.truncate((input.len() as u64 * out_rate as u64 / in_rate as u64) as usize);
    out
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

fn kaiser(t: f64) -> f64 {
    if t.abs() > 1.0 {
        0.0
    } else {
        bessel_i0(KAISER_BETA * (1.0 - t * t).sqrt()) / bessel_i0(KAISER_BETA)
    }
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    for k in 1..50 {
        term *= q / (k * k) as f64;
        sum += term;
        if term < sum * 1e-12 {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn tone(f: f32, fs: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * f * i as f32 / fs as f32).sin())
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn downsample_keeps_passband_tone_in_phase() {
        let input = tone(1000.0, 48_000, 48_000);
        let out = resample_all(&input, 48_000, 16_000);
        assert_eq!(out.len(), 16_000);
        let expected = tone(1000.0, 16_000, 16_000);
        let err: f32 = out[1000..15_000]
            .iter()
            .zip(&expected[1000..15_000])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(err < 0.01, "max error {err}");
    }

    #[test]
    fn downsample_rejects_tone_above_new_nyquist() {
        let input = tone(10_000.0, 48_000, 48_000);
        let out = resample_all(&input, 48_000, 16_000);
        assert!(rms(&out[1000..15_000]) < 0.001);
    }

    #[test]
    fn upsample_and_streaming_match_one_shot() {
        let input = tone(440.0, 44_100, 10_000);
        let one_shot = resample_all(&input, 44_100, 48_000);
        let mut r = Resampler::new(44_100, 48_000);
        let mut streamed = Vec::new();
        for chunk in input.chunks(333) {
            r.process(chunk, &mut streamed);
        }
        r.flush(&mut streamed);
        streamed.truncate(one_shot.len());
        assert_eq!(one_shot.len(), streamed.len());
        let diff = one_shot
            .iter()
            .zip(&streamed)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(diff < 1e-4, "streaming diverged by {diff}");
        assert!((rms(&one_shot[1000..9000]) - 0.7071).abs() < 0.01);
    }
}
