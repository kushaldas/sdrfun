//! Windowed-sinc FIR design and a complex decimating FIR.

use rustfft::num_complex::Complex32;

/// Linear-phase low-pass, Blackman-windowed. `cutoff` is in Hz at sample rate `fs`.
/// Taps are normalised to unity DC gain.
pub fn lowpass(num_taps: usize, cutoff: f32, fs: f32) -> Vec<f32> {
    let fc = cutoff / fs;
    let m = (num_taps - 1) as f32;
    let mut taps: Vec<f32> = (0..num_taps)
        .map(|n| {
            let x = n as f32 - m / 2.0;
            let sinc = if x == 0.0 {
                2.0 * fc
            } else {
                (2.0 * std::f32::consts::PI * fc * x).sin() / (std::f32::consts::PI * x)
            };
            let t = 2.0 * std::f32::consts::PI * n as f32 / m;
            let window = 0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos();
            sinc * window
        })
        .collect();
    let sum: f32 = taps.iter().sum();
    taps.iter_mut().for_each(|t| *t /= sum);
    taps
}

/// Complex-input FIR filter that keeps every `factor`-th output.
pub struct Decimator {
    taps: Vec<f32>,
    factor: usize,
    buf: Vec<Complex32>,
    next: usize,
}

impl Decimator {
    pub fn new(taps: Vec<f32>, factor: usize) -> Self {
        let history = taps.len() - 1;
        Self {
            taps,
            factor,
            buf: vec![Complex32::default(); history],
            next: 0,
        }
    }

    pub fn process(&mut self, input: &[Complex32], out: &mut Vec<Complex32>) {
        self.buf.extend_from_slice(input);
        let n = self.taps.len();
        while self.next + n <= self.buf.len() {
            let window = &self.buf[self.next..self.next + n];
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (s, &t) in window.iter().zip(&self.taps) {
                re += s.re * t;
                im += s.im * t;
            }
            out.push(Complex32::new(re, im));
            self.next += self.factor;
        }
        self.buf.drain(..self.next);
        self.next = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(f: f32, fs: f32, n: usize) -> Vec<Complex32> {
        (0..n)
            .map(|i| Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * f * i as f32 / fs))
            .collect()
    }

    fn gain(dec: &mut Decimator, input: &[Complex32]) -> f32 {
        let mut out = Vec::new();
        dec.process(input, &mut out);
        let tail = &out[out.len() / 2..];
        (tail.iter().map(|c| c.norm_sqr()).sum::<f32>() / tail.len() as f32).sqrt()
    }

    #[test]
    fn decimator_passes_band_and_rejects_aliases() {
        let fs = 240_000.0;
        let mk = || Decimator::new(lowpass(49, 24_000.0, fs), 5);
        assert!((gain(&mut mk(), &tone(3_000.0, fs, 24_000)) - 1.0).abs() < 0.01);
        // 46 kHz would alias to -2 kHz after decimating to 48 kHz.
        assert!(gain(&mut mk(), &tone(46_000.0, fs, 24_000)) < 0.003);
    }

    #[test]
    fn chunked_processing_matches_one_shot() {
        let input = tone(1234.0, 48_000.0, 5000);
        let mut a = Decimator::new(lowpass(31, 5000.0, 48_000.0), 3);
        let mut b = Decimator::new(lowpass(31, 5000.0, 48_000.0), 3);
        let (mut oa, mut ob) = (Vec::new(), Vec::new());
        a.process(&input, &mut oa);
        for chunk in input.chunks(77) {
            b.process(chunk, &mut ob);
        }
        assert_eq!(oa, ob);
    }
}
