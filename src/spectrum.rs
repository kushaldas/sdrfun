//! Power spectrum rows for the waterfall: Hann-windowed FFTs averaged over a row interval,
//! with DC in the middle (fftshift), in dB relative to a full-scale tone.

use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex32};

/// FFTs averaged per row (fewer when the input is too slow to fill them).
const AVERAGE: usize = 8;

pub struct Spectrum {
    n: usize,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    /// 1 / (sum of window)², so a full-scale tone reads 0 dB.
    scale: f32,
    fill: Vec<Complex32>,
    acc: Vec<f32>,
    count: usize,
    per_row: usize,
    /// Samples to skip between FFTs, so the averaged FFTs spread over the whole row.
    gap: usize,
    skip: usize,
}

impl Spectrum {
    /// `n` bins over `rate` samples/s, about `rows_per_s` rows a second.
    pub fn new(n: usize, rate: u32, rows_per_s: u32) -> Self {
        let window: Vec<f32> = (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
            .collect();
        let sum: f32 = window.iter().sum();
        let interval = (rate / rows_per_s.max(1)) as usize;
        let per_row = (interval / n).clamp(1, AVERAGE);
        Self {
            n,
            fft: FftPlanner::new().plan_fft_forward(n),
            scale: 1.0 / (sum * sum),
            window,
            fill: Vec::with_capacity(n),
            acc: vec![0.0; n],
            count: 0,
            per_row,
            gap: (interval / per_row).saturating_sub(n),
            skip: 0,
        }
    }

    /// Raw interleaved u8 IQ from the dongle; finished rows are appended to `rows`.
    pub fn push_u8(&mut self, iq: &[u8], rows: &mut Vec<Vec<f32>>) {
        let mut pairs = iq.chunks_exact(2);
        loop {
            if self.skip > 0 {
                let skip = self.skip.min(pairs.len());
                if skip == 0 {
                    return;
                }
                pairs.nth(skip - 1);
                self.skip -= skip;
                continue;
            }
            let Some(p) = pairs.next() else { return };
            self.push_one(Complex32::new((p[0] as f32 - 127.4) / 128.0, (p[1] as f32 - 127.4) / 128.0), rows);
        }
    }

    pub fn push(&mut self, iq: &[Complex32], rows: &mut Vec<Vec<f32>>) {
        let mut rest = iq;
        while !rest.is_empty() {
            if self.skip > 0 {
                let skip = self.skip.min(rest.len());
                rest = &rest[skip..];
                self.skip -= skip;
                continue;
            }
            self.push_one(rest[0], rows);
            rest = &rest[1..];
        }
    }

    fn push_one(&mut self, z: Complex32, rows: &mut Vec<Vec<f32>>) {
        self.fill.push(z);
        if self.fill.len() < self.n {
            return;
        }
        for (z, w) in self.fill.iter_mut().zip(&self.window) {
            *z *= w;
        }
        self.fft.process(&mut self.fill);
        for (a, z) in self.acc.iter_mut().zip(&self.fill) {
            *a += z.norm_sqr();
        }
        self.fill.clear();
        self.count += 1;
        self.skip = self.gap;
        if self.count == self.per_row {
            let half = self.n / 2;
            let k = self.scale / self.count as f32;
            let row = (0..self.n)
                .map(|i| 10.0 * (self.acc[(i + half) % self.n] * k).max(1e-15).log10())
                .collect();
            rows.push(row);
            self.acc.iter_mut().for_each(|a| *a = 0.0);
            self.count = 0;
        }
    }
}

/// Colour-scale range that follows the noise floor and the strongest signals.
#[derive(Clone, Copy, Debug)]
pub struct Levels {
    pub lo: f32,
    pub hi: f32,
    started: bool,
}

impl Default for Levels {
    fn default() -> Self {
        Self { lo: -110.0, hi: -40.0, started: false }
    }
}

impl Levels {
    /// Update from a row (dB) and quantise it to 0..=255 over the range.
    pub fn quantize(&mut self, row: &[f32]) -> Vec<u8> {
        let mut sorted = row.to_vec();
        sorted.sort_by(f32::total_cmp);
        let floor = sorted[sorted.len() / 5] - 5.0;
        let peak = sorted[sorted.len() - 1] + 5.0;
        let peak = peak.max(floor + 30.0);
        if self.started {
            self.lo += (floor - self.lo) * 0.05;
            self.hi += (peak - self.hi) * 0.05;
        } else {
            (self.lo, self.hi, self.started) = (floor, peak, true);
        }
        let span = (self.hi - self.lo).max(1.0);
        row.iter()
            .map(|db| ((db - self.lo) / span * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_lands_in_its_bin_at_full_scale() {
        let rate = 2_400_000;
        let mut s = Spectrum::new(2048, rate, 10);
        let f = 300_000.0;
        let iq: Vec<u8> = (0..rate as usize / 5)
            .flat_map(|i| {
                let ph = 2.0 * std::f64::consts::PI * f * i as f64 / rate as f64;
                [(127.4 + 100.0 * ph.cos()).round() as u8, (127.4 + 100.0 * ph.sin()).round() as u8]
            })
            .collect();
        let mut rows = Vec::new();
        s.push_u8(&iq, &mut rows);
        assert_eq!(rows.len(), 2, "10 rows/s over 0.2 s");
        let row = &rows[1];
        let peak = (0..row.len()).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
        let expected = 1024 + (f / rate as f64 * 2048.0).round() as usize;
        assert_eq!(peak, expected);
        // Amplitude 100/128 → about -2.1 dB.
        assert!((row[peak] + 2.1).abs() < 1.5, "peak {} dB", row[peak]);
        let noise = row[200];
        assert!(noise < row[peak] - 50.0, "far bin {noise} dB");
    }

    #[test]
    fn slow_input_still_makes_rows() {
        let mut s = Spectrum::new(1024, 48_000, 10);
        let mut rows = Vec::new();
        s.push(&vec![Complex32::new(0.5, 0.0); 48_000], &mut rows);
        assert_eq!(rows.len(), 10);
        // DC sits in the middle after the shift.
        let row = &rows[5];
        let peak = (0..row.len()).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
        assert_eq!(peak, 512);
    }

    #[test]
    fn levels_map_floor_low_and_peak_high() {
        let mut levels = Levels::default();
        let mut row = vec![-100.0; 1000];
        row[500] = -20.0;
        let q = levels.quantize(&row);
        assert!(q[0] < 30, "floor {}", q[0]);
        assert!(q[500] > 220, "peak {}", q[500]);
    }
}
