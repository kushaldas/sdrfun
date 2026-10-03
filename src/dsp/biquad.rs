//! RBJ-cookbook biquads (transposed direct form II) and Butterworth cascades.

use std::f32::consts::PI;

/// Q values of the two second-order sections of a 4th-order Butterworth filter.
const BUTTER4_Q: [f32; 2] = [0.541_196_1, 1.306_563];

#[derive(Clone, Debug)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn from_coeffs(b: [f32; 3], a: [f32; 3]) -> Self {
        Self {
            b0: b[0] / a[0],
            b1: b[1] / a[0],
            b2: b[2] / a[0],
            a1: a[1] / a[0],
            a2: a[2] / a[0],
            z1: 0.0,
            z2: 0.0,
        }
    }

    pub fn lowpass(fs: f32, f0: f32, q: f32) -> Self {
        let w0 = 2.0 * PI * f0 / fs;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        Self::from_coeffs(
            [(1.0 - cos) / 2.0, 1.0 - cos, (1.0 - cos) / 2.0],
            [1.0 + alpha, -2.0 * cos, 1.0 - alpha],
        )
    }

    pub fn highpass(fs: f32, f0: f32, q: f32) -> Self {
        let w0 = 2.0 * PI * f0 / fs;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        Self::from_coeffs(
            [(1.0 + cos) / 2.0, -(1.0 + cos), (1.0 + cos) / 2.0],
            [1.0 + alpha, -2.0 * cos, 1.0 - alpha],
        )
    }

    /// High shelf: `gain_db` above `f0`, unity well below it.
    pub fn highshelf(fs: f32, f0: f32, q: f32, gain_db: f32) -> Self {
        let a = 10f32.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * f0 / fs;
        let (sin, cos) = w0.sin_cos();
        let beta = 2.0 * a.sqrt() * sin / (2.0 * q);
        Self::from_coeffs(
            [
                a * ((a + 1.0) + (a - 1.0) * cos + beta),
                -2.0 * a * ((a - 1.0) + (a + 1.0) * cos),
                a * ((a + 1.0) + (a - 1.0) * cos - beta),
            ],
            [
                (a + 1.0) - (a - 1.0) * cos + beta,
                2.0 * ((a - 1.0) - (a + 1.0) * cos),
                (a + 1.0) - (a - 1.0) * cos - beta,
            ],
        )
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// A cascade of biquads applied in series.
#[derive(Clone, Debug, Default)]
pub struct Cascade(Vec<Biquad>);

impl Cascade {
    /// 4th-order Butterworth band-pass built from a high-pass and a low-pass pair.
    /// Either edge can be disabled by passing `None`.
    pub fn bandpass(fs: f32, low_hz: Option<f32>, high_hz: Option<f32>) -> Self {
        let mut stages = Vec::new();
        if let Some(f) = low_hz {
            stages.extend(BUTTER4_Q.iter().map(|&q| Biquad::highpass(fs, f, q)));
        }
        if let Some(f) = high_hz {
            stages.extend(BUTTER4_Q.iter().map(|&q| Biquad::lowpass(fs, f, q)));
        }
        Self(stages)
    }

    /// Presence boost: a single high shelf.
    pub fn presence(fs: f32, shelf_hz: f32, gain_db: f32) -> Self {
        Self(vec![Biquad::highshelf(fs, shelf_hz, 0.7, gain_db)])
    }

    pub fn process_in_place(&mut self, buf: &mut [f32]) {
        for x in buf.iter_mut() {
            *x = self.0.iter_mut().fold(*x, |acc, s| s.process(acc));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Steady-state gain of a fresh cascade for a sine at `f` Hz.
    fn gain_of(mut c: Cascade, f: f32) -> f32 {
        let fs = 48_000.0;
        let mut buf: Vec<f32> = (0..48_000)
            .map(|n| (2.0 * PI * f * n as f32 / fs).sin())
            .collect();
        c.process_in_place(&mut buf);
        let tail = &buf[24_000..];
        let rms = (tail.iter().map(|x| x * x).sum::<f32>() / tail.len() as f32).sqrt();
        rms * 2f32.sqrt()
    }

    fn gain_at(f: f32) -> f32 {
        gain_of(Cascade::bandpass(48_000.0, Some(250.0), Some(3400.0)), f)
    }

    #[test]
    fn presence_boosts_consonant_band_only() {
        let db = |f: f32| 20.0 * gain_of(Cascade::presence(48_000.0, 1500.0, 14.0), f).log10();
        assert!(db(200.0).abs() < 1.5, "200 Hz: {}", db(200.0));
        assert!((db(1500.0) - 7.0).abs() < 1.0, "corner: {}", db(1500.0));
        assert!((db(5000.0) - 14.0).abs() < 1.0, "5 kHz: {}", db(5000.0));
    }

    #[test]
    fn bandpass_passes_voice_and_rejects_edges() {
        assert!((gain_at(1000.0) - 1.0).abs() < 0.05);
        assert!(gain_at(50.0) < 0.01);
        assert!(gain_at(10_000.0) < 0.02);
        // Butterworth edges are -3 dB at the corner frequencies.
        assert!((gain_at(250.0) - 0.707).abs() < 0.05);
        assert!((gain_at(3400.0) - 0.707).abs() < 0.05);
    }
}
