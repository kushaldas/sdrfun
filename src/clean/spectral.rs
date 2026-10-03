//! Classic STFT noise suppression: minimum-statistics noise tracking plus a
//! decision-directed Wiener gain. No model, good on steady hiss.

use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex32};

use super::{Denoiser, FRAME};

const N: usize = 2 * FRAME;
const BINS: usize = N / 2 + 1;
/// Smoothing of the per-bin power before minimum tracking.
const POWER_SMOOTH: f32 = 0.8;
/// How fast the noise estimate may rise: about 5 dB/s at 100 frames/s.
const NOISE_RISE: f32 = 1.0116;
/// Minimum tracking underestimates the mean noise power; compensate (and over-subtract a bit).
const NOISE_BIAS: f32 = 2.0;
/// Decision-directed a-priori SNR smoothing.
const DD_ALPHA: f32 = 0.98;
/// Lowest gain applied to any bin (-20 dB), which limits musical noise.
const GAIN_FLOOR: f32 = 0.1;

pub struct Spectral {
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    input: Vec<f32>,
    overlap: Vec<f32>,
    spectrum: Vec<Complex32>,
    smoothed: Vec<f32>,
    noise: Vec<f32>,
    prev_clean: Vec<f32>,
    primed: bool,
}

impl Spectral {
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        // Periodic sqrt-Hann for analysis and synthesis sums to unity at 50 % overlap.
        let window = (0..N)
            .map(|n| (0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / N as f32).cos()).sqrt())
            .collect();
        Self {
            fft: planner.plan_fft_forward(N),
            ifft: planner.plan_fft_inverse(N),
            window,
            input: vec![0.0; N],
            overlap: vec![0.0; N],
            spectrum: vec![Complex32::default(); N],
            smoothed: vec![0.0; BINS],
            noise: vec![0.0; BINS],
            prev_clean: vec![0.0; BINS],
            primed: false,
        }
    }
}

impl Denoiser for Spectral {
    fn process_frame(&mut self, frame: &mut [f32; FRAME]) -> Option<f32> {
        self.input.copy_within(FRAME.., 0);
        self.input[FRAME..].copy_from_slice(frame);

        for ((c, x), w) in self.spectrum.iter_mut().zip(&self.input).zip(&self.window) {
            *c = Complex32::new(x * w, 0.0);
        }
        self.fft.process(&mut self.spectrum);

        for k in 0..BINS {
            let power = self.spectrum[k].norm_sqr();
            if !self.primed {
                self.smoothed[k] = power;
                self.noise[k] = power;
            }
            self.smoothed[k] = POWER_SMOOTH * self.smoothed[k] + (1.0 - POWER_SMOOTH) * power;
            self.noise[k] = (self.noise[k] * NOISE_RISE).min(self.smoothed[k]);

            let noise = (self.noise[k] * NOISE_BIAS).max(1e-12);
            let post_snr = power / noise;
            let prio_snr =
                DD_ALPHA * self.prev_clean[k] / noise + (1.0 - DD_ALPHA) * (post_snr - 1.0).max(0.0);
            let gain = (prio_snr / (1.0 + prio_snr)).max(GAIN_FLOOR);
            self.prev_clean[k] = gain * gain * power;

            self.spectrum[k] *= gain;
            if k > 0 && k < N / 2 {
                self.spectrum[N - k] = self.spectrum[k].conj();
            }
        }
        self.primed = true;

        self.ifft.process(&mut self.spectrum);
        let scale = 1.0 / N as f32;
        for ((o, c), w) in self.overlap.iter_mut().zip(&self.spectrum).zip(&self.window) {
            *o += c.re * scale * w;
        }
        frame.copy_from_slice(&self.overlap[..FRAME]);
        self.overlap.copy_within(FRAME.., 0);
        self.overlap[FRAME..].fill(0.0);
        None
    }

    fn latency(&self) -> usize {
        FRAME
    }
}
