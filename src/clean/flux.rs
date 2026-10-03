//! Spectral shape change between frames: speech keeps moving its harmonics and
//! formants, a steady tone or data burst does not.

use std::collections::VecDeque;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex32};

use super::{FRAME, SAMPLE_RATE};

/// Compare each frame with the one this many frames (10 ms each) earlier.
const LAG: usize = 5;
const LOW_HZ: f32 = 250.0;
const HIGH_HZ: f32 = 3400.0;

pub struct SpectralFlux {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    buf: Vec<Complex32>,
    bins: std::ops::RangeInclusive<usize>,
    history: VecDeque<Vec<f32>>,
}

impl SpectralFlux {
    pub fn new() -> Self {
        let bin_hz = SAMPLE_RATE as f32 / FRAME as f32;
        Self {
            fft: FftPlanner::new().plan_fft_forward(FRAME),
            window: (0..FRAME)
                .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / FRAME as f32).cos())
                .collect(),
            buf: vec![Complex32::default(); FRAME],
            bins: (LOW_HZ / bin_hz).ceil() as usize..=(HIGH_HZ / bin_hz).floor() as usize,
            history: VecDeque::with_capacity(LAG + 1),
        }
    }

    /// Change of the normalised spectrum against `LAG` frames ago, 0 (same) to 2 (disjoint).
    pub fn process(&mut self, frame: &[f32; FRAME]) -> f32 {
        for ((c, &x), &w) in self.buf.iter_mut().zip(frame).zip(&self.window) {
            *c = Complex32::new(x * w, 0.0);
        }
        self.fft.process(&mut self.buf);
        let mut shape: Vec<f32> = self.buf[self.bins.clone()].iter().map(|c| c.norm_sqr()).collect();
        let total: f32 = shape.iter().sum();
        if total > 0.0 {
            shape.iter_mut().for_each(|p| *p /= total);
        }

        let flux = if self.history.len() == LAG {
            let old = self.history.pop_front().unwrap();
            old.iter().zip(&shape).map(|(a, b)| (a - b).abs()).sum()
        } else {
            0.0
        };
        self.history.push_back(shape);
        flux
    }
}
