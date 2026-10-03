use nnnoiseless::DenoiseState;

use super::{Denoiser, FRAME};

/// RNNoise (via the pure-Rust `nnnoiseless` port). Expects 48 kHz audio.
pub struct RnNoise {
    state: Box<DenoiseState<'static>>,
    scaled: [f32; FRAME],
    out: [f32; FRAME],
}

impl RnNoise {
    pub fn new() -> Self {
        Self {
            state: DenoiseState::new(),
            scaled: [0.0; FRAME],
            out: [0.0; FRAME],
        }
    }
}

impl Denoiser for RnNoise {
    fn process_frame(&mut self, frame: &mut [f32; FRAME]) -> Option<f32> {
        // nnnoiseless works on 16-bit-scaled floats.
        for (s, x) in self.scaled.iter_mut().zip(frame.iter()) {
            *s = x * 32768.0;
        }
        let vad = self.state.process_frame(&mut self.out, &self.scaled);
        for (x, o) in frame.iter_mut().zip(self.out.iter()) {
            *x = o / 32768.0;
        }
        Some(vad)
    }

    fn latency(&self) -> usize {
        // Output is the overlap-add of the previous frame.
        FRAME
    }
}
