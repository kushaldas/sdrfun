//! Speech AGC on 10 ms blocks followed by a soft limiter.

use super::FRAME;

/// Level-follower coefficients per 10 ms block.
const ATTACK: f32 = 0.3;
const RELEASE: f32 = 0.02;
/// Blocks quieter than this (-60 dBFS) do not raise the gain any further.
const LEVEL_FLOOR: f32 = 1e-3;
/// Above this magnitude the limiter starts to compress.
const LIMIT_KNEE: f32 = 0.8;

pub struct Agc {
    target: f32,
    max_gain: f32,
    level: f32,
    gain: f32,
}

impl Agc {
    pub fn new(target_dbfs: f32, max_gain_db: f32) -> Self {
        Self {
            target: db_to_amp(target_dbfs),
            max_gain: db_to_amp(max_gain_db),
            level: 0.0,
            gain: 1.0,
        }
    }

    pub fn process_frame(&mut self, frame: &mut [f32; FRAME]) {
        let rms = (frame.iter().map(|x| x * x).sum::<f32>() / FRAME as f32).sqrt();
        let coeff = if rms > self.level { ATTACK } else { RELEASE };
        self.level += (rms - self.level) * coeff;

        let wanted = (self.target / self.level.max(LEVEL_FLOOR)).min(self.max_gain);
        // Ramp linearly across the block to avoid zipper noise.
        let start = self.gain;
        let delta = (wanted - start) / FRAME as f32;
        for (i, x) in frame.iter_mut().enumerate() {
            *x = soft_limit(*x * (start + delta * (i + 1) as f32));
        }
        self.gain = wanted;
    }
}

fn soft_limit(x: f32) -> f32 {
    let a = x.abs();
    if a <= LIMIT_KNEE {
        x
    } else {
        let room = 1.0 - LIMIT_KNEE;
        x.signum() * (LIMIT_KNEE + room * ((a - LIMIT_KNEE) / room).tanh())
    }
}

fn db_to_amp(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_and_loud_tones_converge_to_target() {
        for amp in [0.01f32, 0.5] {
            let mut agc = Agc::new(-20.0, 30.0);
            let mut last_rms = 0.0;
            for block in 0..300 {
                let mut frame = [0.0; FRAME];
                for (i, x) in frame.iter_mut().enumerate() {
                    let n = block * FRAME + i;
                    *x = amp * (2.0 * std::f32::consts::PI * 500.0 * n as f32 / 48_000.0).sin();
                }
                agc.process_frame(&mut frame);
                last_rms = (frame.iter().map(|x| x * x).sum::<f32>() / FRAME as f32).sqrt();
            }
            assert!((last_rms - 0.1).abs() < 0.01, "amp {amp}: rms {last_rms}");
        }
    }

    #[test]
    fn limiter_never_exceeds_full_scale() {
        for x in [-10.0f32, -1.0, -0.5, 0.0, 0.79, 0.9, 3.0, 100.0] {
            assert!(soft_limit(x).abs() <= 1.0);
        }
        assert_eq!(soft_limit(0.5), 0.5);
    }
}
