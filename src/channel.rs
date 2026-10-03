//! One AM channel: IQ → frequency shift → decimate to 48 kHz → channel filter →
//! power measurement + envelope demodulation, emitted as 10 ms frames.

use anyhow::{Result, bail};
use rustfft::num_complex::Complex32;

use crate::clean::{FRAME, SAMPLE_RATE};
use crate::dsp::fir::{Decimator, lowpass};

/// Time constant of the carrier-level tracker used to normalise AM depth.
const CARRIER_TAU_S: f32 = 0.05;
const CHANNEL_TAPS: usize = 129;

pub struct Frame {
    /// Demodulated audio, 48 kHz, roughly ±0.5 at 100 % modulation.
    pub audio: [f32; FRAME],
    /// Mean channel power over the frame, dBFS.
    pub power_db: f32,
}

pub struct Channel {
    phasor: Complex32,
    rotation: Complex32,
    stages: Vec<Decimator>,
    channel_filter: Decimator,
    carrier: f32,
    carrier_alpha: f32,
    scratch: [Vec<Complex32>; 2],
    audio: Vec<f32>,
    power_sum: f32,
}

impl Channel {
    /// `offset_hz` is the target frequency relative to the tuner centre.
    pub fn new(sample_rate: u32, offset_hz: f64, bandwidth_hz: f32) -> Result<Self> {
        if sample_rate % SAMPLE_RATE != 0 {
            bail!("sample rate {sample_rate} must be a multiple of {SAMPLE_RATE}");
        }
        if !(1000.0..=40_000.0).contains(&bandwidth_hz) {
            bail!("bandwidth {bandwidth_hz} Hz must be between 1 kHz and 40 kHz");
        }
        if offset_hz.abs() + bandwidth_hz as f64 / 2.0 >= sample_rate as f64 / 2.0 {
            bail!("offset {offset_hz} Hz puts the channel outside the captured band");
        }
        let factors = split_decimation(sample_rate / SAMPLE_RATE)?;

        // Each stage only has to keep aliases out of the final passband.
        let pass = (bandwidth_hz / 2.0).max(5_000.0);
        let mut fin = sample_rate as f32;
        let stages = factors
            .iter()
            .map(|&d| {
                let fout = fin / d as f32;
                let transition = (fout - 2.0 * pass).max(fout * 0.1);
                let taps = ((5.5 * fin / transition).ceil() as usize) | 1;
                let stage = Decimator::new(lowpass(taps, fout / 2.0, fin), d as usize);
                fin = fout;
                stage
            })
            .collect();

        let w = -2.0 * std::f64::consts::PI * offset_hz / sample_rate as f64;
        Ok(Self {
            phasor: Complex32::new(1.0, 0.0),
            rotation: Complex32::new(w.cos() as f32, w.sin() as f32),
            stages,
            channel_filter: Decimator::new(
                lowpass(CHANNEL_TAPS, bandwidth_hz / 2.0, SAMPLE_RATE as f32),
                1,
            ),
            carrier: 0.0,
            carrier_alpha: 1.0 - (-1.0 / (SAMPLE_RATE as f32 * CARRIER_TAU_S)).exp(),
            scratch: [Vec::new(), Vec::new()],
            audio: Vec::with_capacity(FRAME),
            power_sum: 0.0,
        })
    }

    /// Process raw interleaved u8 IQ from the dongle.
    pub fn process_u8(&mut self, iq: &[u8], frames: &mut Vec<Frame>) {
        let [a, b] = &mut self.scratch;
        a.clear();
        a.extend(iq.chunks_exact(2).map(|p| {
            let x = Complex32::new((p[0] as f32 - 127.4) / 128.0, (p[1] as f32 - 127.4) / 128.0);
            let y = x * self.phasor;
            self.phasor *= self.rotation;
            y
        }));
        // Keep the oscillator on the unit circle.
        self.phasor /= self.phasor.norm();

        for stage in &mut self.stages {
            b.clear();
            stage.process(a, b);
            std::mem::swap(a, b);
        }
        b.clear();
        self.channel_filter.process(a, b);

        for &z in b.iter() {
            let power = z.norm_sqr();
            let env = power.sqrt();
            if self.carrier == 0.0 {
                self.carrier = env;
            }
            self.carrier += (env - self.carrier) * self.carrier_alpha;
            self.audio.push(0.5 * (env / self.carrier.max(1e-9) - 1.0));
            self.power_sum += power;
            if self.audio.len() == FRAME {
                let mut audio = [0.0; FRAME];
                audio.copy_from_slice(&self.audio);
                self.audio.clear();
                let power_db = 10.0 * (self.power_sum / FRAME as f32).max(1e-12).log10();
                self.power_sum = 0.0;
                frames.push(Frame { audio, power_db });
            }
        }
    }
}

/// Split a decimation factor into stages of at most 10 each, largest first.
fn split_decimation(mut total: u32) -> Result<Vec<u32>> {
    let original = total;
    let mut factors = Vec::new();
    while total > 1 {
        let Some(d) = (2..=10).rev().find(|d| total % d == 0) else {
            bail!("cannot decimate by {original}: pick a sample rate of 48 kHz × (product of factors ≤ 10)");
        };
        factors.push(d);
        total /= d;
    }
    Ok(factors)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// u8 IQ of an AM carrier at `offset` Hz with a 1 kHz tone at `depth` modulation.
    fn am_iq(fs: u32, offset: f64, depth: f64, amp: f64, seconds: f64) -> Vec<u8> {
        let n = (fs as f64 * seconds) as usize;
        let mut out = Vec::with_capacity(2 * n);
        for i in 0..n {
            let t = i as f64 / fs as f64;
            let m = 1.0 + depth * (2.0 * std::f64::consts::PI * 1000.0 * t).sin();
            let ph = 2.0 * std::f64::consts::PI * offset * t;
            for v in [amp * m * ph.cos(), amp * m * ph.sin()] {
                out.push((127.4 + v * 128.0).round().clamp(0.0, 255.0) as u8);
            }
        }
        out
    }

    #[test]
    fn split_decimation_factors() {
        assert_eq!(split_decimation(50).unwrap(), vec![10, 5]);
        assert_eq!(split_decimation(60).unwrap(), vec![10, 6]);
        assert!(split_decimation(11).is_err());
    }

    #[test]
    fn demodulates_offset_am_tone() {
        let fs = 2_400_000;
        let mut ch = Channel::new(fs, 250_000.0, 10_000.0).unwrap();
        let mut frames = Vec::new();
        for block in am_iq(fs, 250_000.0, 0.8, 0.3, 0.5).chunks(262_144) {
            ch.process_u8(block, &mut frames);
        }
        assert_eq!(frames.len(), 50, "10 ms frames for 0.5 s");
        let audio: Vec<f32> = frames[10..].iter().flat_map(|f| f.audio).collect();
        // 80 % depth, scaled by 0.5 → amplitude 0.4, RMS ≈ 0.283.
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        assert!((rms - 0.283).abs() < 0.02, "rms {rms}");
        // Carrier power: amp² × (1 + depth²/2) ≈ 0.09 × 1.32 → about -9.2 dBFS.
        let p = frames[20].power_db;
        assert!((p + 9.2).abs() < 1.0, "power {p}");
    }

    #[test]
    fn rejects_signal_outside_channel() {
        let fs = 2_400_000;
        let mut ch = Channel::new(fs, 250_000.0, 10_000.0).unwrap();
        let mut frames = Vec::new();
        // Same signal 25 kHz away (one 25 kHz channel up).
        for block in am_iq(fs, 275_000.0, 0.8, 0.3, 0.3).chunks(262_144) {
            ch.process_u8(block, &mut frames);
        }
        assert!(frames[15].power_db < -9.2 - 40.0, "leak {}", frames[15].power_db);
    }
}
