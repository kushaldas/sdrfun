//! One channel: IQ → frequency shift → decimate to 48 kHz → channel filter →
//! power and carrier measurement + AM or FM demodulation, emitted as 10 ms frames.

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::{Result, bail};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};

use crate::clean::{FRAME, SAMPLE_RATE};
use crate::dsp::fir::{Decimator, lowpass};

/// Time constant of the carrier-level tracker used to normalise AM depth.
const CARRIER_TAU_S: f32 = 0.05;
const CHANNEL_TAPS: usize = 129;
/// Narrowband FM: peak deviation that maps to ±0.5 audio (like 100 % AM).
const FM_DEVIATION_HZ: f32 = 2500.0;
/// NBFM de-emphasis corner (6 dB/octave above it), normalised to unity at 1 kHz.
const FM_DEEMPHASIS_HZ: f32 = 300.0;

/// Demodulation: AM for airband, narrowband FM for amateur/PMR voice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    Am,
    Fm,
}

impl Mode {
    /// Channel filter width used when `--bandwidth` is not given.
    pub fn default_bandwidth(self) -> f32 {
        match self {
            Mode::Am => 10_000.0,
            Mode::Fm => 12_500.0,
        }
    }
}

/// Phase-difference FM discriminator with de-emphasis.
struct FmDemod {
    prev: Complex32,
    scale: f32,
    deemph: f32,
    alpha: f32,
    norm: f32,
}

impl FmDemod {
    fn new() -> Self {
        let fs = SAMPLE_RATE as f32;
        Self {
            prev: Complex32::new(1.0, 0.0),
            scale: 0.5 * fs / (2.0 * std::f32::consts::PI * FM_DEVIATION_HZ),
            deemph: 0.0,
            alpha: 1.0 - (-2.0 * std::f32::consts::PI * FM_DEEMPHASIS_HZ / fs).exp(),
            norm: (1.0 + (1000.0 / FM_DEEMPHASIS_HZ).powi(2)).sqrt(),
        }
    }

    fn sample(&mut self, z: Complex32) -> f32 {
        let phase_step = (z * self.prev.conj()).arg();
        self.prev = z;
        self.deemph += (phase_step * self.scale - self.deemph) * self.alpha;
        self.deemph * self.norm
    }
}

pub struct Frame {
    /// Demodulated audio, 48 kHz, roughly ±0.5 at 100 % modulation.
    pub audio: [f32; FRAME],
    /// Mean channel power over the frame, dBFS.
    pub power_db: f32,
    /// Carrier prominence, dB: spectral density at the channel centre over that at
    /// the channel edges. About 0 for noise, well above 0 when a carrier is present.
    pub carrier_db: f32,
}

/// Frames averaged for the carrier prominence.
const PROMINENCE_FRAMES: usize = 3;

/// Compares the channel's centre (where an AM carrier sits) with its edges (noise only).
struct CarrierDetector {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    buf: Vec<Complex32>,
    center_bins: usize,
    edge_bins: std::ops::RangeInclusive<usize>,
    recent: VecDeque<f32>,
}

impl CarrierDetector {
    fn new(bandwidth_hz: f32) -> Self {
        let bin_hz = SAMPLE_RATE as f32 / FRAME as f32;
        let half = bandwidth_hz / 2.0;
        Self {
            fft: FftPlanner::new().plan_fft_forward(FRAME),
            window: (0..FRAME)
                .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / FRAME as f32).cos())
                .collect(),
            buf: Vec::with_capacity(FRAME),
            // ±1 kHz (or less for narrow channels) around the carrier, which also
            // tolerates transmitter frequency offsets.
            center_bins: ((0.2 * half).min(1000.0) / bin_hz).round() as usize,
            // The outer part of the flat passband: voice is mostly below it.
            edge_bins: (0.62 * half / bin_hz).round() as usize..=(0.76 * half / bin_hz).round() as usize,
            recent: VecDeque::with_capacity(PROMINENCE_FRAMES),
        }
    }

    fn measure(&mut self, iq: &[Complex32]) -> f32 {
        self.buf.clear();
        self.buf.extend(iq.iter().zip(&self.window).map(|(z, w)| z * w));
        self.fft.process(&mut self.buf);
        let psd = |k: usize| self.buf[k].norm_sqr();
        let n = FRAME;
        let c = self.center_bins;
        let center = (psd(0) + (1..=c).map(|k| psd(k) + psd(n - k)).sum::<f32>()) / (2 * c + 1) as f32;
        let edges = self.edge_bins.clone();
        let edge_count = 2 * edges.clone().count();
        let edge = edges.map(|k| psd(k) + psd(n - k)).sum::<f32>() / edge_count as f32;

        if self.recent.len() == PROMINENCE_FRAMES {
            self.recent.pop_front();
        }
        self.recent.push_back(center / edge.max(1e-20));
        let mean = self.recent.iter().sum::<f32>() / self.recent.len() as f32;
        10.0 * mean.max(1e-6).log10()
    }
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
    iq: Vec<Complex32>,
    power_sum: f32,
    detector: CarrierDetector,
    fm: Option<FmDemod>,
}

impl Channel {
    /// `offset_hz` is the target frequency relative to the tuner centre.
    pub fn new(sample_rate: u32, offset_hz: f64, bandwidth_hz: f32, mode: Mode) -> Result<Self> {
        if !sample_rate.is_multiple_of(SAMPLE_RATE) {
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
            iq: Vec::with_capacity(FRAME),
            power_sum: 0.0,
            detector: CarrierDetector::new(bandwidth_hz),
            fm: (mode == Mode::Fm).then(FmDemod::new),
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
            let sample = match &mut self.fm {
                Some(fm) => fm.sample(z),
                None => {
                    let env = power.sqrt();
                    if self.carrier == 0.0 {
                        self.carrier = env;
                    }
                    self.carrier += (env - self.carrier) * self.carrier_alpha;
                    0.5 * (env / self.carrier.max(1e-9) - 1.0)
                }
            };
            self.audio.push(sample);
            self.iq.push(z);
            self.power_sum += power;
            if self.audio.len() == FRAME {
                let mut audio = [0.0; FRAME];
                audio.copy_from_slice(&self.audio);
                self.audio.clear();
                let power_db = 10.0 * (self.power_sum / FRAME as f32).max(1e-12).log10();
                self.power_sum = 0.0;
                let carrier_db = self.detector.measure(&self.iq);
                self.iq.clear();
                frames.push(Frame { audio, power_db, carrier_db });
            }
        }
    }
}

/// Split a decimation factor into stages of at most 10 each, largest first.
fn split_decimation(mut total: u32) -> Result<Vec<u32>> {
    let original = total;
    let mut factors = Vec::new();
    while total > 1 {
        let Some(d) = (2..=10).rev().find(|&d| total.is_multiple_of(d)) else {
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
        let mut ch = Channel::new(fs, 250_000.0, 10_000.0, Mode::Am).unwrap();
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

    /// u8 IQ of white noise only.
    fn noise_iq(seconds: f64, fs: u32) -> Vec<u8> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..(2.0 * seconds * fs as f64) as usize)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Sum of 4 uniform bytes-ish: roughly Gaussian, ±~8 LSB.
                let v: i32 = (0..4).map(|i| ((state >> (i * 8)) & 7) as i32).sum::<i32>() - 14;
                (127 + v) as u8
            })
            .collect()
    }

    #[test]
    fn carrier_prominence_separates_noise_and_carrier() {
        let fs = 2_400_000;
        let mut ch = Channel::new(fs, 250_000.0, 10_000.0, Mode::Am).unwrap();
        let mut frames = Vec::new();
        ch.process_u8(&noise_iq(0.5, fs), &mut frames);
        let noise: Vec<f32> = frames[5..].iter().map(|f| f.carrier_db).collect();
        let worst = noise.iter().copied().fold(f32::MIN, f32::max);
        assert!(worst < 4.0, "noise prominence up to {worst} dB");

        let mut ch = Channel::new(fs, 250_000.0, 10_000.0, Mode::Am).unwrap();
        let mut frames = Vec::new();
        ch.process_u8(&am_iq(fs, 250_000.0, 0.8, 0.3, 0.3), &mut frames);
        assert!(frames[10].carrier_db > 20.0, "carrier prominence {}", frames[10].carrier_db);
    }

    #[test]
    fn demodulates_offset_fm_tone() {
        let fs = 2_400_000;
        let mut ch = Channel::new(fs, 250_000.0, 12_500.0, Mode::Fm).unwrap();
        // 1 kHz tone at full (2.5 kHz) deviation.
        let n = fs as usize / 2;
        let mut phase = 0.0f64;
        let mut iq = Vec::with_capacity(2 * n);
        for i in 0..n {
            let t = i as f64 / fs as f64;
            let inst = 250_000.0 + 2500.0 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin();
            phase += 2.0 * std::f64::consts::PI * inst / fs as f64;
            for v in [0.3 * phase.cos(), 0.3 * phase.sin()] {
                iq.push((127.4 + v * 128.0).round().clamp(0.0, 255.0) as u8);
            }
        }
        let mut frames = Vec::new();
        ch.process_u8(&iq, &mut frames);
        let audio: Vec<f32> = frames[10..].iter().flat_map(|f| f.audio).collect();
        let rms = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        // ±0.5 at full deviation, de-emphasis unity at 1 kHz → RMS ≈ 0.354.
        assert!((rms - 0.354).abs() < 0.03, "rms {rms}");
        assert!(frames[20].carrier_db > 6.0, "FM carrier prominence {}", frames[20].carrier_db);
    }

    #[test]
    fn rejects_signal_outside_channel() {
        let fs = 2_400_000;
        let mut ch = Channel::new(fs, 250_000.0, 10_000.0, Mode::Am).unwrap();
        let mut frames = Vec::new();
        // Same signal 25 kHz away (one 25 kHz channel up).
        for block in am_iq(fs, 275_000.0, 0.8, 0.3, 0.3).chunks(262_144) {
            ch.process_u8(block, &mut frames);
        }
        assert!(frames[15].power_db < -9.2 - 40.0, "leak {}", frames[15].power_db);
    }
}
