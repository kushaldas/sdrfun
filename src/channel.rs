//! One channel: IQ → frequency shift → decimate to the demodulator rate → channel filter →
//! power and carrier measurement + demodulation (AM, NFM, WFM, USB, LSB or CW), emitted as
//! 10 ms frames of 48 kHz audio.

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::{Result, bail};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};

use crate::clean::{FRAME, SAMPLE_RATE};
use crate::dsp::fir::{Decimator, lowpass};

/// Time constant of the carrier-level tracker used to normalise AM depth.
const CARRIER_TAU_S: f32 = 0.05;
const CHANNEL_TAPS: usize = 129;
/// Sideband and CW filters need much steeper skirts than the AM/FM channel filter.
const NARROW_TAPS: usize = 401;
/// Narrowband FM: peak deviation that maps to ±0.5 audio (like 100 % AM).
const FM_DEVIATION_HZ: f32 = 2500.0;
/// NBFM de-emphasis corner (6 dB/octave above it), normalised to unity at 1 kHz.
const FM_DEEMPHASIS_HZ: f32 = 300.0;
/// Broadcast FM peak deviation.
const WFM_DEVIATION_HZ: f32 = 75_000.0;
/// Broadcast FM de-emphasis time constant: 50 µs in Europe, 75 µs in the Americas.
pub const WFM_DEEMPHASIS_US: f32 = 50.0;
/// Broadcast FM: demodulate at this rate, then keep the mono audio up to 15 kHz.
const WFM_RATE: u32 = 240_000;
const WFM_AUDIO_HZ: f32 = 15_000.0;
/// Sidebands start this far from the suppressed carrier.
const SSB_LOW_HZ: f32 = 300.0;
/// CW beat note for a signal exactly on frequency.
pub const CW_PITCH_HZ: f32 = 700.0;
/// Sideband/CW level tracker (normalises audio before the AGC): fast attack, slow release.
const SSB_ATTACK_S: f32 = 0.002;
const SSB_RELEASE_S: f32 = 0.5;

/// Demodulation: AM for airband, NFM for amateur/PMR voice, WFM for broadcast,
/// USB/LSB for sideband voice and CW for Morse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, clap::ValueEnum)]
pub enum Mode {
    Am,
    #[value(alias = "fm")]
    Nfm,
    Wfm,
    Usb,
    Lsb,
    Cw,
}

impl Mode {
    pub const ALL: [Mode; 6] = [Mode::Am, Mode::Nfm, Mode::Wfm, Mode::Usb, Mode::Lsb, Mode::Cw];

    /// Lower-case name, as on the command line and in the web protocol.
    pub fn name(self) -> &'static str {
        match self {
            Mode::Am => "am",
            Mode::Nfm => "nfm",
            Mode::Wfm => "wfm",
            Mode::Usb => "usb",
            Mode::Lsb => "lsb",
            Mode::Cw => "cw",
        }
    }

    pub fn from_name(name: &str) -> Option<Mode> {
        match name.to_ascii_lowercase().as_str() {
            "fm" => Some(Mode::Nfm),
            n => Mode::ALL.into_iter().find(|m| m.name() == n),
        }
    }

    /// Channel filter width used when `--bandwidth` is not given.
    pub fn default_bandwidth(self) -> f32 {
        match self {
            Mode::Am => 10_000.0,
            Mode::Nfm => 12_500.0,
            Mode::Wfm => 180_000.0,
            Mode::Usb | Mode::Lsb => 2_400.0,
            Mode::Cw => 500.0,
        }
    }

    /// Accepted channel filter widths, Hz.
    pub fn bandwidth_range(self) -> std::ops::RangeInclusive<f32> {
        match self {
            Mode::Am | Mode::Nfm => 1_000.0..=40_000.0,
            Mode::Wfm => 50_000.0..=220_000.0,
            Mode::Usb | Mode::Lsb => 1_000.0..=4_000.0,
            Mode::Cw => 100.0..=2_000.0,
        }
    }

    /// Sample rate the demodulator runs at.
    pub fn demod_rate(self) -> u32 {
        match self {
            Mode::Wfm => WFM_RATE,
            _ => SAMPLE_RATE,
        }
    }

    /// Tuning step: the channel grid tap-to-tune snaps to.
    pub fn step_hz(self) -> f64 {
        match self {
            Mode::Am => 5_000.0,
            Mode::Nfm => 6_250.0,
            Mode::Wfm => 100_000.0,
            Mode::Usb | Mode::Lsb => 100.0,
            Mode::Cw => 10.0,
        }
    }

    /// Audio band (high-pass, low-pass corners in Hz) kept after demodulation.
    pub fn audio_band(self) -> (f32, f32) {
        match self {
            Mode::Am => (150.0, 4_000.0),
            // Above CTCSS tones.
            Mode::Nfm => (300.0, 3_400.0),
            Mode::Wfm => (30.0, WFM_AUDIO_HZ),
            Mode::Usb | Mode::Lsb => (200.0, 3_000.0),
            Mode::Cw => (300.0, 1_500.0),
        }
    }

    /// Default squelch, dB over the noise floor (`None` = open).
    pub fn default_squelch(self) -> Option<f32> {
        matches!(self, Mode::Am | Mode::Nfm).then_some(6.0)
    }

    /// Whether voice cleanup is on by default.
    pub fn default_cleanup(self) -> bool {
        matches!(self, Mode::Am | Mode::Nfm)
    }
}

/// Phase-difference FM discriminator with single-pole de-emphasis, normalised to unity
/// gain at 1 kHz.
struct FmDemod {
    prev: Complex32,
    scale: f32,
    deemph: f32,
    alpha: f32,
    norm: f32,
}

impl FmDemod {
    fn new(fs: f32, deviation_hz: f32, deemphasis_hz: f32) -> Self {
        Self {
            prev: Complex32::new(1.0, 0.0),
            scale: 0.5 * fs / (2.0 * std::f32::consts::PI * deviation_hz),
            deemph: 0.0,
            alpha: 1.0 - (-2.0 * std::f32::consts::PI * deemphasis_hz / fs).exp(),
            norm: (1.0 + (1000.0 / deemphasis_hz).powi(2)).sqrt(),
        }
    }

    fn sample(&mut self, z: Complex32) -> f32 {
        let phase_step = (z * self.prev.conj()).arg();
        self.prev = z;
        self.deemph += (phase_step * self.scale - self.deemph) * self.alpha;
        self.deemph * self.norm
    }
}

/// Numerically controlled oscillator.
struct Nco {
    phasor: Complex32,
    rotation: Complex32,
}

impl Nco {
    fn new(hz: f64, fs: f64) -> Self {
        let mut nco = Self { phasor: Complex32::new(1.0, 0.0), rotation: Complex32::default() };
        nco.set(hz, fs);
        nco
    }

    fn set(&mut self, hz: f64, fs: f64) {
        let w = 2.0 * std::f64::consts::PI * hz / fs;
        self.rotation = Complex32::new(w.cos() as f32, w.sin() as f32);
    }

    #[inline]
    fn next(&mut self) -> Complex32 {
        let p = self.phasor;
        self.phasor *= self.rotation;
        p
    }

    /// Keep the oscillator on the unit circle.
    fn renormalize(&mut self) {
        self.phasor /= self.phasor.norm();
    }
}

/// Sideband / CW: shift the wanted passband to 0 Hz, filter, shift to the audio
/// frequency and keep the real part, normalised by a tracked signal level.
struct Product {
    down: Option<Nco>,
    up: Nco,
    level: f32,
    attack: f32,
    release: f32,
    shifted: Vec<Complex32>,
}

impl Product {
    fn new(mode: Mode, bandwidth_hz: f32, fs: f64) -> Self {
        let center = (SSB_LOW_HZ + bandwidth_hz / 2.0) as f64;
        let (down, up) = match mode {
            Mode::Usb => (Some(-center), center),
            Mode::Lsb => (Some(center), -center),
            _ => (None, CW_PITCH_HZ as f64),
        };
        let coeff = |tau: f32| 1.0 - (-1.0 / (fs as f32 * tau)).exp();
        Self {
            down: down.map(|hz| Nco::new(hz, fs)),
            up: Nco::new(up, fs),
            level: 0.0,
            attack: coeff(SSB_ATTACK_S),
            release: coeff(SSB_RELEASE_S),
            shifted: Vec::new(),
        }
    }

    /// Frequency shift applied before the channel filter.
    fn shift<'a>(&'a mut self, input: &'a [Complex32]) -> &'a [Complex32] {
        let Some(down) = &mut self.down else { return input };
        self.shifted.clear();
        self.shifted.extend(input.iter().map(|&z| z * down.next()));
        down.renormalize();
        &self.shifted
    }

    fn sample(&mut self, z: Complex32) -> f32 {
        let power = z.norm_sqr();
        let coeff = if power > self.level { self.attack } else { self.release };
        self.level += (power - self.level) * coeff;
        let y = (z * self.up.next()).re;
        0.3 * y / self.level.max(1e-12).sqrt()
    }
}

/// How a channel turns filtered IQ into audio.
enum Demod {
    Am,
    Fm(FmDemod),
    /// FM at `WFM_RATE`, then a low-pass decimator down to 48 kHz audio.
    Wfm(FmDemod, Decimator, Vec<Complex32>),
    Product(Product),
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
/// Works on 10 ms of channel IQ, so its bins are 100 Hz wide at any rate.
struct CarrierDetector {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    buf: Vec<Complex32>,
    center_bins: usize,
    edge_bins: std::ops::RangeInclusive<usize>,
    recent: VecDeque<f32>,
}

impl CarrierDetector {
    fn new(bandwidth_hz: f32, fs: u32) -> Self {
        let n = fs as usize / 100;
        let bin_hz = 100.0;
        let half = bandwidth_hz / 2.0;
        Self {
            fft: FftPlanner::new().plan_fft_forward(n),
            window: (0..n)
                .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
                .collect(),
            buf: Vec::with_capacity(n),
            // ±1 kHz (or less for narrow channels) around the carrier, which also
            // tolerates transmitter frequency offsets.
            center_bins: ((0.2 * half).min(1000.0) / bin_hz).round().max(1.0) as usize,
            // The outer part of the flat passband: voice is mostly below it.
            edge_bins: (0.62 * half / bin_hz).round().max(2.0) as usize
                ..=(0.76 * half / bin_hz).round().max(3.0) as usize,
            recent: VecDeque::with_capacity(PROMINENCE_FRAMES),
        }
    }

    fn measure(&mut self, iq: &[Complex32]) -> f32 {
        let n = self.window.len();
        self.buf.clear();
        self.buf.extend(iq.iter().take(n).zip(&self.window).map(|(z, w)| z * w));
        self.buf.resize(n, Complex32::default());
        self.fft.process(&mut self.buf);
        let psd = |k: usize| self.buf[k].norm_sqr();
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
    sample_rate: u32,
    mode: Mode,
    bandwidth_hz: f32,
    nco: Nco,
    stages: Vec<Decimator>,
    channel_filter: Decimator,
    carrier: f32,
    carrier_alpha: f32,
    scratch: [Vec<Complex32>; 2],
    audio: Vec<f32>,
    iq: Vec<Complex32>,
    power_sum: f32,
    detector: CarrierDetector,
    demod: Demod,
    /// Channel-rate IQ (before the channel filter) kept for a zoomed spectrum.
    tap: Option<Vec<Complex32>>,
}

impl Channel {
    /// `offset_hz` is the target frequency relative to the tuner centre.
    pub fn new(sample_rate: u32, offset_hz: f64, bandwidth_hz: f32, mode: Mode) -> Result<Self> {
        let fs = mode.demod_rate();
        if !sample_rate.is_multiple_of(fs) {
            bail!("sample rate {sample_rate} must be a multiple of {fs} for {}", mode.name());
        }
        let range = mode.bandwidth_range();
        if !range.contains(&bandwidth_hz) {
            bail!(
                "bandwidth {bandwidth_hz} Hz must be between {} and {} Hz for {}",
                range.start(),
                range.end(),
                mode.name()
            );
        }
        check_offset(sample_rate, offset_hz, bandwidth_hz)?;
        let factors = split_decimation(sample_rate / fs)?;

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

        let fs_f = fs as f32;
        let (taps, cutoff) = match mode {
            Mode::Usb | Mode::Lsb | Mode::Cw => (NARROW_TAPS, bandwidth_hz / 2.0),
            _ => (CHANNEL_TAPS, bandwidth_hz / 2.0),
        };
        let demod = match mode {
            Mode::Am => Demod::Am,
            Mode::Nfm => Demod::Fm(FmDemod::new(fs_f, FM_DEVIATION_HZ, FM_DEEMPHASIS_HZ)),
            Mode::Wfm => Demod::Wfm(
                FmDemod::new(fs_f, WFM_DEVIATION_HZ, deemphasis_corner(WFM_DEEMPHASIS_US)),
                Decimator::new(lowpass(81, WFM_AUDIO_HZ, fs_f), (fs / SAMPLE_RATE) as usize),
                Vec::new(),
            ),
            Mode::Usb | Mode::Lsb | Mode::Cw => Demod::Product(Product::new(mode, bandwidth_hz, fs as f64)),
        };

        Ok(Self {
            sample_rate,
            mode,
            bandwidth_hz,
            nco: Nco::new(-offset_hz, sample_rate as f64),
            stages,
            channel_filter: Decimator::new(lowpass(taps, cutoff, fs_f), 1),
            carrier: 0.0,
            carrier_alpha: 1.0 - (-1.0 / (fs_f * CARRIER_TAU_S)).exp(),
            scratch: [Vec::new(), Vec::new()],
            audio: Vec::with_capacity(FRAME),
            iq: Vec::with_capacity(fs as usize / 100),
            power_sum: 0.0,
            detector: CarrierDetector::new(bandwidth_hz, fs),
            demod,
            tap: None,
        })
    }

    /// Broadcast FM de-emphasis time constant in µs (no effect in other modes).
    pub fn with_deemphasis(mut self, tau_us: f32) -> Self {
        if let Demod::Wfm(fm, ..) = &mut self.demod {
            *fm = FmDemod::new(WFM_RATE as f32, WFM_DEVIATION_HZ, deemphasis_corner(tau_us));
        }
        self
    }

    /// Retune within the captured band without rebuilding the filters.
    pub fn set_offset(&mut self, offset_hz: f64) -> Result<()> {
        check_offset(self.sample_rate, offset_hz, self.bandwidth_hz)?;
        self.nco.set(-offset_hz, self.sample_rate as f64);
        Ok(())
    }

    /// Start keeping the channel-rate IQ (before the channel filter) for `take_tap`.
    pub fn enable_tap(&mut self) {
        self.tap.get_or_insert_with(Vec::new);
    }

    /// Sample rate of the IQ returned by `take_tap`.
    pub fn tap_rate(&self) -> u32 {
        self.mode.demod_rate()
    }

    /// Move the IQ kept since the last call into `out`.
    pub fn take_tap(&mut self, out: &mut Vec<Complex32>) {
        if let Some(tap) = &mut self.tap {
            out.append(tap);
        }
    }

    /// Process raw interleaved u8 IQ from the dongle.
    pub fn process_u8(&mut self, iq: &[u8], frames: &mut Vec<Frame>) {
        let [a, b] = &mut self.scratch;
        a.clear();
        a.extend(iq.chunks_exact(2).map(|p| {
            let x = Complex32::new((p[0] as f32 - 127.4) / 128.0, (p[1] as f32 - 127.4) / 128.0);
            x * self.nco.next()
        }));
        self.nco.renormalize();

        for stage in &mut self.stages {
            b.clear();
            stage.process(a, b);
            std::mem::swap(a, b);
        }
        if let Some(tap) = &mut self.tap {
            // Bounded, in case nobody collects it.
            if tap.len() < self.mode.demod_rate() as usize {
                tap.extend_from_slice(a);
            }
        }
        b.clear();
        match &mut self.demod {
            Demod::Product(p) => self.channel_filter.process(p.shift(a), b),
            _ => self.channel_filter.process(a, b),
        }

        let frame_iq = self.mode.demod_rate() as usize / 100;
        for &z in b.iter() {
            let power = z.norm_sqr();
            match &mut self.demod {
                Demod::Am => {
                    let env = power.sqrt();
                    if self.carrier == 0.0 {
                        self.carrier = env;
                    }
                    self.carrier += (env - self.carrier) * self.carrier_alpha;
                    self.audio.push(0.5 * (env / self.carrier.max(1e-9) - 1.0));
                }
                Demod::Fm(fm) => self.audio.push(fm.sample(z)),
                Demod::Wfm(fm, decim, out) => {
                    out.clear();
                    decim.process(&[Complex32::new(fm.sample(z), 0.0)], out);
                    self.audio.extend(out.iter().map(|c| c.re));
                }
                Demod::Product(p) => self.audio.push(p.sample(z)),
            }
            if self.iq.len() < frame_iq {
                self.iq.push(z);
            }
            self.power_sum += power;
            if self.audio.len() == FRAME {
                let mut audio = [0.0; FRAME];
                audio.copy_from_slice(&self.audio);
                self.audio.clear();
                let n = self.iq.len().max(1);
                let power_db = 10.0 * (self.power_sum / n as f32).max(1e-12).log10();
                self.power_sum = 0.0;
                let carrier_db = self.detector.measure(&self.iq);
                self.iq.clear();
                frames.push(Frame { audio, power_db, carrier_db });
            }
        }
    }
}

fn check_offset(sample_rate: u32, offset_hz: f64, bandwidth_hz: f32) -> Result<()> {
    if offset_hz.abs() + bandwidth_hz as f64 / 2.0 >= sample_rate as f64 / 2.0 {
        bail!("offset {offset_hz} Hz puts the channel outside the captured band");
    }
    Ok(())
}

/// Corner frequency of a de-emphasis time constant in µs.
fn deemphasis_corner(tau_us: f32) -> f32 {
    1.0 / (2.0 * std::f32::consts::PI * tau_us * 1e-6)
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
        let mut ch = Channel::new(fs, 250_000.0, 12_500.0, Mode::Nfm).unwrap();
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
    /// u8 IQ of a constant-amplitude complex signal with instantaneous frequency `freq(t)`.
    fn fm_iq(fs: u32, seconds: f64, amp: f64, freq: impl Fn(f64) -> f64) -> Vec<u8> {
        let n = (fs as f64 * seconds) as usize;
        let mut phase = 0.0f64;
        let mut out = Vec::with_capacity(2 * n);
        for i in 0..n {
            phase += 2.0 * std::f64::consts::PI * freq(i as f64 / fs as f64) / fs as f64;
            for v in [amp * phase.cos(), amp * phase.sin()] {
                out.push((127.4 + v * 128.0).round().clamp(0.0, 255.0) as u8);
            }
        }
        out
    }

    fn run(ch: &mut Channel, iq: &[u8]) -> Vec<Frame> {
        let mut frames = Vec::new();
        for block in iq.chunks(262_144) {
            ch.process_u8(block, &mut frames);
        }
        frames
    }

    /// Fraction of the audio's power that is a sine at `f` Hz (1.0 for a pure tone).
    fn tone_fraction(audio: &[f32], f: f32) -> f32 {
        let w = 2.0 * std::f32::consts::PI * f / SAMPLE_RATE as f32;
        let (mut re, mut im, mut total) = (0.0f64, 0.0f64, 0.0f64);
        for (i, &x) in audio.iter().enumerate() {
            re += (x * (w * i as f32).cos()) as f64;
            im += (x * (w * i as f32).sin()) as f64;
            total += (x * x) as f64;
        }
        (2.0 * (re * re + im * im) / (audio.len() as f64 * total)) as f32
    }

    fn rms(audio: &[f32]) -> f32 {
        (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt()
    }

    #[test]
    fn sidebands_demodulate_to_audio_and_reject_the_other_side() {
        let fs = 240_000;
        let offset = 50_000.0;
        for (mode, side) in [(Mode::Usb, 1.0), (Mode::Lsb, -1.0)] {
            let bw = mode.default_bandwidth();
            let wanted = fm_iq(fs, 0.5, 0.3, |_| offset + side * 1000.0);
            let mut ch = Channel::new(fs, offset, bw, mode).unwrap();
            let frames = run(&mut ch, &wanted);
            let audio: Vec<f32> = frames[20..].iter().flat_map(|f| f.audio).collect();
            let frac = tone_fraction(&audio, 1000.0);
            assert!(frac > 0.95, "{mode:?}: 1 kHz fraction {frac}");
            assert!(rms(&audio) > 0.1, "{mode:?}: rms {}", rms(&audio));

            let opposite = fm_iq(fs, 0.5, 0.3, |_| offset - side * 1000.0);
            let mut ch = Channel::new(fs, offset, bw, mode).unwrap();
            let other = run(&mut ch, &opposite);
            let rejection = frames[30].power_db - other[30].power_db;
            assert!(rejection > 30.0, "{mode:?}: opposite sideband only {rejection} dB down");
        }
    }

    #[test]
    fn cw_carrier_beats_at_the_pitch() {
        let fs = 240_000;
        let mut ch = Channel::new(fs, 30_000.0, 500.0, Mode::Cw).unwrap();
        let frames = run(&mut ch, &fm_iq(fs, 0.5, 0.3, |_| 30_000.0));
        let audio: Vec<f32> = frames[20..].iter().flat_map(|f| f.audio).collect();
        let frac = tone_fraction(&audio, CW_PITCH_HZ);
        assert!(frac > 0.95, "pitch fraction {frac}");
    }

    #[test]
    fn wfm_recovers_tone_with_deemphasis() {
        let fs = 960_000;
        let tone = |f: f64| {
            move |t: f64| 200_000.0 + 75_000.0 * (2.0 * std::f64::consts::PI * f * t).sin()
        };
        let mut ch = Channel::new(fs, 200_000.0, 180_000.0, Mode::Wfm).unwrap();
        let frames = run(&mut ch, &fm_iq(fs, 0.3, 0.3, tone(1000.0)));
        assert_eq!(frames.len(), 30);
        let audio: Vec<f32> = frames[10..].iter().flat_map(|f| f.audio).collect();
        // Full deviation → ±0.5, unity de-emphasis gain at 1 kHz.
        assert!((rms(&audio) - 0.354).abs() < 0.03, "1 kHz rms {}", rms(&audio));
        assert!(tone_fraction(&audio, 1000.0) > 0.95);

        let mut ch = Channel::new(fs, 200_000.0, 180_000.0, Mode::Wfm).unwrap();
        let frames = run(&mut ch, &fm_iq(fs, 0.3, 0.3, tone(5000.0)));
        let audio: Vec<f32> = frames[10..].iter().flat_map(|f| f.audio).collect();
        // 50 µs: corner 3183 Hz, so 5 kHz is down to 1.048 / 1.862 of 1 kHz.
        assert!((rms(&audio) - 0.199).abs() < 0.02, "5 kHz rms {}", rms(&audio));
    }

    #[test]
    fn set_offset_retunes_in_place() {
        let fs = 240_000;
        let mut ch = Channel::new(fs, 50_000.0, 10_000.0, Mode::Am).unwrap();
        let first = run(&mut ch, &am_iq(fs, 50_000.0, 0.8, 0.3, 0.2));
        ch.set_offset(-60_000.0).unwrap();
        let second = run(&mut ch, &am_iq(fs, -60_000.0, 0.8, 0.3, 0.3));
        assert!((first[15].power_db + 9.2).abs() < 1.0);
        let audio: Vec<f32> = second[10..].iter().flat_map(|f| f.audio).collect();
        assert!((rms(&audio) - 0.283).abs() < 0.02, "rms after retune {}", rms(&audio));
        assert!(ch.set_offset(119_000.0).is_err());
    }

    #[test]
    fn mode_names_round_trip() {
        for m in Mode::ALL {
            assert_eq!(Mode::from_name(m.name()), Some(m));
        }
        assert_eq!(Mode::from_name("FM"), Some(Mode::Nfm));
        assert_eq!(Mode::from_name("dmr"), None);
    }
}
