//! Voice cleanup chain shared by `listen` and `clean`:
//! band-pass → denoiser (dry/wet mix) → speech AGC + limiter, at 48 kHz in 10 ms frames.

mod agc;
mod flux;
mod rnnoise;
mod spectral;

use std::collections::VecDeque;

use clap::{Args, ValueEnum};

use crate::dsp::biquad::Cascade;

pub const SAMPLE_RATE: u32 = 48_000;
/// 10 ms at 48 kHz; also RNNoise's native frame size.
pub const FRAME: usize = 480;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum DenoiserKind {
    Rnnoise,
    Spectral,
    Off,
}

trait Denoiser: Send {
    /// Denoise one frame in place; may return a voice probability.
    fn process_frame(&mut self, frame: &mut [f32; FRAME]) -> Option<f32>;
    /// Delay introduced, in samples.
    fn latency(&self) -> usize;
}

struct Passthrough;

impl Denoiser for Passthrough {
    fn process_frame(&mut self, _frame: &mut [f32; FRAME]) -> Option<f32> {
        None
    }
    fn latency(&self) -> usize {
        0
    }
}

#[derive(Args, Clone, Debug)]
pub struct CleanConfig {
    /// High-pass corner in Hz (0 disables)
    #[arg(long, default_value_t = 250.0)]
    pub highpass: f32,
    /// Low-pass corner in Hz (0 disables)
    #[arg(long, default_value_t = 3400.0)]
    pub lowpass: f32,
    /// Noise suppression algorithm
    #[arg(long, value_enum, default_value_t = DenoiserKind::Rnnoise)]
    pub denoiser: DenoiserKind,
    /// Denoiser wet/dry mix, 0 = untouched, 1 = fully denoised
    #[arg(long, default_value_t = 1.0)]
    pub denoise_mix: f32,
    /// AGC target level in dBFS (RMS)
    #[arg(long, default_value_t = -20.0, allow_negative_numbers = true)]
    pub agc_target: f32,
    /// Maximum AGC boost in dB
    #[arg(long, default_value_t = 30.0)]
    pub agc_max_gain: f32,
    /// Disable AGC and limiter
    #[arg(long, default_value_t = false)]
    pub no_agc: bool,
}

#[derive(Debug, Default, Clone)]
pub struct ChainStats {
    /// RNNoise voice probability of every processed frame.
    pub vad: Vec<f32>,
    /// Spectral shape change of every processed frame.
    pub flux: Vec<f32>,
}

/// What the chain learned about one input frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameInfo {
    /// RNNoise voice probability, 0..1.
    pub vad: f32,
    /// Spectral shape change against 50 ms earlier, 0..2.
    pub flux: f32,
}

impl ChainStats {
    pub fn mean_vad(&self) -> Option<f64> {
        (!self.vad.is_empty())
            .then(|| self.vad.iter().map(|&v| v as f64).sum::<f64>() / self.vad.len() as f64)
    }
}

pub struct CleanChain {
    bandpass: Cascade,
    denoiser: Box<dyn Denoiser>,
    /// Detection-only RNNoise, used when the denoiser does not report voice probability.
    side_vad: Option<rnnoise::RnNoise>,
    flux: flux::SpectralFlux,
    mix: f32,
    dry_delay: VecDeque<f32>,
    agc: Option<agc::Agc>,
    pending: Vec<f32>,
    keep_trace: bool,
    pub stats: ChainStats,
}

impl CleanChain {
    pub fn new(cfg: &CleanConfig) -> Self {
        let edge = |f: f32| (f > 0.0).then_some(f);
        let denoiser: Box<dyn Denoiser> = match cfg.denoiser {
            DenoiserKind::Rnnoise => Box::new(rnnoise::RnNoise::new()),
            DenoiserKind::Spectral => Box::new(spectral::Spectral::new()),
            DenoiserKind::Off => Box::new(Passthrough),
        };
        let dry_delay = VecDeque::from(vec![0.0; denoiser.latency()]);
        Self {
            bandpass: Cascade::bandpass(SAMPLE_RATE as f32, edge(cfg.highpass), edge(cfg.lowpass)),
            side_vad: (cfg.denoiser != DenoiserKind::Rnnoise).then(rnnoise::RnNoise::new),
            flux: flux::SpectralFlux::new(),
            denoiser,
            mix: cfg.denoise_mix.clamp(0.0, 1.0),
            dry_delay,
            agc: (!cfg.no_agc).then(|| agc::Agc::new(cfg.agc_target, cfg.agc_max_gain)),
            pending: Vec::with_capacity(FRAME),
            keep_trace: false,
            stats: ChainStats::default(),
        }
    }

    /// Record the voice probability of every frame in `stats` (offline use).
    pub fn with_trace(mut self) -> Self {
        self.keep_trace = true;
        self
    }

    /// Samples of delay between input and output.
    pub fn latency(&self) -> usize {
        self.denoiser.latency()
    }

    /// Feed 48 kHz mono samples in [-1, 1]; cleaned samples are appended to `out`
    /// whenever a full 10 ms frame is available.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let mut input = input;
        while !input.is_empty() {
            let take = (FRAME - self.pending.len()).min(input.len());
            self.pending.extend_from_slice(&input[..take]);
            input = &input[take..];
            if self.pending.len() == FRAME {
                let mut frame = [0.0; FRAME];
                frame.copy_from_slice(&self.pending);
                self.pending.clear();
                self.process_frame(&mut frame);
                out.extend_from_slice(&frame);
            }
        }
    }

    /// Clean one frame in place (output delayed by `latency()`); returns what was
    /// measured on the input frame.
    pub fn process_frame(&mut self, frame: &mut [f32; FRAME]) -> FrameInfo {
        self.bandpass.process_in_place(frame);
        let flux = self.flux.process(frame);

        let mut probe = self.side_vad.is_some().then(|| *frame);
        let dry = (self.mix < 1.0).then(|| *frame);
        let mut vad = self.denoiser.process_frame(frame);
        if let (Some(side), Some(probe)) = (&mut self.side_vad, &mut probe) {
            vad = side.process_frame(probe);
        }
        let vad = vad.unwrap_or(0.0);
        if self.keep_trace {
            self.stats.vad.push(vad);
            self.stats.flux.push(flux);
        }

        if let Some(dry) = dry {
            for (x, d) in frame.iter_mut().zip(dry) {
                self.dry_delay.push_back(d);
                let delayed = self.dry_delay.pop_front().unwrap_or(0.0);
                *x = self.mix * *x + (1.0 - self.mix) * delayed;
            }
        }

        if let Some(agc) = &mut self.agc {
            agc.process_frame(frame);
        }
        FrameInfo { vad, flux }
    }
}

/// Run a whole buffer through a fresh chain, compensating latency so the
/// output lines up with (and has the same length as) the input.
pub fn clean_all(input: &[f32], cfg: &CleanConfig) -> (Vec<f32>, ChainStats) {
    let mut chain = CleanChain::new(cfg).with_trace();
    let latency = chain.latency();
    let mut out = Vec::with_capacity(input.len() + latency + FRAME);
    chain.process(input, &mut out);
    chain.process(&vec![0.0; latency + FRAME], &mut out);
    out.drain(..latency.min(out.len()));
    out.truncate(input.len());
    chain.stats.vad.truncate(input.len().div_ceil(FRAME));
    chain.stats.flux.truncate(input.len().div_ceil(FRAME));
    (out, chain.stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Wrapper {
        #[command(flatten)]
        cfg: CleanConfig,
    }

    fn cfg(args: &[&str]) -> CleanConfig {
        Wrapper::parse_from(std::iter::once("x").chain(args.iter().copied())).cfg
    }

    /// Voice-like test signal: harmonics of a pitch gliding 100 → 250 Hz (so it is not
    /// periodic), syllable-modulated.
    fn voiceish(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                let env = (0.5 + 0.5 * (2.0 * std::f32::consts::PI * 4.0 * t).sin()).powi(2);
                let phase = 2.0 * std::f32::consts::PI * (100.0 * t + 75.0 * t * t);
                let tone: f32 = (1..20)
                    .map(|h| (phase * h as f32).sin() / h as f32)
                    .sum();
                0.1 * env * tone
            })
            .collect()
    }

    fn best_lag(a: &[f32], b: &[f32], max_lag: usize) -> usize {
        (0..max_lag)
            .max_by(|&l1, &l2| {
                let c = |l: usize| -> f32 { a.iter().zip(&b[l..]).map(|(x, y)| x * y).sum() };
                c(l1).total_cmp(&c(l2))
            })
            .unwrap()
    }

    #[test]
    fn denoisers_are_latency_compensated() {
        let input = voiceish(SAMPLE_RATE as usize);
        let reference = clean_all(&input, &cfg(&["--denoiser", "off", "--no-agc"])).0;
        for kind in ["rnnoise", "spectral"] {
            let (out, _) = clean_all(&input, &cfg(&["--denoiser", kind, "--no-agc"]));
            assert_eq!(out.len(), input.len());
            let lag = best_lag(&reference[4800..40_000], &out[4800..], 1000);
            assert!(lag < 5, "{kind}: residual lag {lag}");
        }
    }

    #[test]
    fn denoise_mix_zero_matches_bypass() {
        let input = voiceish(SAMPLE_RATE as usize / 2);
        let off = clean_all(&input, &cfg(&["--denoiser", "off", "--no-agc"])).0;
        let dry = clean_all(&input, &cfg(&["--denoise-mix", "0", "--no-agc"])).0;
        let diff = off.iter().zip(&dry).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(diff < 1e-6, "diff {diff}");
    }
}
