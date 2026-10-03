//! Transmission gate: carrier squelch against an auto-tracked noise floor, with
//! pre-roll, hang time and RNNoise voice confirmation. Works on 10 ms frames.
//!
//! The floor is learned only from frames without a visible carrier, so a carrier
//! that is already on at start-up (an ATIS, say) is not mistaken for noise; until
//! the first quiet frame, carrier prominence alone opens the gate.

use std::collections::VecDeque;

use clap::Args;

use crate::clean::{FRAME, FrameInfo, SAMPLE_RATE};

const FRAMES_PER_S: f32 = SAMPLE_RATE as f32 / FRAME as f32;
/// The carrier must fall this far below the open threshold before hang time starts.
const HYSTERESIS_DB: f32 = 3.0;
/// Audio kept after the carrier drops (the rest of the hang time is trimmed).
const TAIL_S: f32 = 0.3;
/// Noise floor tracking per frame: falls quickly (~0.2 s), rises slowly (~10 s) while idle
/// and very slowly (~5 min) during a transmission, so a stuck carrier is eventually absorbed.
const FLOOR_FALL: f32 = 0.05;
const FLOOR_RISE_IDLE: f32 = 0.001;
const FLOOR_RISE_OPEN: f32 = 1.0 / 30_000.0;
/// Frames ignored after start-up while the channel filters and carrier detector settle.
const WARMUP_FRAMES: u64 = 5;

#[derive(Args, Clone, Debug)]
pub struct GateConfig {
    /// Open the gate when channel power is this many dB above the noise floor
    #[arg(long, default_value_t = 8.0)]
    pub squelch_margin: f32,
    /// Seconds the gate stays open after the carrier drops
    #[arg(long, default_value_t = 1.5)]
    pub hang: f32,
    /// Seconds of audio kept from before the gate opened
    #[arg(long, default_value_t = 0.3)]
    pub pre_roll: f32,
    /// Drop transmissions whose carrier lasted less than this many seconds
    #[arg(long, default_value_t = 0.5)]
    pub min_duration: f32,
    /// RNNoise voice probability above which a frame counts as speech
    #[arg(long, default_value_t = 0.5)]
    pub vad_threshold: f32,
    /// Keep a transmission only if at least this fraction of its carrier frames is speech
    #[arg(long, default_value_t = 0.1)]
    pub vad_ratio: f32,
    /// Drop as a steady tone (e.g. ACARS data) if the mean spectral change over speech
    /// frames is below this (0 = identical spectra, 2 = completely different)
    #[arg(long, default_value_t = 0.25)]
    pub min_spectral_change: f32,
    /// Carrier prominence (dB, channel centre over edges) above which a frame is treated
    /// as carrying a signal: such frames never train the noise floor
    #[arg(long, default_value_t = 6.0)]
    pub carrier_prominence: f32,
    /// Split transmissions longer than this many seconds (bounds memory on continuous carriers)
    #[arg(long, default_value_t = 60.0)]
    pub max_duration: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Kept,
    TooShort,
    NoVoice,
    SteadyTone,
}

pub struct Transmission {
    /// Index of the first frame (including pre-roll) since the gate was created.
    pub start_frame: u64,
    pub clean: Vec<f32>,
    pub raw: Vec<f32>,
    pub carrier_s: f32,
    pub voiced_ratio: f32,
    /// Mean spectral change over the voiced frames.
    pub spectral_change: f32,
    pub peak_db: f32,
    pub floor_db: f32,
    pub verdict: Verdict,
}

impl Transmission {
    pub fn seconds(&self) -> f32 {
        self.clean.len() as f32 / SAMPLE_RATE as f32
    }
}

pub enum GateEvent {
    /// The gate opened; carries the cleaned audio so far (pre-roll + this frame).
    Opened(Vec<f32>),
    Closed(Transmission),
}

struct Open {
    tx: Transmission,
    carrier_frames: u32,
    voiced_frames: u32,
    voiced_flux: f32,
    hang_frames: u32,
    /// Audio length at the last frame that had carrier.
    carrier_end: usize,
}

pub struct Gate {
    cfg: GateConfig,
    hang_frames: u32,
    pre_roll_frames: usize,
    max_samples: usize,
    floor: Option<f32>,
    pre_roll: VecDeque<([f32; FRAME], [f32; FRAME])>,
    open: Option<Open>,
    frame_index: u64,
}

impl Gate {
    pub fn new(cfg: GateConfig) -> Self {
        Self {
            hang_frames: (cfg.hang * FRAMES_PER_S).round().max(1.0) as u32,
            pre_roll_frames: (cfg.pre_roll * FRAMES_PER_S).round() as usize,
            max_samples: (cfg.max_duration.max(1.0) * SAMPLE_RATE as f32) as usize,
            cfg,
            floor: None,
            pre_roll: VecDeque::new(),
            open: None,
            frame_index: 0,
        }
    }

    pub fn floor_db(&self) -> Option<f32> {
        self.floor
    }

    /// Seconds of audio in the transmission currently being received.
    pub fn open_seconds(&self) -> Option<f32> {
        self.open.as_ref().map(|o| o.tx.seconds())
    }

    pub fn is_open(&self) -> bool {
        self.open.is_some()
    }

    /// Feed one frame: its channel power and carrier prominence, what the cleanup
    /// chain measured, raw and cleaned audio.
    pub fn push(
        &mut self,
        power_db: f32,
        carrier_db: f32,
        info: FrameInfo,
        raw: &[f32; FRAME],
        clean: &[f32; FRAME],
    ) -> Option<GateEvent> {
        let index = self.frame_index;
        self.frame_index += 1;
        if index < WARMUP_FRAMES {
            self.remember(raw, clean);
            return None;
        }

        let prominent = carrier_db >= self.cfg.carrier_prominence;
        // Whether this frame opens the gate, and whether it counts as carrier once open.
        let (opens, carrier) = match self.floor {
            Some(floor) => {
                let open_at = floor + self.cfg.squelch_margin;
                (power_db > open_at, power_db > open_at - HYSTERESIS_DB)
            }
            None => (prominent, prominent),
        };
        let voiced = u32::from(info.vad >= self.cfg.vad_threshold);
        let flux = if voiced == 1 { info.flux } else { 0.0 };

        let Some(open) = &mut self.open else {
            if opens {
                let mut tx = Transmission {
                    start_frame: index - self.pre_roll.len() as u64,
                    clean: Vec::new(),
                    raw: Vec::new(),
                    carrier_s: 0.0,
                    voiced_ratio: 0.0,
                    spectral_change: 0.0,
                    peak_db: power_db,
                    floor_db: self.floor.unwrap_or(f32::NAN),
                    verdict: Verdict::Kept,
                };
                for (r, c) in self.pre_roll.drain(..) {
                    tx.raw.extend_from_slice(&r);
                    tx.clean.extend_from_slice(&c);
                }
                tx.raw.extend_from_slice(raw);
                tx.clean.extend_from_slice(clean);
                let carrier_end = tx.clean.len();
                self.open = Some(Open {
                    tx,
                    carrier_frames: 1,
                    voiced_frames: voiced,
                    voiced_flux: flux,
                    hang_frames: 0,
                    carrier_end,
                });
                let so_far = self.open.as_ref().map(|o| o.tx.clean.clone()).unwrap_or_default();
                return Some(GateEvent::Opened(so_far));
            }
            if !prominent {
                self.track_floor(power_db, FLOOR_RISE_IDLE);
            }
            self.remember(raw, clean);
            return None;
        };

        open.tx.raw.extend_from_slice(raw);
        open.tx.clean.extend_from_slice(clean);
        if carrier {
            open.carrier_frames += 1;
            open.voiced_frames += voiced;
            open.voiced_flux += flux;
            open.hang_frames = 0;
            open.carrier_end = open.tx.clean.len();
            open.tx.peak_db = open.tx.peak_db.max(power_db);
        } else {
            open.hang_frames += 1;
        }
        let done = open.hang_frames >= self.hang_frames || open.tx.clean.len() >= self.max_samples;
        if !prominent {
            self.track_floor(power_db, FLOOR_RISE_OPEN);
        }
        done.then(|| GateEvent::Closed(self.close()))
    }

    /// Close any transmission in progress (e.g. on shutdown).
    pub fn flush(&mut self) -> Option<Transmission> {
        self.open.is_some().then(|| self.close())
    }

    fn close(&mut self) -> Transmission {
        let open = self.open.take().expect("gate is open");
        let mut tx = open.tx;
        let keep = (open.carrier_end + (TAIL_S * SAMPLE_RATE as f32) as usize).min(tx.clean.len());
        tx.clean.truncate(keep);
        tx.raw.truncate(keep);
        tx.carrier_s = open.carrier_frames as f32 / FRAMES_PER_S;
        tx.voiced_ratio = open.voiced_frames as f32 / open.carrier_frames.max(1) as f32;
        if tx.floor_db.is_nan() {
            tx.floor_db = self.floor.unwrap_or(f32::NAN);
        }
        tx.spectral_change = open.voiced_flux / open.voiced_frames.max(1) as f32;
        tx.verdict = if tx.carrier_s < self.cfg.min_duration {
            Verdict::TooShort
        } else if tx.voiced_ratio < self.cfg.vad_ratio {
            Verdict::NoVoice
        } else if tx.spectral_change < self.cfg.min_spectral_change {
            Verdict::SteadyTone
        } else {
            Verdict::Kept
        };
        tx
    }

    /// Follow the noise floor: down quickly, up at `rise`. The first quiet frame sets it.
    fn track_floor(&mut self, power_db: f32, rise: f32) {
        match &mut self.floor {
            Some(f) => {
                let rate = if power_db < *f { FLOOR_FALL } else { rise };
                *f += (power_db - *f) * rate;
            }
            None => self.floor = Some(power_db),
        }
    }

    fn remember(&mut self, raw: &[f32; FRAME], clean: &[f32; FRAME]) {
        if self.pre_roll_frames == 0 {
            return;
        }
        if self.pre_roll.len() == self.pre_roll_frames {
            self.pre_roll.pop_front();
        }
        self.pre_roll.push_back((*raw, *clean));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Wrapper {
        #[command(flatten)]
        cfg: GateConfig,
    }

    fn gate() -> Gate {
        Gate::new(Wrapper::parse_from(["x"]).cfg)
    }

    /// Feed `n` frames at `power` dB with voice probability `vad` and speech-like
    /// spectral change; collect events.
    fn feed(g: &mut Gate, n: usize, power: f32, vad: f32, events: &mut Vec<GateEvent>) {
        feed_flux(g, n, power, vad, 1.0, events);
    }

    fn feed_flux(g: &mut Gate, n: usize, power: f32, vad: f32, flux: f32, events: &mut Vec<GateEvent>) {
        let silence = [0.0; FRAME];
        // Prominence tracks SNR in these synthetic traces (noise sits at -65 dBFS).
        let carrier_db = power + 65.0;
        for _ in 0..n {
            events.extend(g.push(power, carrier_db, FrameInfo { vad, flux }, &silence, &silence));
        }
    }

    #[test]
    fn carrier_already_on_at_start_opens_the_gate() {
        let mut g = gate();
        let mut ev = Vec::new();
        feed(&mut g, 300, -45.0, 0.8, &mut ev);
        assert!(matches!(ev[0], GateEvent::Opened(_)), "opens right after warm-up");
        assert!(g.floor_db().is_none(), "a carrier must not train the floor");
        feed(&mut g, 200, -65.0, 0.02, &mut ev);
        let txs = closed(ev);
        assert_eq!(txs.len(), 1);
        assert_eq!(txs[0].start_frame, 0, "warm-up frames are kept as pre-roll");
        assert_eq!(txs[0].verdict, Verdict::Kept);
        // The 5 warm-up frames are audio but not counted as carrier.
        assert!((txs[0].carrier_s - 2.95).abs() < 0.02, "{}", txs[0].carrier_s);
        assert!((g.floor_db().unwrap() + 65.0).abs() < 0.5, "floor learned once quiet");
    }

    #[test]
    fn startup_transient_does_not_open_the_gate() {
        let mut g = gate();
        let mut ev = Vec::new();
        feed(&mut g, 1, -40.0, 0.9, &mut ev); // filter settling looks like a carrier
        feed(&mut g, 300, -65.0, 0.02, &mut ev);
        assert!(ev.is_empty());
        assert!((g.floor_db().unwrap() + 65.0).abs() < 0.5);
    }

    #[test]
    fn continuous_carrier_is_split_at_max_duration() {
        let mut g = gate();
        let mut ev = Vec::new();
        feed(&mut g, 100, -65.0, 0.02, &mut ev);
        feed(&mut g, 15_000, -45.0, 0.8, &mut ev); // 150 s of carrier
        let txs = closed(ev);
        assert_eq!(txs.len(), 2, "two full 60 s pieces closed so far");
        assert!(txs.iter().all(|t| (t.seconds() - 60.0).abs() < 0.5));
        assert!(g.is_open(), "third piece still recording");
        assert!((g.floor_db().unwrap() + 65.0).abs() < 0.5, "carrier did not raise the floor");
    }

    fn closed(events: Vec<GateEvent>) -> Vec<Transmission> {
        events
            .into_iter()
            .filter_map(|e| match e {
                GateEvent::Closed(t) => Some(t),
                GateEvent::Opened(_) => None,
            })
            .collect()
    }

    #[test]
    fn voice_transmission_is_kept_with_pre_roll_and_trimmed_tail() {
        let mut g = gate();
        let mut ev = Vec::new();
        feed(&mut g, 500, -65.0, 0.02, &mut ev);
        assert!(ev.is_empty());
        feed(&mut g, 300, -45.0, 0.8, &mut ev); // 3 s of speech on a strong carrier
        let GateEvent::Opened(so_far) = &ev[0] else { panic!("expected Opened") };
        assert_eq!(so_far.len(), 31 * FRAME, "pre-roll plus the opening frame");
        assert!(g.is_open());
        feed(&mut g, 200, -65.0, 0.02, &mut ev);
        assert!(!g.is_open());

        let txs = closed(ev);
        assert_eq!(txs.len(), 1);
        let t = &txs[0];
        assert_eq!(t.verdict, Verdict::Kept);
        assert_eq!(t.start_frame, 500 - 30);
        assert!((t.carrier_s - 3.0).abs() < 0.02);
        // 0.3 s pre-roll + 3 s carrier + 0.3 s tail.
        assert!((t.seconds() - 3.6).abs() < 0.02, "{}", t.seconds());
        assert!((t.floor_db + 65.0).abs() < 0.5);
        assert_eq!(t.peak_db, -45.0);
    }

    #[test]
    fn carrier_without_voice_is_dropped() {
        let mut g = gate();
        let mut ev = Vec::new();
        feed(&mut g, 300, -65.0, 0.02, &mut ev);
        feed(&mut g, 200, -45.0, 0.05, &mut ev);
        feed(&mut g, 200, -65.0, 0.02, &mut ev);
        let txs = closed(ev);
        assert_eq!(txs.len(), 1);
        assert_eq!(txs[0].verdict, Verdict::NoVoice);
    }

    #[test]
    fn voiced_but_steady_spectrum_is_dropped_as_tone() {
        let mut g = gate();
        let mut ev = Vec::new();
        feed(&mut g, 300, -65.0, 0.02, &mut ev);
        feed_flux(&mut g, 300, -45.0, 0.95, 0.05, &mut ev);
        feed(&mut g, 200, -65.0, 0.02, &mut ev);
        let txs = closed(ev);
        assert_eq!(txs[0].verdict, Verdict::SteadyTone);
        assert!((txs[0].spectral_change - 0.05).abs() < 0.01);
    }

    #[test]
    fn short_click_is_dropped() {
        let mut g = gate();
        let mut ev = Vec::new();
        feed(&mut g, 300, -65.0, 0.02, &mut ev);
        feed(&mut g, 20, -45.0, 0.9, &mut ev);
        feed(&mut g, 200, -65.0, 0.02, &mut ev);
        assert_eq!(closed(ev)[0].verdict, Verdict::TooShort);
    }

    #[test]
    fn short_fades_inside_hang_time_do_not_split_a_transmission() {
        let mut g = gate();
        let mut ev = Vec::new();
        feed(&mut g, 300, -65.0, 0.02, &mut ev);
        for _ in 0..3 {
            feed(&mut g, 100, -45.0, 0.8, &mut ev);
            feed(&mut g, 50, -66.0, 0.0, &mut ev); // 0.5 s gap < 1.5 s hang
        }
        feed(&mut g, 300, -65.0, 0.02, &mut ev);
        let txs = closed(ev);
        assert_eq!(txs.len(), 1);
        assert!((txs[0].carrier_s - 3.0).abs() < 0.02);
    }

    #[test]
    fn noise_below_margin_never_opens() {
        let mut g = gate();
        let mut ev = Vec::new();
        for i in 0..2000 {
            let wobble = if i % 2 == 0 { 3.0 } else { -3.0 };
            feed(&mut g, 1, -65.0 + wobble, 0.3, &mut ev);
        }
        assert!(ev.is_empty());
        let floor = g.floor_db().unwrap();
        assert!((-69.0..-62.0).contains(&floor), "floor {floor}");
    }

    #[test]
    fn flush_closes_open_transmission() {
        let mut g = gate();
        let mut ev = Vec::new();
        feed(&mut g, 100, -65.0, 0.0, &mut ev);
        feed(&mut g, 100, -40.0, 0.9, &mut ev);
        let t = g.flush().expect("open transmission");
        assert_eq!(t.verdict, Verdict::Kept);
        assert!(g.flush().is_none());
    }
}
