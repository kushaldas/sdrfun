//! `sdrfun listen`: receive from the RTL-SDR, demodulate, clean and record.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::sync_channel;
use std::thread;

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, TimeDelta, Utc};
use clap::Args;

use crate::audio_out::AudioOut;
use crate::channel::{Channel, Frame, Mode};
use crate::clean::{CleanChain, CleanConfig, FRAME, SAMPLE_RATE};
use crate::gate::{Gate, GateConfig, GateEvent, Transmission, Verdict};
use crate::recorder::{StreamWriter, TransmissionLog, recording_path};
use crate::sdr::{Block, Device, Gain, SdrSelect};
use crate::serve::Streamer;

/// Frames between status-line refreshes (0.5 s).
const STATUS_FRAMES: usize = 50;

#[derive(Args, Debug)]
pub struct ListenArgs {
    /// Frequency to listen on, MHz
    #[arg(default_value_t = 120.150)]
    pub freq: f64,
    /// Radio to use: rtlsdr, hackrf or both (first one attached wins; env SDRFUN_SDR)
    #[arg(long, value_enum, ignore_case = true, env = "SDRFUN_SDR")]
    pub sdr: Option<SdrSelect>,
    /// RTL-SDR device index
    #[arg(long, default_value_t = 0)]
    pub device: u32,
    /// RF tuner gain in dB (snapped to the nearest supported step, 0–49.6 on the V4), or "auto".
    /// Raise it until the status line shows ADC clipping, then back off a step or two.
    #[arg(long, default_value_t = Gain::Manual(32.8))]
    pub gain: Gain,
    /// Frequency correction, ppm
    #[arg(long, default_value_t = 0, allow_negative_numbers = true)]
    pub ppm: i32,
    /// IQ sample rate, a multiple of 48000
    #[arg(long, default_value_t = 2_400_000)]
    pub sample_rate: u32,
    /// Tune this far above the target, kHz, to keep the DC spike out of the channel
    #[arg(long, default_value_t = 250.0, allow_negative_numbers = true)]
    pub offset_khz: f64,
    /// Demodulation: am (airband), nfm/fm (amateur, PMR), wfm (broadcast), usb, lsb or cw
    #[arg(long, value_enum, default_value_t = Mode::Am)]
    pub mode: Mode,
    /// Channel filter width, Hz [default: 10000 AM, 12500 NFM, 180000 WFM, 2400 USB/LSB, 500 CW]
    #[arg(long)]
    pub bandwidth: Option<f32>,
    /// Stop after this many seconds (0 = run until Ctrl-C)
    #[arg(long, default_value_t = 0.0)]
    pub duration: f64,
    /// Directory for recordings
    #[arg(long, default_value = "recordings")]
    pub out_dir: PathBuf,
    /// Also save the un-cleaned demodulated audio next to each recording
    #[arg(long, default_value_t = false)]
    pub save_raw: bool,
    /// Also record the whole session, ungated, to one file
    #[arg(long, default_value_t = false)]
    pub continuous: bool,
    /// Do not play transmissions on the speaker
    #[arg(long, default_value_t = false)]
    pub no_play: bool,
    /// Audio output: "default" or part of a device name (see `sdrfun devices`)
    #[arg(long, default_value = "default")]
    pub audio_device: String,
    /// Playback volume multiplier
    #[arg(long, default_value_t = 1.0)]
    pub volume: f32,
    /// Serve a web player (live cleaned audio + status) on this address; `--serve`
    /// alone listens on all interfaces, port 8010
    #[arg(long, value_name = "ADDR", num_args = 0..=1, default_missing_value = "0.0.0.0:8010")]
    pub serve: Option<String>,
    #[command(flatten)]
    pub gate: GateConfig,
    #[command(flatten)]
    pub clean: CleanConfig,
}

pub fn run(args: ListenArgs) -> Result<()> {
    let target_hz = args.freq * 1e6;
    let wanted_center = (target_hz + args.offset_khz * 1e3).round() as u32;

    let kind = crate::sdr::selected(SdrSelect::resolve(args.sdr))?[0];
    let mut dev = Device::open(kind, args.device)?;
    let rate = dev.set_sample_rate(args.sample_rate)?;
    dev.set_ppm(args.ppm)?;
    let gain = dev.set_gain(args.gain)?;
    let center = dev.set_center_freq(wanted_center)?;
    let bandwidth = args.bandwidth.unwrap_or(args.mode.default_bandwidth());
    let mut receiver = Receiver::new(
        Channel::new(rate, target_hz - center as f64, bandwidth, args.mode)?,
        &args.clean,
        args.gate.clone(),
    );

    eprintln!(
        "listening on {:.3} MHz {:?} (tuner {:.3} MHz, {:.2} MS/s, gain {}, bandwidth {:.1} kHz, denoiser {:?})",
        args.freq,
        args.mode,
        center as f64 / 1e6,
        rate as f64 / 1e6,
        gain.map_or("auto".into(), |g| format!("{g:.1} dB")),
        bandwidth / 1e3,
        args.clean.denoiser,
    );

    let mut speaker = if args.no_play {
        None
    } else {
        match AudioOut::open(&args.audio_device, args.volume) {
            Ok(out) => {
                eprintln!("playing on {}", out.description);
                Some(out)
            }
            Err(e) => {
                eprintln!("warning: no live playback ({e:#}); recording only");
                None
            }
        }
    };

    let mut web = match &args.serve {
        Some(addr) => {
            let (streamer, url) = Streamer::start(addr)?;
            eprintln!("web player on {url}");
            Some(streamer)
        }
        None => None,
    };

    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        ctrlc::set_handler(move || stop.store(true, Ordering::Relaxed))
            .context("installing Ctrl-C handler")?;
    }
    let (tx, rx) = sync_channel::<Block>(64);
    let reader = {
        let stop = stop.clone();
        thread::spawn(move || dev.stream(tx, stop, None))
    };

    let start = Utc::now();
    let mut session = Session::new(args.out_dir.clone(), args.save_raw)?;
    let mut continuous = args
        .continuous
        .then(|| -> Result<_> {
            let clean = StreamWriter::create(recording_path(&args.out_dir, args.freq, start, "continuous"))?;
            let raw = args
                .save_raw
                .then(|| {
                    StreamWriter::create(recording_path(&args.out_dir, args.freq, start, "continuous_raw"))
                })
                .transpose()?;
            Ok((clean, raw))
        })
        .transpose()?;

    let max_frames = (args.duration * SAMPLE_RATE as f64 / FRAME as f64).ceil() as u64;
    let mut status = Status::default();
    let mut total_frames = 0u64;

    'outer: for block in rx {
        for p in receiver.process(&block.data) {
            if let Some((clean_out, raw_out)) = &mut continuous {
                clean_out.write(&p.clean)?;
                if let Some(raw_out) = raw_out {
                    raw_out.write(&p.raw)?;
                }
            }
            match p.event {
                Some(GateEvent::Opened(so_far)) => {
                    if let Some(s) = &mut speaker {
                        s.play(&so_far);
                    }
                    if let Some(w) = &mut web {
                        w.event(serde_json::json!({"type": "open"}));
                        w.audio(&so_far);
                    }
                    status.force = true;
                }
                Some(GateEvent::Closed(tx)) => {
                    let event = tx_event(start, &tx);
                    let (_, file) = session.finish(args.freq, start, tx)?;
                    if let Some(w) = &mut web {
                        w.flush();
                        w.transmission(event, file.as_deref());
                    }
                    status.force = true;
                }
                None if p.gate_open => {
                    if let Some(s) = &mut speaker {
                        s.play(&p.clean);
                    }
                    if let Some(w) = &mut web {
                        w.audio(&p.clean);
                    }
                }
                None => {}
            }

            status.add(p.power_db);
            if status.frames >= STATUS_FRAMES || status.force {
                let adc = receiver.take_adc_stats();
                status.print(&receiver.gate, &session, adc);
                if let Some(w) = &web {
                    w.event(serde_json::json!({
                        "type": "status",
                        "freq_mhz": args.freq,
                        "mode": format!("{:?}", args.mode).to_uppercase(),
                        "channel_db": status.mean_db(),
                        "floor_db": receiver.gate.floor_db(),
                        "open": receiver.gate.is_open(),
                        "saved": session.kept,
                        "dropped": session.dropped,
                    }));
                }
                status = Status::default();
            }
            total_frames += 1;
            if max_frames > 0 && total_frames >= max_frames {
                break 'outer;
            }
        }
        if stop.load(Ordering::Relaxed) {
            break;
        }
    }
    stop.store(true, Ordering::Relaxed);

    reader
        .join()
        .map_err(|_| anyhow!("SDR reader thread panicked"))??;
    if let Some(tx) = receiver.gate.flush() {
        let event = tx_event(start, &tx);
        let (_, file) = session.finish(args.freq, start, tx)?;
        if let Some(w) = &web {
            w.transmission(event, file.as_deref());
        }
    }
    eprintln!("\r\x1b[Kdone: {} transmission(s) saved, {} dropped", session.kept, session.dropped);
    if let Some((clean_out, raw_out)) = continuous {
        eprintln!("saved {}", clean_out.finish()?.display());
        if let Some(raw) = raw_out {
            eprintln!("saved {}", raw.finish()?.display());
        }
    }
    Ok(())
}

/// Demodulator, cleanup chain and gate for one channel.
pub struct Receiver {
    channel: Channel,
    chain: CleanChain,
    pub gate: Gate,
    frames: Vec<Frame>,
    adc: AdcStats,
}

/// Raw ADC level and clipping since the last status line, to judge the RF gain.
#[derive(Default)]
struct AdcStats {
    bytes: u64,
    clipped: u64,
    power_sum: f64,
}

impl AdcStats {
    fn add(&mut self, iq: &[u8]) {
        for &b in iq {
            let x = (b as f64 - 127.4) / 128.0;
            self.power_sum += x * x;
            self.clipped += u64::from(b == 0 || b == 255);
        }
        self.bytes += iq.len() as u64;
    }
}

/// One 10 ms frame after demodulation, cleanup and gating.
pub struct Processed {
    pub raw: [f32; FRAME],
    pub clean: [f32; FRAME],
    pub power_db: f32,
    /// Carrier prominence, dB (see `channel::Frame`).
    pub carrier_db: f32,
    pub event: Option<GateEvent>,
    /// Whether the gate was open after this frame.
    pub gate_open: bool,
}

impl Receiver {
    pub fn new(channel: Channel, clean: &CleanConfig, gate: GateConfig) -> Self {
        Self {
            channel,
            chain: CleanChain::new(clean),
            gate: Gate::new(gate),
            frames: Vec::new(),
            adc: AdcStats::default(),
        }
    }

    /// ADC RMS level (dBFS, I and Q together) and clipped-sample fraction since the last call.
    pub fn take_adc_stats(&mut self) -> (f32, f32) {
        let s = std::mem::take(&mut self.adc);
        let n = s.bytes.max(1) as f64;
        (
            (10.0 * (2.0 * s.power_sum / n).max(1e-12).log10()) as f32,
            (s.clipped as f64 / n) as f32,
        )
    }

    pub fn process(&mut self, iq: &[u8]) -> Vec<Processed> {
        self.adc.add(iq);
        self.frames.clear();
        self.channel.process_u8(iq, &mut self.frames);
        self.frames
            .iter()
            .map(|frame| {
                let mut clean = frame.audio;
                let info = self.chain.process_frame(&mut clean);
                let event = self.gate.push(frame.power_db, frame.carrier_db, info, &frame.audio, &clean);
                Processed {
                    raw: frame.audio,
                    clean,
                    power_db: frame.power_db,
                    carrier_db: frame.carrier_db,
                    event,
                    gate_open: self.gate.is_open(),
                }
            })
            .collect()
    }
}

/// Saves finished transmissions and keeps the log, for one or more receivers.
pub struct Session {
    out_dir: PathBuf,
    save_raw: bool,
    log: TransmissionLog,
    pub kept: u32,
    pub dropped: u32,
}

impl Session {
    pub fn new(out_dir: PathBuf, save_raw: bool) -> Result<Self> {
        Ok(Self {
            log: TransmissionLog::open(&out_dir.join("log.jsonl"))?,
            out_dir,
            save_raw,
            kept: 0,
            dropped: 0,
        })
    }

    /// Save (or drop) a finished transmission on `freq_mhz` and log it. `start` is when
    /// the receiver that produced it saw its first frame. Returns the verdict and,
    /// if it was kept, the saved file.
    pub fn finish(
        &mut self,
        freq_mhz: f64,
        start: DateTime<Utc>,
        tx: Transmission,
    ) -> Result<(Verdict, Option<PathBuf>)> {
        let started = start + TimeDelta::milliseconds((tx.start_frame * 10) as i64);
        let seconds = tx.seconds();
        let snr = tx.peak_db - tx.floor_db;
        let mut file = None;
        if tx.verdict == Verdict::Kept {
            let tag = format!("{:03}s", seconds.round() as u32);
            let path = recording_path(&self.out_dir, freq_mhz, started, &tag);
            let mut w = StreamWriter::create(path)?;
            w.write(&tx.clean)?;
            file = Some(w.finish()?);
            if self.save_raw {
                let raw_path = recording_path(&self.out_dir, freq_mhz, started, &format!("{tag}_raw"));
                let mut w = StreamWriter::create(raw_path)?;
                w.write(&tx.raw)?;
                w.finish()?;
            }
            self.kept += 1;
        } else {
            self.dropped += 1;
        }
        self.log.append(freq_mhz, started, &tx, file.as_deref())?;

        let what = match (tx.verdict, &file) {
            (Verdict::Kept, Some(p)) => format!("saved {}", p.display()),
            (Verdict::TooShort, _) => "dropped (too short)".to_string(),
            (Verdict::SteadyTone, _) => "dropped (steady tone)".to_string(),
            _ => "dropped (no voice)".to_string(),
        };
        eprintln!(
            "\r\x1b[K{}  {freq_mhz:.3}  {seconds:5.1} s  peak {:6.1} dBFS  snr {snr:5.1} dB  voice {:3.0}%  change {:.2}  {what}",
            started.format("%H:%M:%SZ"),
            tx.peak_db,
            tx.voiced_ratio * 100.0,
            tx.spectral_change,
        );
        Ok((tx.verdict, file))
    }
}

#[derive(Default)]
struct Status {
    frames: usize,
    power_sum: f32,
    force: bool,
}

/// JSON summary of a finished transmission for the web player.
fn tx_event(start: DateTime<Utc>, tx: &Transmission) -> serde_json::Value {
    let started = start + TimeDelta::milliseconds((tx.start_frame * 10) as i64);
    let snr = tx.peak_db - tx.floor_db;
    serde_json::json!({
        "type": "tx",
        "start": started.to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        "seconds": tx.seconds(),
        "snr_db": snr.is_finite().then_some(snr),
        "voiced": tx.voiced_ratio,
        "verdict": tx.verdict.as_str(),
    })
}

impl Status {
    fn mean_db(&self) -> f32 {
        self.power_sum / self.frames.max(1) as f32
    }

    fn add(&mut self, power_db: f32) {
        self.frames += 1;
        self.power_sum += power_db;
    }

    fn print(&self, gate: &Gate, session: &Session, (adc_db, clip): (f32, f32)) {
        let mean = self.mean_db();
        let floor = gate.floor_db().unwrap_or(f32::NAN);
        let state = match gate.open_seconds() {
            Some(s) => format!("\x1b[1;31m● REC {s:5.1} s\x1b[0m"),
            None => "  idle     ".to_string(),
        };
        // Any clipping means the RF gain is too high; colour it so it stands out.
        let clip_colour = match clip {
            c if c > 0.005 => "\x1b[1;31m",
            c if c > 0.0001 => "\x1b[33m",
            _ => "",
        };
        eprint!(
            "\r\x1b[K{}  adc {adc_db:5.1} dBFS {clip_colour}clip {:5.2}%\x1b[0m  channel {mean:6.1} dBFS  floor {floor:6.1}  {state}  saved {} dropped {}",
            Utc::now().format("%H:%M:%SZ"),
            clip * 100.0,
            session.kept,
            session.dropped,
        );
        let _ = std::io::stderr().flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsig::speech;
    use clap::Parser;

    #[derive(Parser)]
    struct Wrapper {
        #[command(flatten)]
        gate: GateConfig,
        #[command(flatten)]
        clean: CleanConfig,
    }

    /// Deterministic roughly-Gaussian noise (sum of four uniforms).
    struct Noise(u64);

    impl Noise {
        fn next(&mut self) -> f64 {
            (0..4)
                .map(|_| {
                    self.0 ^= self.0 << 13;
                    self.0 ^= self.0 >> 7;
                    self.0 ^= self.0 << 17;
                    (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                })
                .sum::<f64>()
                * 3f64.sqrt()
        }
    }

    /// `modulation` (at `fs`, peaking near ±1) AM-modulated at 80 % onto a carrier at
    /// `offset` Hz, with `lead`/`trail` seconds of carrier-less noise around it, as u8 IQ.
    fn synth_iq(fs: u32, offset: f64, lead: f64, trail: f64, modulation: &[f32]) -> Vec<u8> {
        let mut noise = Noise(0x9E37_79B9_7F4A_7C15);
        let (lead_n, trail_n) = ((lead * fs as f64) as usize, (trail * fs as f64) as usize);
        let total = lead_n + modulation.len() + trail_n;
        let mut iq = Vec::with_capacity(2 * total);
        for i in 0..total {
            let carrier = match i.checked_sub(lead_n) {
                Some(j) if j < modulation.len() => 0.1 * (1.0 + 0.8 * modulation[j] as f64),
                _ => 0.0,
            };
            let ph = 2.0 * std::f64::consts::PI * offset * i as f64 / fs as f64;
            for v in [carrier * ph.cos() + 0.01 * noise.next(), carrier * ph.sin() + 0.01 * noise.next()] {
                iq.push((127.4 + v * 128.0).round().clamp(0.0, 255.0) as u8);
            }
        }
        iq
    }

    /// Run IQ through a default receiver; return the closed transmissions.
    fn receive(fs: u32, offset: f64, iq: &[u8]) -> Vec<Transmission> {
        let args = Wrapper::parse_from(["x"]);
        let channel = Channel::new(fs, offset, 10_000.0, Mode::Am).unwrap();
        let mut rx = Receiver::new(channel, &args.clean, args.gate);
        let mut closed = Vec::new();
        for block in iq.chunks(2 * 12_000) {
            for p in rx.process(block) {
                if let Some(GateEvent::Closed(tx)) = p.event {
                    closed.push(tx);
                }
            }
        }
        assert!(rx.gate.flush().is_none(), "gate should have closed after the carrier");
        closed
    }

    #[test]
    fn records_one_voice_transmission_from_iq() {
        let fs = 240_000;
        let voice_s = 8.0;
        let speech: Vec<f32> = speech(fs, voice_s).iter().map(|x| x * 2.0).collect();
        let iq = synth_iq(fs, -50_000.0, 2.0, 3.0, &speech);
        let closed = receive(fs, -50_000.0, &iq);
        assert_eq!(closed.len(), 1, "exactly one transmission");
        let t = &closed[0];
        assert_eq!(t.verdict, Verdict::Kept, "voiced ratio {}", t.voiced_ratio);
        assert!((t.carrier_s as f64 - voice_s).abs() < 0.1, "carrier {} vs {voice_s}", t.carrier_s);
        let start_s = t.start_frame as f64 * FRAME as f64 / SAMPLE_RATE as f64;
        assert!((start_s - 1.7).abs() < 0.05, "start {start_s}");
        assert!(t.peak_db - t.floor_db > 20.0, "snr {}", t.peak_db - t.floor_db);
        assert!(t.voiced_ratio > 0.5, "voiced {}", t.voiced_ratio);
        eprintln!("speech: voiced {:.2} change {:.2}", t.voiced_ratio, t.spectral_change);
    }

    #[test]
    fn records_transmission_already_on_at_start() {
        let fs = 240_000;
        let speech: Vec<f32> = speech(fs, 5.0).iter().map(|x| x * 2.0).collect();
        let closed = receive(fs, -50_000.0, &synth_iq(fs, -50_000.0, 0.0, 3.0, &speech));
        assert_eq!(closed.len(), 1);
        let t = &closed[0];
        assert_eq!(t.verdict, Verdict::Kept);
        assert_eq!(t.start_frame, 0);
        assert!((t.carrier_s - 5.0).abs() < 0.1, "carrier {}", t.carrier_s);
        assert!(t.floor_db.is_finite(), "floor learned after the carrier dropped");
    }

    #[test]
    fn drops_unmodulated_carrier() {
        let fs = 240_000;
        let iq = synth_iq(fs, -50_000.0, 2.0, 3.0, &vec![0.0; fs as usize * 4]);
        let closed = receive(fs, -50_000.0, &iq);
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].verdict, Verdict::NoVoice, "voiced {}", closed[0].voiced_ratio);
    }

    #[test]
    fn drops_carrier_modulated_by_a_steady_tone() {
        let fs = 240_000;
        let tone: Vec<f32> = (0..fs as usize * 4)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / fs as f32).sin())
            .collect();
        let iq = synth_iq(fs, -50_000.0, 2.0, 3.0, &tone);
        let closed = receive(fs, -50_000.0, &iq);
        assert_eq!(closed.len(), 1);
        eprintln!("tone: voiced {:.2} change {:.2}", closed[0].voiced_ratio, closed[0].spectral_change);
        assert_eq!(closed[0].verdict, Verdict::SteadyTone, "change {}", closed[0].spectral_change);
    }
}
