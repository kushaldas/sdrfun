//! `sdrfun web`: an interactive receiver for phones, in the spirit of OpenWebRX. The page
//! shows a waterfall of the whole captured band, and any connected client can tune, change
//! mode, squelch, voice cleanup, gain and span; every client hears the same receiver and
//! sees the same state. Nothing is recorded.
//!
//! WebSocket messages, server → client (see also `serve.rs`):
//! - text JSON: `state` (full receiver state after every change), `bookmarks`, `meter`
//!   (level, floor, squelch open, ADC clipping; 10 a second) and `error`.
//! - binary, first byte = type: `1` audio (IMA ADPCM chunk of `STREAM_RATE` mono, see
//!   `adpcm.rs`); `2` wideband and `3` detail waterfall row: `ROW_HEADER` bytes (centre Hz
//!   u32, span Hz u32, lo dB f32, hi dB f32, bins u16, all LE) then an ADPCM chunk of
//!   `bins` levels (0..255 scaled by `ROW_SCALE`).
//!
//! Client → server: JSON with `cmd` = `tune` {hz, mode?, bandwidth?, squelch?}, `mode`
//! {mode}, `bandwidth` {hz}, `squelch` {db | null}, `cleanup` {on}, `gain` {db | "auto"},
//! `span` {rate}, `bookmark_add` {name, hz, mode, bandwidth?, squelch?}, `bookmark_del`
//! {id}, and `lowdata` {on} (handled per client in `serve.rs`).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel, sync_channel};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;
use rustfft::num_complex::Complex32;
use serde_json::{Value, json};

use crate::adpcm;
use crate::audio_out::AudioOut;
use crate::bookmarks::{Bookmark, Bookmarks};
use crate::channel::{Channel, Frame, Mode, WFM_DEEMPHASIS_US};
use crate::clean::{BasicChain, CleanChain, CleanConfig, SAMPLE_RATE};
use crate::dsp::resample::Resampler;
use crate::scan::{DC_GUARD_HZ, USABLE_FRACTION};
use crate::sdr::{Block, Control, Device, Gain};
use crate::serve::{CHUNK, Hub, STREAM_RATE, to_i16};
use crate::spectrum::{Levels, Spectrum};
use crate::squelch::Squelch;

const PAGE: &str = include_str!("web.html");
/// IQ rates offered as the waterfall span; all are multiples of 240 kHz (WFM) and 48 kHz.
pub const SPANS: [u32; 4] = [2_400_000, 1_920_000, 1_200_000, 960_000];
const WIDE_BINS: usize = 2048;
const DETAIL_BINS: usize = 1024;
const ROWS_PER_S: u32 = 10;
/// Frames between meter messages (100 ms).
const METER_FRAMES: usize = 10;
pub const ROW_HEADER: usize = 19;
/// Carrier prominence (see `channel::Frame`) that counts as a carrier, as `--carrier-prominence`
/// for the gate: noise measures below 4 dB.
const CARRIER_PROMINENCE_DB: f32 = 6.0;
/// After a gain or rate change, learn the noise floor from scratch for this many frames
/// (200 ms), until the tuner and filters have settled.
const RELEARN_FRAMES: u32 = 20;
/// Waterfall levels are sent as ADPCM samples of `level * ROW_SCALE`.
pub const ROW_SCALE: i16 = 64;
/// Tunable range: HF needs the V4's upconverter (handled by its driver), the R828D
/// tuner tops out at 1766 MHz.
const MIN_HZ: f64 = 100e3;
const MAX_HZ: f64 = 1766e6;
/// Agc for the basic (no cleanup) chain.
const AGC_TARGET_DBFS: f32 = -20.0;
const AGC_MAX_GAIN_DB: f32 = 40.0;

#[derive(Args, Debug)]
pub struct WebArgs {
    /// Frequency to start on, MHz
    #[arg(default_value_t = 145.500)]
    pub freq: f64,
    /// Mode to start in: am, nfm, wfm, usb, lsb or cw
    #[arg(long, value_enum, default_value_t = Mode::Nfm)]
    pub mode: Mode,
    /// RTL-SDR device index
    #[arg(long, default_value_t = 0)]
    pub device: u32,
    /// RF tuner gain in dB, or "auto" (changeable from the page)
    #[arg(long, default_value_t = Gain::Manual(32.8))]
    pub gain: Gain,
    /// Frequency correction, ppm
    #[arg(long, default_value_t = 0, allow_negative_numbers = true)]
    pub ppm: i32,
    /// IQ sample rate (= waterfall span), a multiple of 240000 (changeable from the page)
    #[arg(long, default_value_t = 2_400_000)]
    pub sample_rate: u32,
    /// Address to serve the page on
    #[arg(long, value_name = "ADDR", default_value = "0.0.0.0:8010")]
    pub serve: String,
    /// Bookmarks file, created with defaults if missing
    #[arg(long, default_value = "bookmarks.json")]
    pub bookmarks: PathBuf,
    /// Broadcast FM de-emphasis, µs: 50 (Europe) or 75 (Americas)
    #[arg(long, default_value_t = WFM_DEEMPHASIS_US)]
    pub deemphasis: f32,
    /// Do not also play on this machine's speaker
    #[arg(long, default_value_t = false)]
    pub no_play: bool,
    /// Audio output: "default" or part of a device name (see `sdrfun devices`)
    #[arg(long, default_value = "default")]
    pub audio_device: String,
    /// Speaker volume multiplier
    #[arg(long, default_value_t = 1.0)]
    pub volume: f32,
    #[command(flatten)]
    pub clean: CleanConfig,
}

/// Tuner centre for listening on `hz`, or `None` if the current `center` already works:
/// the channel must be inside the usable part of the band and clear of the DC spike.
/// A new centre puts the channel a sixth of the span from it, on the side the channel
/// moved to, so tuning onwards in that direction has room before the next hop.
pub fn plan_center(hz: f64, center: f64, rate: u32, bandwidth: f32) -> Option<f64> {
    let half_bw = bandwidth as f64 / 2.0;
    let reach = rate as f64 * USABLE_FRACTION / 2.0 - half_bw;
    let guard = DC_GUARD_HZ.max(half_bw + 5_000.0);
    let offset = hz - center;
    if offset.abs() <= reach && offset.abs() >= guard {
        return None;
    }
    let dir = if offset >= 0.0 { 1.0 } else { -1.0 };
    Some((hz - dir * rate as f64 / 6.0).round().max(rate as f64 / 2.0))
}

/// What one block produced.
#[derive(Default)]
pub struct Output {
    /// 48 kHz audio after squelch (silence while closed), for the speaker and the stream.
    pub audio: Vec<f32>,
    /// Binary waterfall messages.
    pub rows: Vec<Vec<u8>>,
    pub meters: Vec<Value>,
}

/// The receiver behind the page: owns the DSP and the settings clients can change.
pub struct Radio {
    rate: u32,
    /// Where the tuner is, or was last asked to go.
    center_hz: f64,
    tuned_hz: f64,
    mode: Mode,
    bandwidth: f32,
    cleanup: bool,
    gain: Gain,
    /// Supported tuner gains, dB.
    gains: Vec<f32>,
    deemphasis: f32,
    channel: Channel,
    /// Tuner centre and rate the channel's offset was computed for.
    channel_at: (u32, u32),
    clean: CleanChain,
    basic: BasicChain,
    squelch: Squelch,
    wide: Spectrum,
    wide_levels: Levels,
    wide_enc: adpcm::Encoder,
    detail: Spectrum,
    detail_levels: Levels,
    detail_enc: adpcm::Encoder,
    tap: Vec<Complex32>,
    frames: Vec<Frame>,
    rows: Vec<Vec<f32>>,
    meter_frames: usize,
    /// `Block::settings` of the last block, to relearn the noise floor after a gain or
    /// rate change has reached the IQ.
    settings: u32,
    /// Frames left during which the squelch keeps relearning the floor.
    relearn: u32,
    /// ADC samples at the rails, and all samples, since the last meter message.
    clipped: (u64, u64),
}

impl Radio {
    pub fn new(
        rate: u32,
        tuned_hz: f64,
        mode: Mode,
        gain: Gain,
        gains: Vec<f32>,
        deemphasis: f32,
        clean_cfg: CleanConfig,
    ) -> Result<Self> {
        if !rate.is_multiple_of(Mode::Wfm.demod_rate()) {
            bail!("sample rate {rate} must be a multiple of {}", Mode::Wfm.demod_rate());
        }
        let bandwidth = mode.default_bandwidth();
        let center_hz = plan_center(tuned_hz, tuned_hz, rate, bandwidth).unwrap_or(tuned_hz);
        let channel = build_channel(rate, center_hz, tuned_hz, mode, bandwidth, deemphasis)?;
        let (hp, lp) = mode.audio_band();
        Ok(Self {
            rate,
            center_hz,
            tuned_hz,
            mode,
            bandwidth,
            cleanup: mode.default_cleanup(),
            gain,
            gains,
            deemphasis,
            clean: CleanChain::new(&clean_cfg),
            channel,
            channel_at: (center_hz as u32, rate),
            basic: BasicChain::new(hp, lp, AGC_TARGET_DBFS, AGC_MAX_GAIN_DB),
            squelch: Squelch::new(mode.default_squelch()),
            wide: Spectrum::new(WIDE_BINS, rate, ROWS_PER_S),
            wide_levels: Levels::default(),
            wide_enc: adpcm::Encoder::default(),
            detail: Spectrum::new(DETAIL_BINS, mode.demod_rate(), ROWS_PER_S),
            detail_levels: Levels::default(),
            detail_enc: adpcm::Encoder::default(),
            tap: Vec::new(),
            frames: Vec::new(),
            rows: Vec::new(),
            meter_frames: 0,
            settings: 0,
            relearn: 0,
            clipped: (0, 0),
        })
    }

    pub fn center_hz(&self) -> u32 {
        self.center_hz as u32
    }

    /// Apply a client command; returns the tuner changes it needs.
    pub fn command(&mut self, cmd: &Value) -> Result<Vec<Control>> {
        let name = cmd["cmd"].as_str().unwrap_or_default();
        match name {
            "tune" => {
                let hz = cmd["hz"].as_f64().ok_or_else(|| anyhow!("tune needs hz"))?;
                if let Some(m) = cmd["mode"].as_str() {
                    self.set_mode(parse_mode(m)?)?;
                }
                if let Some(bw) = cmd["bandwidth"].as_f64() {
                    self.set_bandwidth(bw as f32)?;
                }
                if !cmd["squelch"].is_null() {
                    self.squelch.threshold = cmd["squelch"].as_f64().map(|db| db as f32);
                }
                self.tune(hz)
            }
            "mode" => {
                let mode = parse_mode(cmd["mode"].as_str().unwrap_or_default())?;
                self.set_mode(mode)?;
                self.tune(self.tuned_hz)
            }
            "bandwidth" => {
                let hz = cmd["hz"].as_f64().ok_or_else(|| anyhow!("bandwidth needs hz"))?;
                self.set_bandwidth(hz as f32)?;
                self.tune(self.tuned_hz)
            }
            "squelch" => {
                self.squelch.threshold = match &cmd["db"] {
                    Value::Null => None,
                    v => Some(v.as_f64().ok_or_else(|| anyhow!("squelch needs db or null"))? as f32),
                };
                Ok(Vec::new())
            }
            "cleanup" => {
                self.cleanup = cmd["on"].as_bool().ok_or_else(|| anyhow!("cleanup needs on"))?;
                Ok(Vec::new())
            }
            "gain" => {
                self.gain = match &cmd["db"] {
                    Value::String(s) => s.parse().map_err(|e: String| anyhow!(e))?,
                    v => Gain::Manual(v.as_f64().ok_or_else(|| anyhow!("gain needs db or \"auto\""))? as f32),
                };
                if let Gain::Manual(db) = self.gain {
                    self.gain = Gain::Manual(snap(db, &self.gains));
                }
                Ok(vec![Control::Gain(self.gain)])
            }
            "span" => {
                let rate = cmd["rate"].as_u64().unwrap_or_default() as u32;
                if !SPANS.contains(&rate) {
                    bail!("span must be one of {SPANS:?}");
                }
                if rate == self.rate {
                    return Ok(Vec::new());
                }
                self.rate = rate;
                // The channel is rebuilt when blocks at the new rate arrive.
                let mut controls = vec![Control::Rate(rate)];
                if let Some(c) = plan_center(self.tuned_hz, self.center_hz, rate, self.bandwidth) {
                    self.center_hz = c;
                }
                controls.push(Control::Center(self.center_hz as u32));
                Ok(controls)
            }
            other => bail!("unknown command {other:?}"),
        }
    }

    fn set_mode(&mut self, mode: Mode) -> Result<()> {
        if mode == self.mode {
            return Ok(());
        }
        let old_bw = self.bandwidth;
        self.mode = mode;
        self.bandwidth = mode.default_bandwidth();
        self.cleanup = mode.default_cleanup();
        self.squelch.threshold = mode.default_squelch();
        let (hp, lp) = mode.audio_band();
        self.basic = BasicChain::new(hp, lp, AGC_TARGET_DBFS, AGC_MAX_GAIN_DB);
        self.detail = Spectrum::new(DETAIL_BINS, mode.demod_rate(), ROWS_PER_S);
        self.detail_levels = Levels::default();
        self.rebuild(old_bw)
    }

    fn set_bandwidth(&mut self, hz: f32) -> Result<()> {
        let range = self.mode.bandwidth_range();
        if !range.contains(&hz) {
            bail!("bandwidth for {} must be {}–{} Hz", self.mode.name(), range.start(), range.end());
        }
        let old_bw = self.bandwidth;
        self.bandwidth = hz;
        self.rebuild(old_bw)
    }

    /// New channel after a mode or bandwidth change. The noise floor moves with the
    /// bandwidth, so the squelch's learned floor is shifted rather than relearned.
    fn rebuild(&mut self, old_bw: f32) -> Result<()> {
        let (center, rate) = (self.center_hz, self.channel_at.1);
        let channel =
            build_channel(rate, center, self.tuned_hz, self.mode, self.bandwidth, self.deemphasis)
                .or_else(|_| {
                    // The channel no longer fits around the old centre: build it for the
                    // middle of the band; it is retuned when blocks at the new centre arrive.
                    build_channel(rate, center, center + 100e3, self.mode, self.bandwidth, self.deemphasis)
                })?;
        self.channel = channel;
        // Offset it for the actual tuner centre with the next block.
        self.channel_at.0 = 0;
        self.squelch.shift_floor(10.0 * (self.bandwidth / old_bw).log10());
        Ok(())
    }

    fn tune(&mut self, hz: f64) -> Result<Vec<Control>> {
        if !(MIN_HZ..=MAX_HZ).contains(&hz) {
            bail!("{:.3} MHz is outside {}–{} MHz", hz / 1e6, MIN_HZ / 1e6, MAX_HZ / 1e6);
        }
        self.tuned_hz = hz;
        let mut controls = Vec::new();
        if let Some(c) = plan_center(hz, self.center_hz, self.rate, self.bandwidth) {
            self.center_hz = c;
            controls.push(Control::Center(c as u32));
        }
        let (center, _) = self.channel_at;
        // Fails while the channel is outside the band of the old centre; `process` then
        // retunes once blocks from the new centre arrive.
        if self.channel.set_offset(hz - center as f64).is_err() {
            self.channel_at.0 = 0;
        }
        Ok(controls)
    }

    /// Demodulate one block and make the waterfall rows for it.
    pub fn process(&mut self, block: &Block, out: &mut Output) {
        if block.rate != self.rate {
            return; // captured before a span change took effect
        }
        if (block.center_hz, block.rate) != self.channel_at {
            let offset = self.tuned_hz - block.center_hz as f64;
            if block.rate != self.channel_at.1 {
                match build_channel(block.rate, block.center_hz as f64, self.tuned_hz, self.mode, self.bandwidth, self.deemphasis) {
                    Ok(c) => self.channel = c,
                    Err(_) => return,
                }
                self.wide = Spectrum::new(WIDE_BINS, block.rate, ROWS_PER_S);
            } else if self.channel.set_offset(offset).is_err() {
                return; // captured before the retune took effect
            }
            self.channel_at = (block.center_hz, block.rate);
        }

        if block.settings != self.settings {
            self.settings = block.settings;
            self.relearn = RELEARN_FRAMES;
        }
        self.clipped.0 += block.data.iter().filter(|&&b| b == 0 || b == 255).count() as u64;
        self.clipped.1 += block.data.len() as u64;

        self.rows.clear();
        self.wide.push_u8(&block.data, &mut self.rows);
        for row in self.rows.drain(..) {
            let levels = self.wide_levels.quantize(&row);
            out.rows.push(row_message(2, block.center_hz, block.rate, &self.wide_levels, &levels, &mut self.wide_enc));
        }

        self.frames.clear();
        self.channel.process_u8(&block.data, &mut self.frames);
        self.channel.take_tap(&mut self.tap);
        self.detail.push(&self.tap, &mut self.rows);
        self.tap.clear();
        let span = self.channel.tap_rate();
        for row in self.rows.drain(..) {
            let levels = self.detail_levels.quantize(&row);
            out.rows.push(row_message(3, self.tuned_hz.round() as u32, span, &self.detail_levels, &levels, &mut self.detail_enc));
        }

        for frame in &self.frames {
            if self.relearn > 0 {
                self.relearn -= 1;
                self.squelch.relearn();
            }
            let carrier = matches!(self.mode, Mode::Am | Mode::Nfm) && frame.carrier_db >= CARRIER_PROMINENCE_DB;
            let open = self.squelch.push(frame.power_db, carrier);
            let mut audio = frame.audio;
            if self.cleanup {
                self.clean.process_frame(&mut audio);
            } else {
                self.basic.process_frame(&mut audio);
            }
            if !open {
                audio = [0.0; crate::clean::FRAME];
            }
            out.audio.extend_from_slice(&audio);
            self.meter_frames += 1;
            if self.meter_frames >= METER_FRAMES {
                self.meter_frames = 0;
                let (clipped, total) = std::mem::take(&mut self.clipped);
                out.meters.push(json!({
                    "type": "meter",
                    "level_db": round1(self.squelch.level_db()),
                    "floor_db": self.squelch.floor_db().map(round1),
                    "open": open,
                    "carrier_db": round1(frame.carrier_db),
                    // Fraction of ADC samples at the rails: lower the gain if this is not ~0.
                    "clip": clipped as f64 / total.max(1) as f64,
                }));
            }
        }
    }

    pub fn state_json(&self) -> Value {
        json!({
            "type": "state",
            "hz": self.tuned_hz,
            "center_hz": self.center_hz,
            "rate": self.rate,
            "mode": self.mode.name(),
            "bandwidth": self.bandwidth,
            "bandwidth_min": self.mode.bandwidth_range().start(),
            "bandwidth_max": self.mode.bandwidth_range().end(),
            "step_hz": self.mode.step_hz(),
            "squelch": self.squelch.threshold,
            "cleanup": self.cleanup,
            "gain": match self.gain { Gain::Auto => Value::Null, Gain::Manual(db) => json!(db) },
            "gains": self.gains,
            "spans": SPANS,
            "modes": Mode::ALL.iter().map(|m| json!({
                "name": m.name(),
                "bandwidth": m.default_bandwidth(),
                "step_hz": m.step_hz(),
                "squelch": m.default_squelch(),
                "cleanup": m.default_cleanup(),
            })).collect::<Vec<_>>(),
        })
    }
}

fn build_channel(rate: u32, center: f64, hz: f64, mode: Mode, bandwidth: f32, deemphasis: f32) -> Result<Channel> {
    let mut channel = Channel::new(rate, hz - center, bandwidth, mode)?.with_deemphasis(deemphasis);
    channel.enable_tap();
    Ok(channel)
}

fn parse_mode(name: &str) -> Result<Mode> {
    Mode::from_name(name).ok_or_else(|| anyhow!("unknown mode {name:?}"))
}

/// The supported gain step nearest `db`.
fn snap(db: f32, gains: &[f32]) -> f32 {
    gains.iter().copied().min_by(|a, b| (a - db).abs().total_cmp(&(b - db).abs())).unwrap_or(db)
}

fn round1(x: f32) -> f32 {
    (x * 10.0).round() / 10.0
}

/// Binary waterfall row: type, header, then the ADPCM-coded levels.
fn row_message(kind: u8, center_hz: u32, span: u32, range: &Levels, levels: &[u8], enc: &mut adpcm::Encoder) -> Vec<u8> {
    let samples: Vec<i16> = levels.iter().map(|&l| l as i16 * ROW_SCALE).collect();
    let mut msg = Vec::with_capacity(ROW_HEADER + adpcm::HEADER + levels.len() / 2);
    msg.push(kind);
    msg.extend_from_slice(&center_hz.to_le_bytes());
    msg.extend_from_slice(&span.to_le_bytes());
    msg.extend_from_slice(&range.lo.to_le_bytes());
    msg.extend_from_slice(&range.hi.to_le_bytes());
    msg.extend_from_slice(&(levels.len() as u16).to_le_bytes());
    msg.extend(enc.encode(&samples));
    msg
}

/// 48 kHz audio → `STREAM_RATE` ADPCM messages of `CHUNK` samples.
struct AudioStream {
    resampler: Resampler,
    scratch: Vec<f32>,
    pending: Vec<i16>,
    enc: adpcm::Encoder,
}

impl AudioStream {
    fn new() -> Self {
        Self {
            resampler: Resampler::new(SAMPLE_RATE, STREAM_RATE),
            scratch: Vec::new(),
            pending: Vec::with_capacity(CHUNK),
            enc: adpcm::Encoder::default(),
        }
    }

    fn push(&mut self, audio: &[f32], out: &mut Vec<Vec<u8>>) {
        self.scratch.clear();
        self.resampler.process(audio, &mut self.scratch);
        for &s in &self.scratch {
            self.pending.push(to_i16(s));
            if self.pending.len() == CHUNK {
                let mut msg = vec![1u8];
                msg.extend(self.enc.encode(&self.pending));
                out.push(msg);
                self.pending.clear();
            }
        }
    }
}

pub fn run(args: WebArgs) -> Result<()> {
    let mut dev = Device::open(args.device)?;
    let rate = dev.set_sample_rate(args.sample_rate)?;
    dev.set_ppm(args.ppm)?;
    let gain_db = dev.set_gain(args.gain)?;
    let gains: Vec<f32> = dev.gains().iter().map(|&g| g as f32 / 10.0).collect();
    let gain = gain_db.map_or(Gain::Auto, Gain::Manual);
    let mut radio = Radio::new(rate, args.freq * 1e6, args.mode, gain, gains, args.deemphasis, args.clean.clone())?;
    dev.set_center_freq(radio.center_hz())?;
    let mut bookmarks = Bookmarks::open(&args.bookmarks)?;

    let (cmd_tx, cmd_rx) = channel::<Value>();
    let (hub, url) = Hub::start(&args.serve, PAGE, Some(cmd_tx))?;
    hub.latest("state", radio.state_json());
    hub.latest("bookmarks", bookmarks.json());
    eprintln!(
        "receiver on {url} ({:.4} MHz {}, span {:.2} MHz, gain {gain}; bookmarks {})",
        args.freq,
        args.mode.name(),
        rate as f64 / 1e6,
        args.bookmarks.display()
    );
    eprintln!("anyone who can reach that address can tune the receiver: keep it on a trusted network");

    let mut speaker = if args.no_play {
        None
    } else {
        match AudioOut::open(&args.audio_device, args.volume) {
            Ok(out) => {
                eprintln!("also playing on {}", out.description);
                Some(out)
            }
            Err(e) => {
                eprintln!("warning: no local playback ({e:#})");
                None
            }
        }
    };

    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        ctrlc::set_handler(move || stop.store(true, Ordering::Relaxed)).context("installing Ctrl-C handler")?;
    }
    let (tx, rx) = sync_channel::<Block>(64);
    let (control_tx, control_rx) = channel::<Control>();
    let reader = {
        let stop = stop.clone();
        thread::spawn(move || dev.stream(tx, stop, Some(control_rx)))
    };

    serve(&mut radio, &mut bookmarks, &hub, &cmd_rx, &rx, &control_tx, speaker.as_mut(), &stop);
    stop.store(true, Ordering::Relaxed);
    drop(rx);
    reader.join().map_err(|_| anyhow!("SDR reader thread panicked"))??;
    Ok(())
}

/// Apply client commands and turn IQ blocks into audio and waterfall messages until `stop`
/// is set or the block source goes away.
#[allow(clippy::too_many_arguments)]
fn serve(
    radio: &mut Radio,
    bookmarks: &mut Bookmarks,
    hub: &Hub,
    cmd_rx: &Receiver<Value>,
    rx: &Receiver<Block>,
    control_tx: &Sender<Control>,
    mut speaker: Option<&mut AudioOut>,
    stop: &AtomicBool,
) {
    let mut stream = AudioStream::new();
    let mut audio_msgs = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        for cmd in cmd_rx.try_iter() {
            let result = match cmd["cmd"].as_str().unwrap_or_default() {
                "bookmark_add" => serde_json::from_value::<Bookmark>(json!({
                    "id": 0,
                    "name": cmd["name"],
                    "hz": cmd["hz"],
                    "mode": cmd["mode"],
                    "bandwidth": cmd["bandwidth"],
                    "squelch": cmd["squelch"],
                }))
                .map_err(anyhow::Error::from)
                .and_then(|b| bookmarks.add(b))
                .map(|_| hub.latest("bookmarks", bookmarks.json())),
                "bookmark_del" => bookmarks
                    .remove(cmd["id"].as_u64().unwrap_or_default() as u32)
                    .map(|_| hub.latest("bookmarks", bookmarks.json())),
                _ => radio.command(&cmd).map(|controls| {
                    for c in controls {
                        let _ = control_tx.send(c);
                    }
                    hub.latest("state", radio.state_json());
                }),
            };
            if let Err(e) = result {
                hub.event(json!({"type": "error", "message": format!("{e:#}")}));
            }
        }

        let block = match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(b) => b,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        let mut out = Output::default();
        radio.process(&block, &mut out);
        if let Some(s) = speaker.as_deref_mut() {
            s.play(&out.audio);
        }
        if hub.clients() > 0 {
            stream.push(&out.audio, &mut audio_msgs);
            for msg in audio_msgs.drain(..).chain(out.rows) {
                hub.binary(msg);
            }
            for m in out.meters {
                hub.event(m);
            }
        }
    }
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

    fn radio(rate: u32, mhz: f64, mode: Mode) -> Radio {
        let cfg = Wrapper::parse_from(["x"]).cfg;
        Radio::new(rate, mhz * 1e6, mode, Gain::Manual(30.0), vec![0.0, 29.7, 32.8, 49.6], 50.0, cfg).unwrap()
    }

    /// One block of u8 IQ at the radio's tuner centre with an NFM 1 kHz tone at `hz`.
    fn block(r: &Radio, hz: f64, seconds: f64, amp: f64) -> Block {
        let rate = r.rate;
        let offset = hz - r.center_hz() as f64;
        let n = (rate as f64 * seconds) as usize;
        let mut phase = 0.0f64;
        let mut data = Vec::with_capacity(2 * n);
        for i in 0..n {
            let t = i as f64 / rate as f64;
            let inst = offset + 2500.0 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin();
            phase += 2.0 * std::f64::consts::PI * inst / rate as f64;
            for v in [amp * phase.cos(), amp * phase.sin()] {
                data.push((127.4 + v * 128.0).round().clamp(0.0, 255.0) as u8);
            }
        }
        Block { center_hz: r.center_hz(), rate, settings: 0, data }
    }

    #[test]
    fn plan_center_keeps_channel_in_band_and_off_dc() {
        let rate = 2_400_000;
        // Already fine: no hop.
        assert_eq!(plan_center(145.5e6, 145.1e6, rate, 12_500.0), None);
        // On the DC spike: hop so the channel sits a sixth of the span away.
        let c = plan_center(145.5e6, 145.5e6, rate, 12_500.0).unwrap();
        assert_eq!(c, 145.1e6);
        // Off the top of the band: the channel lands in the upper part of the new band.
        let c = plan_center(146.5e6, 145.1e6, rate, 12_500.0).unwrap();
        assert_eq!(c, 146.1e6);
        // WFM needs more room around DC than a narrow channel.
        assert!(plan_center(100.0e6, 100.05e6, rate, 180_000.0).is_some());
        assert_eq!(plan_center(145.5e6, 145.47e6, rate, 12_500.0), None);
    }

    #[test]
    fn tunes_demodulates_and_makes_waterfall_rows() {
        let mut r = radio(960_000, 145.5, Mode::Nfm);
        r.command(&json!({"cmd": "squelch", "db": null})).unwrap();
        let b = block(&r, 145.5e6, 0.5, 0.3);
        let mut out = Output::default();
        r.process(&b, &mut out);
        assert_eq!(out.audio.len(), SAMPLE_RATE as usize / 2);
        assert_eq!(out.meters.len(), 5);
        let wide = out.rows.iter().filter(|m| m[0] == 2).count();
        let detail = out.rows.iter().filter(|m| m[0] == 3).count();
        assert_eq!((wide, detail), (5, 5), "10 rows/s of each");
        let row = out.rows.iter().find(|m| m[0] == 2).unwrap();
        assert_eq!(u32::from_le_bytes(row[1..5].try_into().unwrap()), r.center_hz());
        assert_eq!(u16::from_le_bytes(row[17..19].try_into().unwrap()) as usize, WIDE_BINS);
        assert_eq!(row.len(), ROW_HEADER + adpcm::HEADER + WIDE_BINS / 2);
        let tail = &out.audio[out.audio.len() / 2..];
        let rms = (tail.iter().map(|x| x * x).sum::<f32>() / tail.len() as f32).sqrt();
        assert!(rms > 0.03, "audio rms {rms}");
    }

    #[test]
    fn tuning_within_band_retunes_without_hop_and_far_tune_hops() {
        let mut r = radio(960_000, 145.5, Mode::Nfm);
        let center = r.center_hz();
        assert!(r.command(&json!({"cmd": "tune", "hz": 145.45e6})).unwrap().is_empty());
        assert_eq!(r.center_hz(), center);
        let controls = r.command(&json!({"cmd": "tune", "hz": 433.5e6})).unwrap();
        assert!(matches!(controls[..], [Control::Center(c)] if c == r.center_hz()));
        // Blocks from the old centre are ignored until the tuner has moved.
        let stale = Block { center_hz: center, rate: 960_000, settings: 0, data: vec![127; 96_000] };
        let mut out = Output::default();
        r.process(&stale, &mut out);
        assert!(out.audio.is_empty());
        let fresh = block(&r, 433.5e6, 0.1, 0.3);
        r.process(&fresh, &mut out);
        assert_eq!(out.audio.len(), SAMPLE_RATE as usize / 10);
        assert!(r.command(&json!({"cmd": "tune", "hz": 2000e6})).is_err());
    }

    #[test]
    fn mode_change_applies_mode_defaults() {
        let mut r = radio(960_000, 100.0, Mode::Nfm);
        r.command(&json!({"cmd": "mode", "mode": "wfm"})).unwrap();
        let s = r.state_json();
        assert_eq!(s["mode"], "wfm");
        assert_eq!(s["bandwidth"], 180_000.0);
        assert_eq!(s["cleanup"], false);
        assert!(s["squelch"].is_null());
        // A WFM channel this close to DC forces a hop.
        let center = r.center_hz() as f64;
        assert!((100e6 - center).abs() >= 95_000.0);

        r.command(&json!({"cmd": "cleanup", "on": true})).unwrap();
        r.command(&json!({"cmd": "squelch", "db": 4.5})).unwrap();
        let s = r.state_json();
        assert_eq!(s["cleanup"], true);
        assert_eq!(s["squelch"], 4.5);
        assert!(r.command(&json!({"cmd": "bandwidth", "hz": 5000.0})).is_err());
        assert!(r.command(&json!({"cmd": "mode", "mode": "dmr"})).is_err());
    }

    #[test]
    fn gain_snaps_and_span_changes_rate() {
        let mut r = radio(2_400_000, 145.5, Mode::Nfm);
        let c = r.command(&json!({"cmd": "gain", "db": 33.0})).unwrap();
        assert!(matches!(c[..], [Control::Gain(Gain::Manual(g))] if g == 32.8));
        let c = r.command(&json!({"cmd": "gain", "db": "auto"})).unwrap();
        assert!(matches!(c[..], [Control::Gain(Gain::Auto)]));
        assert!(r.state_json()["gain"].is_null());

        let c = r.command(&json!({"cmd": "span", "rate": 960_000})).unwrap();
        assert!(matches!(c[0], Control::Rate(960_000)));
        assert!(r.command(&json!({"cmd": "span", "rate": 1_000_000})).is_err());
        // Blocks at the new rate rebuild the channel.
        let b = block(&r, 145.5e6, 0.1, 0.3);
        let mut out = Output::default();
        r.process(&b, &mut out);
        assert_eq!(out.audio.len(), SAMPLE_RATE as usize / 10);
    }

    /// Serves the page on 127.0.0.1:8011 with simulated IQ (noise plus a few FM and AM
    /// signals), to check the page in a browser without an SDR:
    /// `SDRFUN_DEMO_SECS=300 cargo test --release web::tests::demo -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn demo() {
        let secs: u64 = std::env::var("SDRFUN_DEMO_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
        let mut radio = radio(2_400_000, 145.5, Mode::Nfm);
        let path = std::env::temp_dir().join(format!("sdrfun-demo-{}.json", std::process::id()));
        let mut bookmarks = Bookmarks::open(&path).unwrap();
        let (cmd_tx, cmd_rx) = channel();
        let (hub, url) = Hub::start("127.0.0.1:8011", PAGE, Some(cmd_tx)).unwrap();
        hub.latest("state", radio.state_json());
        hub.latest("bookmarks", bookmarks.json());
        println!("demo receiver on {url} for {secs} s");

        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = sync_channel::<Block>(64);
        let (control_tx, control_rx) = channel::<Control>();
        let (mut center, mut rate) = (radio.center_hz(), 2_400_000u32);
        {
            let stop = stop.clone();
            thread::spawn(move || {
                // (frequency, amplitude, AM?) with a voice-ish 600 Hz + 1100 Hz tone.
                let signals = [
                    (145.500e6, 0.05, false),
                    (145.6125e6, 0.15, false),
                    (145.3375e6, 0.02, false),
                    (145.900e6, 0.08, true),
                    (144.300e6, 0.04, true),
                ];
                let mut phases = vec![0.0f64; signals.len()];
                let mut noise = 0x2545_F491_4F6C_DD1Du64;
                let mut t = 0.0f64;
                let n = 131_072;
                let start = std::time::Instant::now();
                let mut sent = 0.0f64;
                while !stop.load(Ordering::Relaxed) {
                    for c in control_rx.try_iter() {
                        match c {
                            Control::Center(hz) => center = hz,
                            Control::Rate(r) => rate = r,
                            Control::Gain(_) => {}
                        }
                    }
                    let mut data = Vec::with_capacity(2 * n);
                    for _ in 0..n {
                        let (mut i, mut q) = (0.0f64, 0.0f64);
                        let tone = 0.6 * (2.0 * std::f64::consts::PI * 600.0 * t).sin()
                            + 0.4 * (2.0 * std::f64::consts::PI * 1100.0 * t).sin();
                        // Transmissions come and go: on for 3 s of every 5.
                        let keyed = (t % 5.0) < 3.0;
                        for (k, &(hz, amp, am)) in signals.iter().enumerate() {
                            if k == 0 && !keyed {
                                continue;
                            }
                            let offset = hz - center as f64;
                            if offset.abs() > rate as f64 / 2.0 {
                                continue;
                            }
                            let (inst, a) = if am { (offset, amp * (1.0 + 0.7 * tone)) } else { (offset + 2500.0 * tone, amp) };
                            phases[k] += 2.0 * std::f64::consts::PI * inst / rate as f64;
                            i += a * phases[k].cos();
                            q += a * phases[k].sin();
                        }
                        for v in [&mut i, &mut q] {
                            noise ^= noise << 13;
                            noise ^= noise >> 7;
                            noise ^= noise << 17;
                            let g: f64 = (0..4).map(|s| ((noise >> (s * 8)) & 0xff) as f64).sum::<f64>() / 4.0 - 127.5;
                            *v += g / 128.0 * 0.03;
                        }
                        data.push((127.4 + i * 128.0).round().clamp(0.0, 255.0) as u8);
                        data.push((127.4 + q * 128.0).round().clamp(0.0, 255.0) as u8);
                        t += 1.0 / rate as f64;
                    }
                    if tx.send(Block { center_hz: center, rate, settings: 0, data }).is_err() {
                        break;
                    }
                    sent += n as f64 / rate as f64;
                    let ahead = sent - start.elapsed().as_secs_f64();
                    if ahead > 0.0 {
                        thread::sleep(Duration::from_secs_f64(ahead));
                    }
                }
            });
        }
        {
            let stop = stop.clone();
            thread::spawn(move || {
                thread::sleep(Duration::from_secs(secs));
                stop.store(true, Ordering::Relaxed);
            });
        }
        serve(&mut radio, &mut bookmarks, &hub, &cmd_rx, &rx, &control_tx, None, &stop);
        let _ = std::fs::remove_file(&path);
    }
}
