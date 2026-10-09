//! Device access for the radios sdrfun can use: RTL-SDR (librtlsdr) and HackRF One
//! (libhackrf). Both are exposed as one `Device` enum producing the same u8 IQ blocks
//! (HackRF's signed 8-bit samples are biased to the RTL's unsigned convention), so the
//! DSP never notices which radio is running.
//!
//! `Device::stream` is the simple single-radio loop `listen` and `scan` use; `pump` is
//! the richer loop behind `sdrfun web`: it can switch between attached radios and run a
//! spectrum sweep (see `sweep.rs`) without audio.

use std::collections::VecDeque;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Mutex, Once};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};

// Linked via build.rs (pkg-config).
unsafe extern "C" {
    fn rtlsdr_get_device_count() -> u32;
    fn rtlsdr_get_device_name(index: u32) -> *const c_char;
    fn rtlsdr_get_device_usb_strings(
        index: u32,
        manufact: *mut c_char,
        product: *mut c_char,
        serial: *mut c_char,
    ) -> c_int;
    fn rtlsdr_open(dev: *mut *mut c_void, index: u32) -> c_int;
    fn rtlsdr_close(dev: *mut c_void) -> c_int;
    fn rtlsdr_set_center_freq(dev: *mut c_void, freq: u32) -> c_int;
    fn rtlsdr_get_center_freq(dev: *mut c_void) -> u32;
    fn rtlsdr_set_freq_correction(dev: *mut c_void, ppm: c_int) -> c_int;
    fn rtlsdr_get_tuner_gains(dev: *mut c_void, gains: *mut c_int) -> c_int;
    fn rtlsdr_set_tuner_gain(dev: *mut c_void, gain: c_int) -> c_int;
    fn rtlsdr_set_tuner_gain_mode(dev: *mut c_void, manual: c_int) -> c_int;
    fn rtlsdr_set_sample_rate(dev: *mut c_void, rate: u32) -> c_int;
    fn rtlsdr_get_sample_rate(dev: *mut c_void) -> u32;
    fn rtlsdr_set_agc_mode(dev: *mut c_void, on: c_int) -> c_int;
    fn rtlsdr_reset_buffer(dev: *mut c_void) -> c_int;
    fn rtlsdr_read_sync(dev: *mut c_void, buf: *mut c_void, len: c_int, n_read: *mut c_int)
    -> c_int;
}

// libhackrf (HackRF One). Samples arrive through a callback, signed 8-bit I/Q.
unsafe extern "C" {
    fn hackrf_init() -> c_int;
    fn hackrf_device_list() -> *mut HackrfDeviceList;
    fn hackrf_device_list_free(list: *mut HackrfDeviceList);
    fn hackrf_device_list_open(list: *mut HackrfDeviceList, idx: c_int, dev: *mut *mut c_void) -> c_int;
    fn hackrf_close(dev: *mut c_void) -> c_int;
    fn hackrf_set_freq(dev: *mut c_void, hz: u64) -> c_int;
    fn hackrf_set_sample_rate(dev: *mut c_void, rate: f64) -> c_int;
    fn hackrf_set_lna_gain(dev: *mut c_void, db: u32) -> c_int;
    fn hackrf_set_vga_gain(dev: *mut c_void, db: u32) -> c_int;
    fn hackrf_start_rx(dev: *mut c_void, cb: HackrfCb, ctx: *mut c_void) -> c_int;
    fn hackrf_stop_rx(dev: *mut c_void) -> c_int;
}

/// `hackrf_transfer` from libhackrf.h; the callback receives one per USB completion.
#[repr(C)]
struct HackrfTransfer {
    device: *mut c_void,
    buffer: *mut u8,
    buffer_length: c_int,
    valid_length: c_int,
    rx_ctx: *mut c_void,
    tx_ctx: *mut c_void,
}

/// `hackrf_device_list` from libhackrf.h (only the fields we read).
#[repr(C)]
struct HackrfDeviceList {
    serial_numbers: *mut *mut c_char,
    usb_board_ids: *mut c_int,
    usb_device_index: *mut c_int,
    devicecount: c_int,
    usb_devices: *mut *mut c_void,
    usb_devicecount: c_int,
}

type HackrfCb = unsafe extern "C" fn(*mut HackrfTransfer) -> c_int;

/// Samples buffered from the HackRF callback before the reader drains them (~1 s at
/// 20 MS/s); the oldest are dropped when it overflows.
const HACKRF_BUF_BYTES: usize = 8 << 20;

extern "C" fn hackrf_rx_cb(t: *mut HackrfTransfer) -> c_int {
    unsafe {
        let t = &*t;
        let buf = &*(t.rx_ctx as *const Mutex<VecDeque<u8>>);
        let data = std::slice::from_raw_parts(t.buffer, t.valid_length.max(0) as usize);
        let mut b = buf.lock().unwrap_or_else(|e| e.into_inner());
        while b.len() + data.len() > HACKRF_BUF_BYTES {
            let cut = (b.len() + data.len() - HACKRF_BUF_BYTES).min(b.len());
            b.drain(..cut);
        }
        // Signed 8-bit → the RTL's unsigned convention (flip the sign bit), so the DSP
        // reads both radios alike.
        b.extend(data.iter().map(|&x| x ^ 0x80));
    }
    0
}

fn hackrf_init_once() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let ret = unsafe { hackrf_init() };
        if ret != 0 {
            eprintln!("warning: hackrf_init failed ({ret}); HackRF support unavailable");
        }
    });
}

/// Bytes per synchronous read: 128 Ki IQ pairs, ~55 ms at 2.4 MS/s.
const READ_LEN: usize = 16 * 16384;

/// A block of raw IQ, tagged with the tuner frequency and sample rate it was captured at.
pub struct Block {
    pub center_hz: u32,
    pub rate: u32,
    /// Counts gain and sample-rate changes applied so far, so consumers can tell which
    /// blocks were captured with new settings.
    pub settings: u32,
    pub data: Vec<u8>,
}

/// What the web reader thread puts on its channel: IQ blocks, or one finished sweep row.
pub enum Capture {
    Block(Block),
    SweepRow(Vec<u8>),
}

/// Changes applied by `Device::stream` and `pump` between blocks.
#[derive(Clone, Copy, Debug)]
pub enum Control {
    Center(u32),
    Gain(Gain),
    Rate(u32),
    /// Make the other attached radio the active one (web only).
    Switch(SdrKind),
    /// Run spectrum sweeps over this range instead of streaming IQ (web only).
    Sweep(Option<(u32, u32)>),
    /// Waterfall dB range (lo, hi), or None to follow the signals automatically.
    Levels(Option<(f32, f32)>),
    /// Bins per waterfall row (a power of two), for sweep rows.
    Bins(usize),
}

/// Which family a device belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SdrKind {
    Rtlsdr,
    Hackrf,
}

impl SdrKind {
    pub fn name(self) -> &'static str {
        match self {
            SdrKind::Rtlsdr => "rtlsdr",
            SdrKind::Hackrf => "hackrf",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "rtlsdr" | "rtl-sdr" | "rtl" => Some(SdrKind::Rtlsdr),
            "hackrf" => Some(SdrKind::Hackrf),
            _ => None,
        }
    }

    /// Frequencies the tuner can reach, Hz.
    pub fn freq_range(self) -> (f64, f64) {
        match self {
            // The V4's upconverter covers HF; the R828D tops out at 1766 MHz.
            SdrKind::Rtlsdr => (100e3, 1766e6),
            // HackRF One: 10 MHz – 6 GHz (reduced sensitivity 2.4–2.5 GHz).
            SdrKind::Hackrf => (10e6, 6e9),
        }
    }

    /// Sample rate, hop width and FFT size for spectrum sweeps, Hz.
    pub fn sweep_plan(self) -> (u32, u32, usize) {
        match self {
            SdrKind::Rtlsdr => (2_400_000, 2_000_000, 1024),
            // 20 MS/s is the HackRF's maximum: 8x the RTL's hops.
            SdrKind::Hackrf => (20_000_000, 16_000_000, 1024),
        }
    }

    /// Gain used for sweeps when the receiver is on auto: an AGC would hunt from hop to
    /// hop and band the waterfall, so sweeps always run with a fixed manual gain.
    pub fn sweep_gain(self) -> Gain {
        match self {
            SdrKind::Rtlsdr => Gain::Manual(33.8),
            // LNA 40 + VGA 8: low-noise side of the ladder, plenty for VHF/UHF.
            SdrKind::Hackrf => Gain::Manual(48.0),
        }
    }
}

/// What `--sdr` / `SDRFUN_SDR` selects.
#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
pub enum SdrSelect {
    Rtlsdr,
    Hackrf,
    Both,
}

impl SdrSelect {
    /// Flag value, else `SDRFUN_SDR`, else auto (whatever is attached).
    pub fn resolve(flag: Option<Self>) -> Self {
        flag.or_else(|| {
            std::env::var("SDRFUN_SDR")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(SdrSelect::Both)
    }
}

impl std::str::FromStr for SdrSelect {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "rtlsdr" | "rtl-sdr" | "rtl" => Ok(SdrSelect::Rtlsdr),
            "hackrf" => Ok(SdrSelect::Hackrf),
            "both" | "auto" => Ok(SdrSelect::Both),
            _ => Err(format!("expected rtlsdr, hackrf or both, got {s:?}")),
        }
    }
}

pub struct DeviceInfo {
    pub kind: SdrKind,
    pub index: u32,
    pub name: String,
    pub manufacturer: String,
    pub product: String,
    pub serial: String,
}

pub fn list_devices() -> Vec<DeviceInfo> {
    let mut out = Vec::new();
    let count = unsafe { rtlsdr_get_device_count() };
    for index in 0..count {
        let mut bufs = [[0 as c_char; 256]; 3];
        let [m, p, s] = &mut bufs;
        let ok = unsafe {
            rtlsdr_get_device_usb_strings(index, m.as_mut_ptr(), p.as_mut_ptr(), s.as_mut_ptr())
        } == 0;
        let text = |b: &[c_char; 256]| {
            if ok {
                unsafe { CStr::from_ptr(b.as_ptr()) }.to_string_lossy().into_owned()
            } else {
                String::new()
            }
        };
        let name_ptr = unsafe { rtlsdr_get_device_name(index) };
        let name = if name_ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy().into_owned()
        };
        out.push(DeviceInfo {
            kind: SdrKind::Rtlsdr,
            index,
            name,
            manufacturer: text(&bufs[0]),
            product: text(&bufs[1]),
            serial: text(&bufs[2]),
        });
    }
    for (index, serial) in hackrf_serials().into_iter().enumerate() {
        out.push(DeviceInfo {
            kind: SdrKind::Hackrf,
            index: index as u32,
            name: "HackRF".into(),
            manufacturer: "Great Scott Gadgets".into(),
            product: "HackRF One".into(),
            serial,
        });
    }
    out
}

fn hackrf_serials() -> Vec<String> {
    hackrf_init_once();
    let list = unsafe { hackrf_device_list() };
    if list.is_null() {
        return Vec::new();
    }
    let n = unsafe { (*list).devicecount }.max(0) as usize;
    let out = (0..n)
        .map(|i| {
            let p = unsafe { *(*list).serial_numbers.add(i) };
            if p.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
            }
        })
        .collect();
    unsafe { hackrf_device_list_free(list) };
    out
}

/// How many devices of a kind are attached.
pub fn count(kind: SdrKind) -> usize {
    match kind {
        SdrKind::Rtlsdr => (unsafe { rtlsdr_get_device_count() }) as usize,
        SdrKind::Hackrf => hackrf_serials().len(),
    }
}

/// The devices `--sdr`/`SDRFUN_SDR` selects, in preference order (RTL first).
/// `Both`/auto yields every kind that is attached.
pub fn selected(select: SdrSelect) -> Result<Vec<SdrKind>> {
    let present = |k| count(k) > 0;
    let out = match select {
        SdrSelect::Rtlsdr if present(SdrKind::Rtlsdr) => vec![SdrKind::Rtlsdr],
        SdrSelect::Hackrf if present(SdrKind::Hackrf) => vec![SdrKind::Hackrf],
        SdrSelect::Rtlsdr | SdrSelect::Hackrf => {
            bail!("no {} attached (sdrfun devices)", select_name(select))
        }
        SdrSelect::Both => {
            let mut v = Vec::new();
            if present(SdrKind::Rtlsdr) {
                v.push(SdrKind::Rtlsdr);
            }
            if present(SdrKind::Hackrf) {
                v.push(SdrKind::Hackrf);
            }
            if v.is_empty() {
                bail!("no RTL-SDR or HackRF attached (sdrfun devices)");
            }
            v
        }
    };
    Ok(out)
}

fn select_name(s: SdrSelect) -> &'static str {
    match s {
        SdrSelect::Rtlsdr => "RTL-SDR",
        SdrSelect::Hackrf => "HackRF",
        SdrSelect::Both => "radio",
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gain {
    Auto,
    /// Tuner gain in dB; snapped to the nearest supported step.
    Manual(f32),
}

impl std::str::FromStr for Gain {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        if s.eq_ignore_ascii_case("auto") {
            Ok(Gain::Auto)
        } else {
            s.parse::<f32>()
                .map(Gain::Manual)
                .map_err(|_| format!("expected a number in dB or \"auto\", got {s:?}"))
        }
    }
}

impl std::fmt::Display for Gain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Gain::Auto => write!(f, "auto"),
            Gain::Manual(db) => write!(f, "{db}"),
        }
    }
}

/// Split a total gain into HackRF LNA (0–40 dB, 8 dB steps) + VGA (0–62 dB, 2 dB steps),
/// preferring the lower-noise LNA.
fn hackrf_gains(total_db: f32) -> (u32, u32) {
    let total = total_db.clamp(0.0, 102.0).round() as i32;
    let lna = (total / 8 * 8).clamp(0, 40) as u32;
    let vga = (total - lna as i32).clamp(0, 62) & !1;
    (lna, vga as u32)
}

pub(crate) struct RtlDevice {
    dev: *mut c_void,
}

pub(crate) struct HackRf {
    dev: *mut c_void,
    rx: Mutex<VecDeque<u8>>,
    streaming: bool,
    /// Last values applied, so `stream` can tag blocks correctly (libhackrf has no getters).
    last_center: Option<u32>,
    last_rate: Option<u32>,
}

// Device handles may be used from any single thread at a time.
unsafe impl Send for RtlDevice {}
unsafe impl Send for HackRf {}

/// An attached radio, RTL-SDR or HackRF, with one common interface.
pub enum Device {
    Rtlsdr(RtlDevice),
    Hackrf(HackRf),
}

unsafe impl Send for Device {}

fn check(ret: c_int, what: &str) -> Result<()> {
    if ret < 0 {
        bail!("radio {what} failed ({ret})");
    }
    Ok(())
}

impl Device {
    pub fn open(kind: SdrKind, index: u32) -> Result<Self> {
        match kind {
            SdrKind::Rtlsdr => {
                let count = unsafe { rtlsdr_get_device_count() };
                if index >= count {
                    bail!("RTL-SDR device #{index} not found ({count} device(s) attached)");
                }
                let mut dev = std::ptr::null_mut();
                let ret = unsafe { rtlsdr_open(&mut dev, index) };
                if ret < 0 || dev.is_null() {
                    bail!(
                        "cannot open RTL-SDR device #{index} ({ret}); is it in use, or are the udev rules missing?"
                    );
                }
                Ok(Self::Rtlsdr(RtlDevice { dev }))
            }
            SdrKind::Hackrf => {
                hackrf_init_once();
                let list = unsafe { hackrf_device_list() };
                if list.is_null() {
                    bail!("cannot list HackRF devices");
                }
                let n = unsafe { (*list).devicecount };
                let mut dev = std::ptr::null_mut();
                let ret = if index < n as u32 {
                    unsafe { hackrf_device_list_open(list, index as c_int, &mut dev) }
                } else {
                    -1
                };
                unsafe { hackrf_device_list_free(list) };
                if ret != 0 || dev.is_null() {
                    bail!(
                        "cannot open HackRF device #{index} ({ret}); is it in use, or are the udev rules missing?"
                    );
                }
                Ok(Self::Hackrf(HackRf {
                    dev,
                    rx: Mutex::new(VecDeque::new()),
                    streaming: false,
                    last_center: None,
                    last_rate: None,
                }))
            }
        }
    }

    pub fn kind(&self) -> SdrKind {
        match self {
            Device::Rtlsdr(_) => SdrKind::Rtlsdr,
            Device::Hackrf(_) => SdrKind::Hackrf,
        }
    }

    pub fn set_sample_rate(&mut self, rate: u32) -> Result<u32> {
        match self {
            Device::Rtlsdr(d) => {
                check(unsafe { rtlsdr_set_sample_rate(d.dev, rate) }, "set_sample_rate")?;
                Ok(unsafe { rtlsdr_get_sample_rate(d.dev) })
            }
            Device::Hackrf(d) => {
                // The HackRF One is an 8-bit I/Q radio that runs from 2 to 20 MSPS.
                let rate = rate.clamp(2_000_000, 20_000_000);
                check(unsafe { hackrf_set_sample_rate(d.dev, rate as f64) }, "set_sample_rate")?;
                d.last_rate = Some(rate);
                Ok(rate)
            }
        }
    }

    pub fn set_center_freq(&mut self, hz: u32) -> Result<u32> {
        match self {
            Device::Rtlsdr(d) => {
                check(unsafe { rtlsdr_set_center_freq(d.dev, hz) }, "set_center_freq")?;
                Ok(unsafe { rtlsdr_get_center_freq(d.dev) })
            }
            Device::Hackrf(d) => {
                check(unsafe { hackrf_set_freq(d.dev, hz as u64) }, "set_freq")?;
                d.last_center = Some(hz);
                Ok(hz)
            }
        }
    }

    pub fn set_ppm(&mut self, ppm: i32) -> Result<()> {
        match self {
            Device::Rtlsdr(d) => {
                if ppm == 0 {
                    // librtlsdr returns -2 when the correction is unchanged.
                    return Ok(());
                }
                check(unsafe { rtlsdr_set_freq_correction(d.dev, ppm) }, "set_freq_correction")
            }
            // The HackRF firmware has no frequency correction (no TCXO).
            Device::Hackrf(_) => Ok(()),
        }
    }

    /// Returns the gain actually applied, in dB (`None` for auto).
    pub fn set_gain(&mut self, gain: Gain) -> Result<Option<f32>> {
        let steps = self.gains_db();
        match self {
            Device::Rtlsdr(d) => {
                unsafe { rtlsdr_set_agc_mode(d.dev, 0) };
                match gain {
                    Gain::Auto => {
                        check(unsafe { rtlsdr_set_tuner_gain_mode(d.dev, 0) }, "set_tuner_gain_mode")?;
                        Ok(None)
                    }
                    Gain::Manual(db) => {
                        check(unsafe { rtlsdr_set_tuner_gain_mode(d.dev, 1) }, "set_tuner_gain_mode")?;
                        let best = steps
                            .iter()
                            .copied()
                            .min_by(|a, b| (a - db).abs().total_cmp(&(b - db).abs()))
                            .unwrap_or(db);
                        check(unsafe { rtlsdr_set_tuner_gain(d.dev, (best * 10.0).round() as c_int) }, "set_tuner_gain")?;
                        Ok(Some(best))
                    }
                }
            }
            Device::Hackrf(d) => {
                // No AGC: "auto" is a sensible fixed default (24 dB).
                let (lna, vga) = match gain {
                    Gain::Auto => (16, 8),
                    Gain::Manual(db) => hackrf_gains(db),
                };
                check(unsafe { hackrf_set_lna_gain(d.dev, lna) }, "set_lna_gain")?;
                check(unsafe { hackrf_set_vga_gain(d.dev, vga) }, "set_vga_gain")?;
                match gain {
                    Gain::Auto => Ok(None),
                    Gain::Manual(_) => Ok(Some((lna + vga) as f32)),
                }
            }
        }
    }

    /// Supported total gains in dB, for the page's slider.
    pub fn gains_db(&self) -> Vec<f32> {
        match self {
            Device::Rtlsdr(d) => {
                let n = unsafe { rtlsdr_get_tuner_gains(d.dev, std::ptr::null_mut()) };
                if n <= 0 {
                    return Vec::new();
                }
                let mut v = vec![0; n as usize];
                unsafe { rtlsdr_get_tuner_gains(d.dev, v.as_mut_ptr()) };
                v.iter().map(|&t| t as f32 / 10.0).collect()
            }
            Device::Hackrf(_) => (0..=50).map(|db| db as f32).collect(),
        }
    }

    /// Make the radio ready to read. The HackRF starts its callback pump. The RTL-SDR
    /// reads synchronously, but librtlsdr requires `rtlsdr_reset_buffer` before the
    /// first `rtlsdr_read_sync`; without it every read fails with LIBUSB_ERROR_PIPE (-9).
    pub fn start_streaming(&mut self) -> Result<()> {
        match self {
            Device::Rtlsdr(d) => check(unsafe { rtlsdr_reset_buffer(d.dev) }, "reset_buffer"),
            Device::Hackrf(d) => {
                if !d.streaming {
                    let ctx = &d.rx as *const Mutex<VecDeque<u8>> as *mut c_void;
                    check(unsafe { hackrf_start_rx(d.dev, hackrf_rx_cb, ctx) }, "start_rx")?;
                    d.streaming = true;
                }
                Ok(())
            }
        }
    }

    pub fn stop_streaming(&mut self) {
        if let Device::Hackrf(d) = self
            && d.streaming
        {
            unsafe { hackrf_stop_rx(d.dev) };
            d.streaming = false;
            d.rx.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }
    }

    /// Drop anything buffered in the dongle (after a retune).
    /// Drop everything captured so far, so the next read starts at fresh samples.
    pub fn reset_buffer(&mut self) {
        match self {
            Device::Rtlsdr(d) => {
                unsafe { rtlsdr_reset_buffer(d.dev) };
            }
            Device::Hackrf(d) => d.rx.lock().unwrap_or_else(|e| e.into_inner()).clear(),
        }
    }

    /// After a retune: discard everything captured at the previous frequency.
    /// The HackRF keeps four 131k-sample URBs in flight that a plain queue clear
    /// cannot reach, so stop and restart the stream to cancel them; only the
    /// hardware FIFO survives.
    pub fn purge(&mut self) {
        match self {
            Device::Rtlsdr(d) => {
                unsafe { rtlsdr_reset_buffer(d.dev) };
            }
            Device::Hackrf(d) => {
                if d.streaming {
                    unsafe { hackrf_stop_rx(d.dev) };
                    let ctx = &d.rx as *const Mutex<VecDeque<u8>> as *mut c_void;
                    let _ = unsafe { hackrf_start_rx(d.dev, hackrf_rx_cb, ctx) };
                }
                d.rx.lock().unwrap_or_else(|e| e.into_inner()).clear();
            }
        }
    }

    /// Read `samples` IQ pairs, waiting up to `timeout` for them.
    pub fn read(&mut self, samples: usize, timeout: Duration) -> Result<Vec<u8>> {
        match self {
            Device::Rtlsdr(d) => {
                let mut buf = vec![0u8; 2 * samples];
                let mut n: c_int = 0;
                let ret = unsafe {
                    rtlsdr_read_sync(d.dev, buf.as_mut_ptr().cast(), (2 * samples) as c_int, &mut n)
                };
                check(ret, "read_sync")?;
                buf.truncate(n.max(0) as usize);
                Ok(buf)
            }
            Device::Hackrf(d) => {
                let want = 2 * samples;
                let deadline = Instant::now() + timeout;
                loop {
                    let got = {
                        let mut b = d.rx.lock().unwrap_or_else(|e| e.into_inner());
                        if b.len() >= want {
                            Some(b.drain(..want).collect::<Vec<u8>>())
                        } else {
                            None
                        }
                    };
                    if let Some(v) = got {
                        return Ok(v);
                    }
                    if Instant::now() > deadline {
                        let mut b = d.rx.lock().unwrap_or_else(|e| e.into_inner());
                        let v = b.drain(..).collect::<Vec<u8>>();
                        if !v.is_empty() {
                            return Ok(v);
                        }
                        bail!("HackRF produced no samples");
                    }
                    thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }

    /// Read IQ blocks until `stop` is set or the receiver hangs up. Changes sent on
    /// `control` are applied between blocks; the latest of each kind wins.
    pub fn stream(
        mut self,
        tx: SyncSender<Block>,
        stop: Arc<AtomicBool>,
        control: Option<Receiver<Control>>,
    ) -> Result<()> {
        let mut center_hz = match &self {
            Device::Rtlsdr(d) => unsafe { rtlsdr_get_center_freq(d.dev) },
            Device::Hackrf(d) => d.last_center.unwrap_or(0),
        };
        let mut rate = match &self {
            Device::Rtlsdr(d) => unsafe { rtlsdr_get_sample_rate(d.dev) },
            Device::Hackrf(d) => d.last_rate.unwrap_or(2_400_000),
        };
        let mut settings = 0u32;
        self.start_streaming()?;
        while !stop.load(Ordering::Relaxed) {
            let (mut want_center, mut want_gain, mut want_rate) = (None, None, None);
            for c in control.iter().flat_map(|r| r.try_iter()) {
                match c {
                    Control::Center(hz) => want_center = Some(hz),
                    Control::Gain(g) => want_gain = Some(g),
                    Control::Rate(r) => want_rate = Some(r),
                    Control::Switch(_) | Control::Sweep(_) | Control::Levels(_) | Control::Bins(_) => {}
                }
            }
            // A rejected change (e.g. a frequency the tuner cannot reach) keeps the old
            // setting rather than stopping the stream; consumers see it in the blocks.
            let mut settle = false;
            if let Some(g) = want_gain {
                match self.set_gain(g) {
                    Ok(_) => {
                        settings = settings.wrapping_add(1);
                        settle = true;
                    }
                    Err(e) => eprintln!("\r\x1b[Kwarning: gain {g}: {e:#}"),
                }
            }
            if settle || want_center.is_some() || want_rate.is_some() {
                if let Some(r) = want_rate {
                    match self.set_sample_rate(r) {
                        Ok(r) => {
                            rate = r;
                            settings = settings.wrapping_add(1);
                        }
                        Err(e) => eprintln!("\r\x1b[Kwarning: sample rate {r}: {e:#}"),
                    }
                }
                if let Some(hz) = want_center {
                    match self.set_center_freq(hz) {
                        Ok(hz) => center_hz = hz,
                        Err(e) => eprintln!("\r\x1b[Kwarning: tuning to {hz} Hz: {e:#}"),
                    }
                }
                match &self {
                    Device::Rtlsdr(d) => {
                        check(unsafe { rtlsdr_reset_buffer(d.dev) }, "reset_buffer")?;
                    }
                    Device::Hackrf(_) => {}
                }
                // Discard one block while the PLL and gain settle.
                self.read(READ_LEN / 2, Duration::from_secs(2))?;
            }
            let data = self.read(READ_LEN / 2, Duration::from_secs(2))?;
            if tx.send(Block { center_hz, rate, settings, data }).is_err() {
                break;
            }
        }
        self.stop_streaming();
        Ok(())
    }
}

impl Drop for RtlDevice {
    fn drop(&mut self) {
        if !self.dev.is_null() {
            unsafe { rtlsdr_close(self.dev) };
            self.dev = std::ptr::null_mut();
        }
    }
}

/// The radios and settings the web reader thread (`pump`) drives.
pub struct Rig {
    pub sources: Vec<Device>,
    pub active: usize,
    pub center: u32,
    pub rate: u32,
    pub gain: Gain,
    pub ppm: i32,
    pub sweep: Option<(u32, u32)>,
    /// Waterfall dB range (None = auto-select once the radio has settled).
    pub levels: Option<(f32, f32)>,
    /// Bins per sweep row (the page's FFT size).
    pub bins: usize,
}

impl Rig {
    fn apply_all(&mut self) -> Result<()> {
        let d = &mut self.sources[self.active];
        d.set_sample_rate(self.rate)?;
        d.set_center_freq(self.center)?;
        d.set_gain(self.gain)?;
        d.set_ppm(self.ppm)?;
        Ok(())
    }
}

/// The web reader loop: stream IQ from the active radio, or sweep a wide range when
/// asked, and switch between attached radios — all driven by `Control` messages.
pub fn pump(
    mut rig: Rig,
    tx: SyncSender<Capture>,
    stop: Arc<AtomicBool>,
    control: Receiver<Control>,
) -> Result<()> {
    let mut settings = 0u32;
    let mut sweeper = crate::sweep::Sweeper::new(rig.sources[rig.active].kind());
    sweeper.set_bins(rig.bins);
    rig.apply_all()?;
    let (mut cur_center, mut cur_rate, mut cur_gain) = (rig.center, rig.rate, rig.gain);
    rig.sources[rig.active].start_streaming()?;
    let mut sweeping = rig.sweep.is_some();
    while !stop.load(Ordering::Relaxed) {
        for c in control.try_iter() {
            match c {
                Control::Center(hz) => rig.center = hz,
                Control::Rate(r) => rig.rate = r,
                Control::Gain(g) => rig.gain = g,
                Control::Switch(kind) => {
                    if let Some(i) = rig.sources.iter().position(|d| d.kind() == kind)
                        && i != rig.active
                    {
                        rig.sources[rig.active].stop_streaming();
                        rig.active = i;
                        sweeper = crate::sweep::Sweeper::new(kind);
                        sweeper.set_bins(rig.bins);
                        rig.apply_all()?;
                        (cur_center, cur_rate, cur_gain) = (rig.center, rig.rate, rig.gain);
                        settings = settings.wrapping_add(1);
                        rig.sources[i].start_streaming()?;
                    }
                }
                Control::Sweep(range) => {
                    if rig.sweep != range {
                        sweeper.reset_floor();
                    }
                    rig.sweep = range;
                }
                Control::Levels(v) => {
                    rig.levels = v;
                    sweeper.set_levels(v);
                }
                Control::Bins(n) => {
                    rig.bins = n;
                    sweeper.set_bins(n);
                }
            }
        }
        let want = rig.sweep.is_some();
        if want != sweeping {
            sweeping = want;
            sweeper.set_levels(rig.levels);
            // The sweep left the tuner and filters somewhere else; go back to the channel
            // and drop what was captured on the way.
            rig.apply_all()?;
            (cur_center, cur_rate, cur_gain) = (rig.center, rig.rate, rig.gain);
            settings = settings.wrapping_add(1);
            if !sweeping {
                rig.sources[rig.active].reset_buffer();
                let _ = rig.sources[rig.active].read(READ_LEN / 2, Duration::from_secs(1));
            }
        }
        if let Some((a, b)) = rig.sweep {
            // Sweeps always run with a fixed gain: an AGC (or the low auto default)
            // would band or flatten the waterfall from pass to pass.
            let want_gain = match rig.gain {
                Gain::Auto => rig.sources[rig.active].kind().sweep_gain(),
                manual => manual,
            };
            if want_gain != cur_gain {
                let _ = rig.sources[rig.active].set_gain(want_gain);
                cur_gain = want_gain;
            }
            match sweeper.sweep(&mut rig.sources[rig.active], a, b, &stop) {
                Some(row) => {
                    if tx.send(Capture::SweepRow(row)).is_err() {
                        break;
                    }
                }
                None => continue, // stopped mid-sweep
            }
        } else {
            if rig.rate != cur_rate || rig.gain != cur_gain {
                let d = &mut rig.sources[rig.active];
                let _ = d.set_sample_rate(rig.rate);
                let _ = d.set_gain(rig.gain);
                (cur_rate, cur_gain) = (rig.rate, rig.gain);
                settings = settings.wrapping_add(1);
            }
            if rig.center != cur_center {
                let _ = rig.sources[rig.active].set_center_freq(rig.center);
                cur_center = rig.center;
                rig.sources[rig.active].reset_buffer();
                let _ = rig.sources[rig.active].read(READ_LEN / 2, Duration::from_secs(1));
            }
            let data = match rig.sources[rig.active].read(READ_LEN / 2, Duration::from_secs(2)) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("warning: radio read failed: {e:#}");
                    thread::sleep(Duration::from_millis(100));
                    continue;
                }
            };
            let block = Block { center_hz: cur_center, rate: cur_rate, settings, data };
            if tx.send(Capture::Block(block)).is_err() {
                break;
            }
        }
    }
    for d in rig.sources.iter_mut() {
        d.stop_streaming();
    }
    Ok(())
}

impl Drop for HackRf {
    fn drop(&mut self) {
        if self.streaming {
            unsafe { hackrf_stop_rx(self.dev) };
            self.streaming = false;
        }
        if !self.dev.is_null() {
            unsafe { hackrf_close(self.dev) };
            self.dev = std::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn kind_and_selection_parse() {
        assert_eq!(SdrKind::from_name("HackRF"), Some(SdrKind::Hackrf));
        assert_eq!(SdrKind::from_name("rtl-sdr"), Some(SdrKind::Rtlsdr));
        assert_eq!(SdrKind::from_name("bladerf"), None);
        assert_eq!(SdrSelect::from_str("HACKRF").unwrap(), SdrSelect::Hackrf);
        assert_eq!(SdrSelect::from_str("Both").unwrap(), SdrSelect::Both);
        assert!(SdrSelect::from_str("nope").is_err());
    }

    #[test]
    fn env_selects_when_no_flag() {
        // SAFETY: single-threaded test; SDRFUN_SDR is only read by this test.
        unsafe { std::env::set_var("SDRFUN_SDR", "hackrf") };
        assert_eq!(SdrSelect::resolve(None), SdrSelect::Hackrf);
        assert_eq!(SdrSelect::resolve(Some(SdrSelect::Rtlsdr)), SdrSelect::Rtlsdr);
        unsafe { std::env::remove_var("SDRFUN_SDR") };
        assert_eq!(SdrSelect::resolve(None), SdrSelect::Both);
    }

    #[test]
    fn hackrf_gain_split_prefers_lna() {
        assert_eq!(hackrf_gains(0.0), (0, 0));
        assert_eq!(hackrf_gains(24.0), (24, 0));
        assert_eq!(hackrf_gains(30.0), (24, 6));
        assert_eq!(hackrf_gains(50.0), (40, 10));
        assert_eq!(hackrf_gains(120.0), (40, 62));
    }

    #[test]
    fn freq_ranges_and_sweep_plans_are_sane() {
        for k in [SdrKind::Rtlsdr, SdrKind::Hackrf] {
            let (lo, hi) = k.freq_range();
            let (rate, hop, fft) = k.sweep_plan();
            assert!(lo < hi && hop < rate && fft.is_power_of_two());
        }
        assert!(SdrKind::Hackrf.sweep_plan().1 > SdrKind::Rtlsdr.sweep_plan().1);
    }
}
