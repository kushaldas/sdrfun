//! Minimal safe wrapper over the system librtlsdr.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;

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

/// Bytes per synchronous read: 128 Ki IQ pairs, ~55 ms at 2.4 MS/s.
const READ_LEN: usize = 16 * 16384;

pub struct DeviceInfo {
    pub index: u32,
    pub name: String,
    pub manufacturer: String,
    pub product: String,
    pub serial: String,
}

pub fn list_devices() -> Vec<DeviceInfo> {
    let count = unsafe { rtlsdr_get_device_count() };
    (0..count)
        .map(|index| {
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
            DeviceInfo {
                index,
                name,
                manufacturer: text(&bufs[0]),
                product: text(&bufs[1]),
                serial: text(&bufs[2]),
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
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

pub struct Device {
    dev: *mut c_void,
}

// librtlsdr device handles may be used from any single thread at a time.
unsafe impl Send for Device {}

fn check(ret: c_int, what: &str) -> Result<()> {
    if ret < 0 {
        bail!("librtlsdr {what} failed ({ret})");
    }
    Ok(())
}

impl Device {
    pub fn open(index: u32) -> Result<Self> {
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
        Ok(Self { dev })
    }

    pub fn set_sample_rate(&mut self, rate: u32) -> Result<u32> {
        check(unsafe { rtlsdr_set_sample_rate(self.dev, rate) }, "set_sample_rate")?;
        Ok(unsafe { rtlsdr_get_sample_rate(self.dev) })
    }

    pub fn set_center_freq(&mut self, hz: u32) -> Result<u32> {
        check(unsafe { rtlsdr_set_center_freq(self.dev, hz) }, "set_center_freq")?;
        Ok(unsafe { rtlsdr_get_center_freq(self.dev) })
    }

    pub fn set_ppm(&mut self, ppm: i32) -> Result<()> {
        if ppm == 0 {
            // librtlsdr returns -2 when the correction is unchanged.
            return Ok(());
        }
        check(unsafe { rtlsdr_set_freq_correction(self.dev, ppm) }, "set_freq_correction")
    }

    /// Returns the gain actually applied, in dB (`None` for auto).
    pub fn set_gain(&mut self, gain: Gain) -> Result<Option<f32>> {
        unsafe { rtlsdr_set_agc_mode(self.dev, 0) };
        match gain {
            Gain::Auto => {
                check(unsafe { rtlsdr_set_tuner_gain_mode(self.dev, 0) }, "set_tuner_gain_mode")?;
                Ok(None)
            }
            Gain::Manual(db) => {
                check(unsafe { rtlsdr_set_tuner_gain_mode(self.dev, 1) }, "set_tuner_gain_mode")?;
                let steps = self.gains();
                let wanted = (db * 10.0).round() as c_int;
                let tenths = steps
                    .iter()
                    .copied()
                    .min_by_key(|g| (g - wanted).abs())
                    .unwrap_or(wanted);
                check(unsafe { rtlsdr_set_tuner_gain(self.dev, tenths) }, "set_tuner_gain")?;
                Ok(Some(tenths as f32 / 10.0))
            }
        }
    }

    /// Supported tuner gains in tenths of a dB.
    pub fn gains(&self) -> Vec<c_int> {
        let n = unsafe { rtlsdr_get_tuner_gains(self.dev, std::ptr::null_mut()) };
        if n <= 0 {
            return Vec::new();
        }
        let mut v = vec![0; n as usize];
        unsafe { rtlsdr_get_tuner_gains(self.dev, v.as_mut_ptr()) };
        v
    }

    /// Read IQ blocks until `stop` is set or the receiver hangs up.
    pub fn stream(mut self, tx: SyncSender<Vec<u8>>, stop: Arc<AtomicBool>) -> Result<()> {
        check(unsafe { rtlsdr_reset_buffer(self.dev) }, "reset_buffer")?;
        while !stop.load(Ordering::Relaxed) {
            let mut buf = vec![0u8; READ_LEN];
            let mut n: c_int = 0;
            let ret = unsafe {
                rtlsdr_read_sync(self.dev, buf.as_mut_ptr().cast(), READ_LEN as c_int, &mut n)
            };
            check(ret, "read_sync")?;
            buf.truncate(n.max(0) as usize);
            if tx.send(buf).is_err() {
                break;
            }
        }
        self.close();
        Ok(())
    }

    fn close(&mut self) {
        if !self.dev.is_null() {
            unsafe { rtlsdr_close(self.dev) };
            self.dev = std::ptr::null_mut();
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.close();
    }
}
