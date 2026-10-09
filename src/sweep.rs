//! Spectrum sweeps: hop the tuner across a range far wider than the receiver's IQ
//! bandwidth and FFT each stop, like `hackrf_sweep` does. Used by `sdrfun web` in sweep
//! mode; the HackRF hops 16 MHz at a time (20 MS/s), the RTL-SDR 2 MHz (2.4 MS/s).
//!
//! A finished sweep becomes one waterfall row (message type `SWEEP_ROW`) framed exactly
//! like the wideband rows of `web.rs`, so the page draws it with the same code.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use rustfft::{Fft, FftPlanner, num_complex::Complex32};

use crate::adpcm;
use crate::sdr::{Device, SdrKind};
use crate::spectrum::Levels;
use crate::web::{ROW_HEADER, ROW_SCALE};

/// Binary message type for a sweep row.
pub const SWEEP_ROW: u8 = 6;
/// Bins per finished row until the page picks an FFT size.
const DEFAULT_BINS: usize = 2048;
/// FFTs averaged per hop.
const AVERAGES: usize = 4;

pub struct Sweeper {
    rate: u32,
    hop: u32,
    n: usize,
    /// Bins kept per hop: the central `hop` Hz of the FFT. The skirt of a windowed FFT
    /// would leave a dip at every hop boundary, so no window is applied and only the
    /// flat middle is used; hops then tile the range edge to edge.
    m: usize,
    /// Samples to drop after each hop: everything still in the USB pipeline (or queued
    /// behind it) was captured at the previous hop's frequency.
    flush: usize,
    fft: Arc<dyn Fft<f32>>,
    scale: f32,
    hi_hz: u32,
    levels: Levels,
    enc: adpcm::Encoder,
    /// Running average of per-hop noise floors; each hop is shifted to it so the
    /// radio's per-frequency gain and offset differences don't stripe the waterfall.
    floor_ema: f32,
    floored: bool,
    /// Bins per finished row at most (a power of two: the page's FFT size).
    bins: usize,
}

impl Sweeper {
    pub fn new(kind: SdrKind) -> Self {
        let (rate, hop, n) = kind.sweep_plan();
        let hi_hz = kind.freq_range().1 as u32;
        Self {
            rate,
            hop,
            n,
            m: ((n as u64 * hop as u64 / rate as u64) as usize).clamp(16, n),
            // librtlsdr flushes its own transfers on reset; the HackRF restart in
            // purge() cancels the in-flight URBs, so only a short FIFO drain remains.
            flush: match kind {
                SdrKind::Rtlsdr => (rate / 250).max(64) as usize,
                SdrKind::Hackrf => 100_000,
            },
            fft: FftPlanner::new().plan_fft_forward(n),
            // Rectangular window: a full-scale sine puts n^2 in its bin.
            scale: 1.0 / (n * n) as f32,
            hi_hz,
            levels: {
                let mut l = Levels::default();
                l.settle = 3;
                l
            },
            enc: adpcm::Encoder::default(),
            floor_ema: 0.0,
            floored: false,
            bins: DEFAULT_BINS,
        }
    }

    /// Freeze or re-arm the painted dB range: `None` follows automatically until
    /// settled, `Some((lo, hi))` is the user's choice.
    pub fn set_levels(&mut self, v: Option<(f32, f32)>) {
        self.levels = Levels::default();
        self.levels.settle = 3;
        if let Some((lo, hi)) = v {
            self.levels.set_manual(lo, hi);
        }
        self.floored = false;
    }

    /// Bins per finished row at most; rows are always a power of two wide.
    pub fn set_bins(&mut self, n: usize) {
        self.bins = n.max(1);
    }

    /// Forget the learned noise floor, e.g. when the sweep range changes.
    pub fn reset_floor(&mut self) {
        self.floored = false;
    }

    /// One full pass over `[start, end]`; `None` if `stop` was set mid-pass.
    pub fn sweep(
        &mut self,
        dev: &mut Device,
        start: u32,
        end: u32,
        stop: &AtomicBool,
    ) -> Option<Vec<u8>> {
        let _ = dev.set_sample_rate(self.rate);
        let skip = (self.n - self.m) / 2;
        let mut levels: Vec<f32> = Vec::new();
        let span = end - start;
        let mut covered = 0u32;
        let mut buf = vec![Complex32::new(0.0, 0.0); self.n];
        let mut acc = vec![0.0f32; self.n];
        // Hop in whole `hop`-wide tiles until the range is covered; the last tile is
        // trimmed below. The centre stays inside the tuner's reach.
        while covered < span {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let f = (start + covered + self.hop / 2).min(self.hi_hz - self.rate / 2);
            let _ = dev.set_center_freq(f);
            // Cancel the backlog and the in-flight transfers, then drain what the
            // hardware FIFO still holds; what is left is fresh post-retune samples.
            dev.purge();
            let _ = dev.read(self.flush, Duration::from_millis(500));
            acc.iter_mut().for_each(|a| *a = 0.0);
            for _ in 0..AVERAGES {
                let iq = dev.read(self.n, Duration::from_millis(500)).ok()?;
                for (z, p) in buf.iter_mut().zip(iq.as_chunks::<2>().0) {
                    *z = Complex32::new((p[0] as f32 - 127.4) / 128.0, (p[1] as f32 - 127.4) / 128.0);
                }
                self.fft.process(&mut buf);
                for (a, z) in acc.iter_mut().zip(&buf) {
                    *a += z.norm_sqr();
                }
            }
            let half = self.n / 2;
            let k = self.scale / AVERAGES as f32;
            let mut row: Vec<f32> = (0..self.m)
                .map(|j| 10.0 * (acc[(skip + j + half) % self.n] * k).max(1e-15).log10())
                .collect();
            // The DC spike sits in the middle of every hop; hide it.
            let c = self.m / 2;
            row[c - 1] = row[c - 3];
            row[c] = row[c - 3];
            row[c + 1] = row[c + 3];
            // Shift this hop's noise floor onto the running average so per-frequency
            // gain/offset differences don't stripe the waterfall.
            let mut sorted = row.clone();
            sorted.sort_by(f32::total_cmp);
            let med = sorted[sorted.len() / 2];
            if !self.floored {
                (self.floor_ema, self.floored) = (med, true);
            }
            let adj = self.floor_ema - med;
            for v in row.iter_mut() {
                *v += adj;
            }
            self.floor_ema += 0.2 * (med - self.floor_ema);
            levels.extend(row);
            covered += self.hop;
        }
        if levels.is_empty() {
            return None;
        }
        // The last hop usually reaches past `end`; trim so the row covers exactly
        // [start, end] and signals sit at the right frequency at any zoom.
        levels.truncate(kept_bins(end - start, self.hop, self.m).min(levels.len()));
        let row = resample(&levels, row_bins(levels.len(), self.bins));
        let center = ((start as u64 + end as u64) / 2) as u32;
        Some(frame_row(center, end - start, &row, &mut self.levels, &mut self.enc))
    }
}

/// Bins covering `span_hz`, given `m` bins per `hop_hz` wide hop.
fn kept_bins(span_hz: u32, hop_hz: u32, m: usize) -> usize {
    ((span_hz as f64 * m as f64 / hop_hz as f64) as usize).max(1)
}

/// Frame one finished sweep row exactly like the wideband rows of `web.rs`, sharing
/// one smoothed (or user-locked) level range across rows so the floor never jumps.
pub fn frame_row(center_hz: u32, span_hz: u32, row: &[f32], levels: &mut Levels, enc: &mut adpcm::Encoder) -> Vec<u8> {
    let q = levels.quantize(row);
    let (lo, hi) = (levels.lo, levels.hi);
    let mut msg = Vec::with_capacity(ROW_HEADER + q.len() / 2 + 8);
    msg.push(SWEEP_ROW);
    msg.extend_from_slice(&center_hz.to_le_bytes());
    msg.extend_from_slice(&span_hz.to_le_bytes());
    msg.extend_from_slice(&lo.to_le_bytes());
    msg.extend_from_slice(&hi.to_le_bytes());
    msg.extend_from_slice(&(q.len() as u16).to_le_bytes());
    let samples: Vec<i16> = q.iter().map(|&l| l as i16 * ROW_SCALE).collect();
    msg.extend(enc.encode(&samples));
    msg
}

/// Bins to send for a row of `len` measured bins: a power of two, at most `max` (the
/// FFT size), and no more than `len` needs.
pub(crate) fn row_bins(len: usize, max: usize) -> usize {
    len.next_power_of_two().min(max.next_power_of_two()).max(1)
}

/// Resample `levels` to exactly `n` bins: each output bin averages the measured bins
/// under it (display resolution, so averaging dB values directly is close enough), or
/// takes the nearest one where there are fewer measured bins than output bins.
pub(crate) fn resample(levels: &[f32], n: usize) -> Vec<f32> {
    let len = levels.len();
    if len == n || len == 0 {
        return levels.to_vec();
    }
    (0..n)
        .map(|k| {
            let a = k * len / n;
            let b = ((k + 1) * len / n).max(a + 1).min(len);
            levels[a..b].iter().sum::<f32>() / (b - a) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kept_bins_cover_the_range_exactly() {
        // HackRF plan: 819 bins per 16 MHz hop = 19.53 kHz bins.
        assert_eq!(kept_bins(20_000_000, 16_000_000, 819), 1023);
        assert_eq!(kept_bins(16_000_000, 16_000_000, 819), 819);
        // A range shorter than one hop still keeps bins.
        assert_eq!(kept_bins(1_000_000, 16_000_000, 819), 51);
    }

    #[test]
    fn a_strong_row_cannot_wreck_the_next_rows_scale() {
        let mut enc = adpcm::Encoder::default();
        let mut levels = Levels::default();
        // A quiet row, one with a huge single carrier, then the quiet row again:
        // the shared range must stay sane throughout (percentile floor and top).
        let quiet: Vec<f32> = (0..512).map(|i| -90.0 + (i % 5) as f32).collect();
        let loud: Vec<f32> = quiet.iter().enumerate().map(|(i, v)| if i == 300 { 0.0 } else { *v }).collect();
        let mut prev: Option<f32> = None;
        for row in [&quiet, &loud, &quiet] {
            let msg = frame_row(100_000_000, 2_000_000, row, &mut levels, &mut enc);
            let lo = f32::from_le_bytes(msg[9..13].try_into().unwrap());
            let hi = f32::from_le_bytes(msg[13..17].try_into().unwrap());
            assert!(hi - lo >= 30.0, "span {} dB too narrow", hi - lo);
            assert!(hi <= 0.1, "peak above full scale");
            if let Some(p) = prev {
                assert!((lo - p).abs() < 10.0, "range jumped {} dB between rows", lo - p);
            }
            prev = Some(lo);
        }
    }

    #[test]
    fn rows_are_a_power_of_two_wide() {
        // HackRF 20 MHz sweep: 1023 measured bins → one 1024-bin row.
        assert_eq!(row_bins(1023, 2048), 1024);
        assert_eq!(row_bins(5000, 2048), 2048);
        assert_eq!(row_bins(5000, 8192), 8192);
        assert_eq!(row_bins(51, 4096), 64);
        let v: Vec<f32> = (0..100).map(|i| i as f32).collect();
        assert_eq!(resample(&v, 64).len(), 64);
        let even: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let d = resample(&even, 64);
        assert!((d[0] - 0.5).abs() < 1e-6, "first bin averages 0 and 1: {}", d[0]);
        assert!((d[63] - 126.5).abs() < 1e-6);
        let up = resample(&v, 128);
        assert_eq!(up.len(), 128);
        assert!(up.windows(2).all(|w| w[1] >= w[0]), "upsampling keeps the order");
        assert_eq!(resample(&v, 100), v);
    }

    #[test]
    fn kept_bins_tile_the_hop_without_gaps() {
        for kind in [SdrKind::Rtlsdr, SdrKind::Hackrf] {
            let s = Sweeper::new(kind);
            let bin_hz = s.rate as f64 / s.n as f64;
            // The kept bins must cover the hop width, and not spill past it either.
            let covered = s.m as f64 * bin_hz;
            assert!(covered <= s.hop as f64 + bin_hz, "{kind:?} hops overlap");
            assert!(covered + bin_hz >= s.hop as f64, "{kind:?} leaves a gap");
        }
    }

    #[test]
    fn sweep_plans_cover_the_stated_bins() {
        for kind in [SdrKind::Rtlsdr, SdrKind::Hackrf] {
            let (rate, hop, n) = kind.sweep_plan();
            // One hop's FFT covers `rate`, binned every rate/n Hz.
            let bin_hz = rate as f64 / n as f64;
            assert!(bin_hz < 50e3, "{kind:?} bin {bin_hz} Hz too wide");
            assert!(hop <= rate);
        }
    }
}
