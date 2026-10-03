//! Streaming WAV output at the recording rate, plus file naming.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use hound::{SampleFormat, WavSpec, WavWriter};

use crate::clean::SAMPLE_RATE;
use crate::dsp::resample::Resampler;
use crate::gate::{Transmission, Verdict};

pub const RECORD_RATE: u32 = 16_000;

/// Writes 48 kHz audio to a 16 kHz mono 16-bit WAV, resampling on the fly.
pub struct StreamWriter {
    path: PathBuf,
    resampler: Resampler,
    writer: WavWriter<BufWriter<File>>,
    scratch: Vec<f32>,
}

impl StreamWriter {
    pub fn create(path: PathBuf) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let spec = WavSpec {
            channels: 1,
            sample_rate: RECORD_RATE,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let writer =
            WavWriter::create(&path, spec).with_context(|| format!("creating {}", path.display()))?;
        Ok(Self {
            path,
            resampler: Resampler::new(SAMPLE_RATE, RECORD_RATE),
            writer,
            scratch: Vec::new(),
        })
    }

    pub fn write(&mut self, audio: &[f32]) -> Result<()> {
        self.scratch.clear();
        self.resampler.process(audio, &mut self.scratch);
        self.write_scratch()
    }

    fn write_scratch(&mut self) -> Result<()> {
        for &s in &self.scratch {
            self.writer
                .write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)?;
        }
        Ok(())
    }

    /// Flush the resampler tail and close the file; returns its path.
    pub fn finish(mut self) -> Result<PathBuf> {
        self.scratch.clear();
        self.resampler.flush(&mut self.scratch);
        self.write_scratch()?;
        self.writer.finalize()?;
        Ok(self.path)
    }
}

/// One JSON object per line for every transmission, kept or dropped.
pub struct TransmissionLog {
    file: File,
}

impl TransmissionLog {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        Ok(Self { file })
    }

    pub fn append(
        &mut self,
        freq_mhz: f64,
        start: DateTime<Utc>,
        tx: &Transmission,
        file: Option<&Path>,
    ) -> Result<()> {
        let round1 = |v: f32| (v * 10.0).round() / 10.0;
        let entry = serde_json::json!({
            "start": start.to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            "freq_mhz": freq_mhz,
            "duration_s": round1(tx.seconds()),
            "carrier_s": round1(tx.carrier_s),
            "peak_dbfs": round1(tx.peak_db),
            "floor_dbfs": round1(tx.floor_db),
            "snr_db": round1(tx.peak_db - tx.floor_db),
            "voiced_ratio": (tx.voiced_ratio * 100.0).round() / 100.0,
            "verdict": match tx.verdict {
                Verdict::Kept => "kept",
                Verdict::TooShort => "too_short",
                Verdict::NoVoice => "no_voice",
            },
            "file": file.map(|p| p.display().to_string()),
        });
        writeln!(self.file, "{entry}")?;
        Ok(())
    }
}

/// `<out>/<YYYY-MM-DD>/<freq>_<YYYYMMDDTHHMMSSZ>_<tag>.wav`, all in UTC.
pub fn recording_path(out_dir: &Path, freq_mhz: f64, start: DateTime<Utc>, tag: &str) -> PathBuf {
    out_dir.join(start.format("%Y-%m-%d").to_string()).join(format!(
        "{freq_mhz:.3}_{}_{tag}.wav",
        start.format("%Y%m%dT%H%M%SZ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn path_layout() {
        let t = Utc.with_ymd_and_hms(2026, 10, 3, 13, 50, 52).unwrap();
        assert_eq!(
            recording_path(Path::new("recordings"), 120.15, t, "014s"),
            PathBuf::from("recordings/2026-10-03/120.150_20261003T135052Z_014s.wav")
        );
    }
}
