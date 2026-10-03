use std::path::Path;

use anyhow::{Context, Result, bail};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

/// Read any PCM/float WAV and downmix it to mono f32 in [-1, 1].
pub fn read_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader =
        WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        SampleFormat::Int => {
            if spec.bits_per_sample == 0 || spec.bits_per_sample > 32 {
                bail!("unsupported bit depth {}", spec.bits_per_sample);
            }
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono = interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok((mono, spec.sample_rate))
}

pub fn write_mono_i16(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer =
        WavWriter::create(path, spec).with_context(|| format!("creating {}", path.display()))?;
    for &s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Peak and RMS level in dBFS.
pub fn levels_dbfs(samples: &[f32]) -> (f32, f32) {
    let peak = samples.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len().max(1) as f32).sqrt();
    let db = |v: f32| 20.0 * v.max(1e-10).log10();
    (db(peak), db(rms))
}
