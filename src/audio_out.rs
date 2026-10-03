//! Speaker output through cpal (ALSA, which PipeWire/PulseAudio also serve).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, Stream, StreamConfig};

use crate::clean::SAMPLE_RATE;
use crate::dsp::resample::Resampler;

/// Drop the oldest audio if more than this is queued, so latency cannot grow
/// without bound when the sound card runs slower than the SDR clock.
const MAX_QUEUED_S: usize = 2;

type Queue = Arc<Mutex<VecDeque<f32>>>;

pub struct AudioOut {
    queue: Queue,
    _stream: Stream,
    resampler: Option<Resampler>,
    scratch: Vec<f32>,
    volume: f32,
    max_queued: usize,
    pub description: String,
}

impl AudioOut {
    /// Open `device` (a substring of its name, or "default") and start playing silence.
    pub fn open(device: &str, volume: f32) -> Result<Self> {
        let host = cpal::default_host();
        let dev = if device == "default" {
            host.default_output_device()
                .ok_or_else(|| anyhow!("no default audio output device"))?
        } else {
            host.output_devices()?
                .find(|d| d.to_string().contains(device))
                .ok_or_else(|| anyhow!("no audio output device matching {device:?} (see `sdrfun devices`)"))?
        };

        // Prefer 48 kHz so no resampling is needed, f32 if offered.
        let mut candidates: Vec<_> = dev
            .supported_output_configs()
            .context("querying audio output formats")?
            .filter(|c| (c.min_sample_rate()..=c.max_sample_rate()).contains(&SAMPLE_RATE))
            .collect();
        candidates.sort_by_key(|c| c.sample_format() != SampleFormat::F32);
        let supported = match candidates.into_iter().next() {
            Some(c) => c.with_sample_rate(SAMPLE_RATE),
            None => dev.default_output_config().context("querying default audio format")?,
        };
        let format = supported.sample_format();
        let config: StreamConfig = supported.into();
        let rate = config.sample_rate;

        let queue: Queue = Arc::new(Mutex::new(VecDeque::new()));
        let stream = match format {
            SampleFormat::F32 => build::<f32>(&dev, &config, queue.clone()),
            SampleFormat::I16 => build::<i16>(&dev, &config, queue.clone()),
            SampleFormat::I32 => build::<i32>(&dev, &config, queue.clone()),
            SampleFormat::U16 => build::<u16>(&dev, &config, queue.clone()),
            SampleFormat::U8 => build::<u8>(&dev, &config, queue.clone()),
            other => bail!("unsupported audio sample format {other}"),
        }?;
        stream.play().context("starting audio output")?;

        Ok(Self {
            queue,
            _stream: stream,
            resampler: (rate != SAMPLE_RATE).then(|| Resampler::new(SAMPLE_RATE, rate)),
            scratch: Vec::new(),
            volume,
            max_queued: MAX_QUEUED_S * rate as usize,
            description: format!("{dev} ({rate} Hz, {} ch, {format})", config.channels),
        })
    }

    /// Seconds of audio still waiting to be played.
    pub fn queued_seconds(&self) -> f32 {
        let q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        q.len() as f32 / (self.max_queued / MAX_QUEUED_S) as f32
    }

    /// Queue 48 kHz mono audio for playback.
    pub fn play(&mut self, audio: &[f32]) {
        let samples = match &mut self.resampler {
            Some(r) => {
                self.scratch.clear();
                r.process(audio, &mut self.scratch);
                &self.scratch[..]
            }
            None => audio,
        };
        let mut q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        q.extend(samples.iter().map(|s| (s * self.volume).clamp(-1.0, 1.0)));
        let excess = q.len().saturating_sub(self.max_queued);
        q.drain(..excess);
    }
}

fn build<T>(dev: &cpal::Device, config: &StreamConfig, queue: Queue) -> Result<Stream>
where
    T: SizedSample + FromSample<f32> + Send + 'static,
{
    let channels = config.channels as usize;
    let stream = dev.build_output_stream::<T, _, _>(
        *config,
        move |data: &mut [T], _| {
            let mut q = queue.lock().unwrap_or_else(|e| e.into_inner());
            for frame in data.chunks_mut(channels) {
                frame.fill(T::from_sample(q.pop_front().unwrap_or(0.0)));
            }
        },
        |e| eprintln!("\naudio output error: {e}"),
        None,
    )?;
    Ok(stream)
}

/// Names of the available audio outputs, the default one first.
pub fn list_outputs() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let default = host.default_output_device().map(|d| d.to_string());
    let mut names: Vec<String> = Vec::new();
    for name in host.output_devices()?.map(|d| d.to_string()) {
        // ALSA lists one entry per plugin variant, often with identical names.
        if !names.contains(&name) {
            names.push(name);
        }
    }
    if let Some(d) = &default {
        names.retain(|n| n != d);
        names.insert(0, format!("{d} (default)"));
    }
    Ok(names)
}
