mod audio_out;
mod channel;
mod clean;
mod dsp;
mod gate;
mod listen;
mod recorder;
mod sdr;
mod wavio;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use clean::{CleanConfig, SAMPLE_RATE};

#[derive(Parser)]
#[command(version, about = "Listen to AM airband voice on an RTL-SDR, clean it up, play and record it")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Receive a frequency, clean the audio and record it
    Listen(listen::ListenArgs),
    /// Run the voice cleanup chain over a WAV file
    Clean {
        /// Input WAV (any rate, mono or stereo)
        input: PathBuf,
        /// Output WAV [default: <input>_clean.wav]
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output sample rate in Hz
        #[arg(long, default_value_t = 16_000)]
        out_rate: u32,
        /// Print the voice probability timeline (0.25 s steps), for tuning the gate
        #[arg(long, default_value_t = false)]
        vad_report: bool,
        /// Also play the cleaned audio on the speaker
        #[arg(long, default_value_t = false)]
        play: bool,
        /// Audio output for --play: "default" or part of a device name
        #[arg(long, default_value = "default")]
        audio_device: String,
        #[command(flatten)]
        cfg: CleanConfig,
    },
    /// List RTL-SDR devices and audio outputs
    Devices,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Listen(args) => listen::run(args),
        Command::Clean {
            input,
            output,
            out_rate,
            vad_report,
            play,
            audio_device,
            cfg,
        } => {
            let cleaned = run_clean(input, output, out_rate, vad_report, cfg)?;
            if play {
                play_all(&cleaned, &audio_device)?;
            }
            Ok(())
        }
        Command::Devices => {
            let devices = sdr::list_devices();
            println!("RTL-SDR devices (--device):");
            if devices.is_empty() {
                println!("  none found");
            }
            for d in devices {
                println!(
                    "  {}: {} — {} {} (serial {})",
                    d.index, d.name, d.manufacturer, d.product, d.serial
                );
            }
            println!("audio outputs (--audio-device):");
            match audio_out::list_outputs() {
                Ok(names) => names.iter().for_each(|n| println!("  {n}")),
                Err(e) => println!("  unavailable: {e}"),
            }
            Ok(())
        }
    }
}

fn run_clean(
    input: PathBuf,
    output: Option<PathBuf>,
    out_rate: u32,
    vad_report: bool,
    cfg: CleanConfig,
) -> Result<Vec<f32>> {
    let output = output.unwrap_or_else(|| {
        let stem = input.file_stem().unwrap_or_default().to_string_lossy();
        input.with_file_name(format!("{stem}_clean.wav"))
    });

    let (samples, rate) = wavio::read_mono(&input)?;
    let audio = dsp::resample::resample_all(&samples, rate, SAMPLE_RATE);
    let (cleaned, stats) = clean::clean_all(&audio, &cfg);
    let out = dsp::resample::resample_all(&cleaned, SAMPLE_RATE, out_rate);
    wavio::write_mono_i16(&output, &out, out_rate)?;

    let (in_peak, in_rms) = wavio::levels_dbfs(&audio);
    let (out_peak, out_rms) = wavio::levels_dbfs(&cleaned);
    println!("input : {} ({:.2} s @ {rate} Hz)", input.display(), samples.len() as f32 / rate as f32);
    println!("output: {} ({out_rate} Hz, denoiser {:?})", output.display(), cfg.denoiser);
    println!("level : in  peak {in_peak:6.1} dBFS, rms {in_rms:6.1} dBFS");
    println!("        out peak {out_peak:6.1} dBFS, rms {out_rms:6.1} dBFS");
    if let Some(vad) = stats.mean_vad() {
        println!("voice : mean RNNoise voice probability {vad:.2}");
    }
    if vad_report {
        println!("voice probability per 0.25 s:");
        for (i, chunk) in stats.vad.chunks(25).enumerate() {
            let mean = chunk.iter().sum::<f32>() / chunk.len() as f32;
            let max = chunk.iter().copied().fold(0.0, f32::max);
            println!(
                "  {:6.2}s  mean {mean:.2}  max {max:.2}  {}",
                i as f32 * 0.25,
                "#".repeat((mean * 40.0).round() as usize)
            );
        }
    }
    Ok(cleaned)
}

/// Play 48 kHz audio on the speaker and wait until it has finished.
fn play_all(audio: &[f32], device: &str) -> Result<()> {
    let mut out = audio_out::AudioOut::open(device, 1.0)?;
    println!("playing on {}", out.description);
    // Feed in 1 s pieces so the bounded playback queue never drops audio.
    for chunk in audio.chunks(SAMPLE_RATE as usize) {
        while out.queued_seconds() > 1.0 {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        out.play(chunk);
    }
    while out.queued_seconds() > 0.0 {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    std::thread::sleep(std::time::Duration::from_millis(200));
    Ok(())
}
