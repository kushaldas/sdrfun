//! `sdrfun scan`: survey a list of frequencies. Frequencies that fit in one tuning
//! are received in parallel (one full receiver each); the tuner then hops to the next
//! group. Kept voice transmissions are saved like in `listen`, and a table of what
//! was heard is printed after every round.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, sync_channel};
use std::thread;

use anyhow::{Context, Result, anyhow, bail};
use chrono::Utc;
use clap::Args;

use crate::channel::{Channel, Mode};
use crate::clean::{CleanConfig, FRAME, SAMPLE_RATE};
use crate::gate::{GateConfig, GateEvent, Verdict};
use crate::listen::{Receiver, Session};
use crate::sdr::{Block, Device, Gain};

/// Default survey: Stockholm Arlanda (ESSA) frequencies from public listings
/// (OurAirports, SkyVector, RadioReference, Flight Plan Database), 2026-10.
const ARLANDA: &[(f64, &str)] = &[
    (118.500, "Arlanda TWR 01L/19R"),
    (119.000, "Arlanda ATIS arrival"),
    (120.150, "Arlanda APP (one listing)"),
    (121.625, "Arlanda ATIS departure"),
    (121.700, "Arlanda GND W"),
    (121.825, "Arlanda DEL"),
    (121.925, "Arlanda GND N"),
    (121.975, "Arlanda GND S"),
    (123.750, "Stockholm APP/CON"),
    (125.125, "Arlanda TWR 01R/19L"),
    (126.650, "Arlanda DEP (unconfirmed)"),
    (128.725, "Arlanda TWR 08/26"),
];

/// Part of the captured bandwidth used for channels (the RTL's edges roll off).
const USABLE_FRACTION: f64 = 0.75;
/// Keep channels at least this far from the tuner centre (DC spike).
const DC_GUARD_HZ: f64 = 20_000.0;

#[derive(Args, Debug)]
pub struct ScanArgs {
    /// Frequencies in MHz, optionally labelled (118.5=Tower)
    /// [default: Stockholm Arlanda tower, ground, ATIS and approach]
    pub freqs: Vec<String>,
    /// Seconds to listen to each group of frequencies per round
    #[arg(long, default_value_t = 30.0)]
    pub dwell: f64,
    /// Rounds over all groups (0 = until Ctrl-C)
    #[arg(long, default_value_t = 0)]
    pub rounds: u32,
    /// RTL-SDR device index
    #[arg(long, default_value_t = 0)]
    pub device: u32,
    /// RF tuner gain in dB, or "auto"
    #[arg(long, default_value_t = Gain::Manual(32.8))]
    pub gain: Gain,
    /// Frequency correction, ppm
    #[arg(long, default_value_t = 0, allow_negative_numbers = true)]
    pub ppm: i32,
    /// IQ sample rate, a multiple of 48000
    #[arg(long, default_value_t = 2_400_000)]
    pub sample_rate: u32,
    /// Demodulation: am (airband) or fm (amateur, PMR)
    #[arg(long, value_enum, default_value_t = Mode::Am)]
    pub mode: Mode,
    /// Channel filter width, Hz [default: 10000 for AM, 12500 for FM]
    #[arg(long)]
    pub bandwidth: Option<f32>,
    /// Directory for recordings
    #[arg(long, default_value = "recordings")]
    pub out_dir: PathBuf,
    /// Also save the un-cleaned demodulated audio next to each recording
    #[arg(long, default_value_t = false)]
    pub save_raw: bool,
    #[command(flatten)]
    pub gate: GateConfig,
    #[command(flatten)]
    pub clean: CleanConfig,
}

struct Target {
    mhz: f64,
    label: String,
    stats: Stats,
}

#[derive(Default)]
struct Stats {
    frames: u64,
    carrier_frames: u64,
    transmissions: u32,
    kept: u32,
    voice_s: f32,
    best_snr: Option<f32>,
}

#[derive(Debug, PartialEq)]
struct Group {
    center_hz: u32,
    members: Vec<usize>,
}

fn parse_targets(args: &[String]) -> Result<Vec<Target>> {
    if args.is_empty() {
        return Ok(ARLANDA
            .iter()
            .map(|&(mhz, label)| Target { mhz, label: label.into(), stats: Stats::default() })
            .collect());
    }
    args.iter()
        .map(|a| {
            let (f, label) = a.split_once('=').unwrap_or((a, ""));
            let mhz: f64 = f.trim().parse().with_context(|| format!("bad frequency {f:?}"))?;
            if !(24.0..=1766.0).contains(&mhz) {
                bail!("{mhz} MHz is outside the RTL-SDR range");
            }
            Ok(Target { mhz, label: label.trim().into(), stats: Stats::default() })
        })
        .collect()
}

/// Group frequencies (sorted by the caller's indices) so each group fits in one tuning.
fn plan_groups(mhz: &[f64], sample_rate: u32) -> Vec<Group> {
    let span = sample_rate as f64 * USABLE_FRACTION;
    let mut order: Vec<usize> = (0..mhz.len()).collect();
    order.sort_by(|&a, &b| mhz[a].total_cmp(&mhz[b]));

    let mut groups: Vec<Group> = Vec::new();
    for i in order {
        let hz = mhz[i] * 1e6;
        match groups.last_mut() {
            Some(g) if hz - mhz[g.members[0]] * 1e6 <= span => g.members.push(i),
            _ => groups.push(Group { center_hz: 0, members: vec![i] }),
        }
    }
    for g in &mut groups {
        let hz: Vec<f64> = g.members.iter().map(|&i| mhz[i] * 1e6).collect();
        g.center_hz = place_center(&hz, sample_rate).round() as u32;
    }
    groups
}

/// Tuner centre for a group: the position nearest the midpoint that keeps every channel
/// `DC_GUARD_HZ` from the DC spike, or, when channels are too dense for that, the one
/// that keeps the nearest channel farthest away (e.g. halfway between two channels).
fn place_center(hz: &[f64], sample_rate: u32) -> f64 {
    let (lo, hi) = (hz[0], hz[hz.len() - 1]);
    let mid = (lo + hi) / 2.0;
    // Channels must stay inside the captured band with room for their filter.
    let reach = sample_rate as f64 * 0.45;
    let slack = (reach - (hi - lo) / 2.0).max(0.0);
    let clearance = |c: f64| hz.iter().map(|f| (f - c).abs()).fold(f64::MAX, f64::min);

    let step = 250.0;
    let steps = (slack / step) as i64;
    // Search outwards from the midpoint: 0, +1, -1, +2, -2, ...
    let candidates = (0..=2 * steps).map(|k| mid + step * if k % 2 == 1 { (k + 1) / 2 } else { -(k / 2) } as f64);
    let mut best = (mid, clearance(mid));
    for c in candidates {
        let d = clearance(c);
        if d >= DC_GUARD_HZ {
            return c;
        }
        if d > best.1 + 1.0 {
            best = (c, d);
        }
    }
    best.0
}

pub fn run(args: ScanArgs) -> Result<()> {
    let mut targets = parse_targets(&args.freqs)?;
    let bandwidth = args.bandwidth.unwrap_or(args.mode.default_bandwidth());
    let groups = plan_groups(&targets.iter().map(|t| t.mhz).collect::<Vec<_>>(), args.sample_rate);

    let mut dev = Device::open(args.device)?;
    let rate = dev.set_sample_rate(args.sample_rate)?;
    dev.set_ppm(args.ppm)?;
    let gain = dev.set_gain(args.gain)?;
    dev.set_center_freq(groups[0].center_hz)?;

    eprintln!(
        "scanning {} frequencies in {} group(s), {:.0} s each ({:?}, gain {}, bandwidth {:.1} kHz)",
        targets.len(),
        groups.len(),
        args.dwell,
        args.mode,
        gain.map_or("auto".into(), |g| format!("{g:.1} dB")),
        bandwidth / 1e3,
    );
    for (n, g) in groups.iter().enumerate() {
        let list: Vec<String> = g.members.iter().map(|&i| format!("{:.3}", targets[i].mhz)).collect();
        eprintln!("  group {}: tuner {:.3} MHz → {}", n + 1, g.center_hz as f64 / 1e6, list.join(" "));
    }

    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        ctrlc::set_handler(move || stop.store(true, Ordering::Relaxed))
            .context("installing Ctrl-C handler")?;
    }
    let (tx, rx) = sync_channel::<Block>(64);
    let (retune_tx, retune_rx) = channel::<u32>();
    let reader = {
        let stop = stop.clone();
        thread::spawn(move || dev.stream(tx, stop, Some(retune_rx)))
    };

    let mut session = Session::new(args.out_dir.clone(), args.save_raw)?;
    let dwell_frames = (args.dwell.max(1.0) * SAMPLE_RATE as f64 / FRAME as f64) as u64;
    let mut tuned = groups[0].center_hz;
    let mut round = 0;

    'rounds: while args.rounds == 0 || round < args.rounds {
        round += 1;
        for (n, group) in groups.iter().enumerate() {
            if group.center_hz != tuned {
                retune_tx.send(group.center_hz).map_err(|_| anyhow!("SDR reader stopped"))?;
                tuned = group.center_hz;
            }
            let mut receivers = group
                .members
                .iter()
                .map(|&i| {
                    let offset = targets[i].mhz * 1e6 - tuned as f64;
                    Ok(Receiver::new(Channel::new(rate, offset, bandwidth, args.mode)?, &args.clean, args.gate.clone()))
                })
                .collect::<Result<Vec<_>>>()?;

            let mut start = None;
            let mut frames = 0u64;
            for block in rx.iter() {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                // Skip blocks captured before the retune took effect.
                if (block.center_hz as i64 - tuned as i64).abs() > 1_000 {
                    continue;
                }
                let start = *start.get_or_insert_with(Utc::now);
                let results: Vec<_> = thread::scope(|s| {
                    let handles: Vec<_> =
                        receivers.iter_mut().map(|r| s.spawn(|| r.process(&block.data))).collect();
                    handles.into_iter().map(|h| h.join().expect("receiver thread panicked")).collect()
                });
                frames += results.first().map_or(0, |r| r.len()) as u64;
                for (&i, processed) in group.members.iter().zip(results) {
                    let t = &mut targets[i];
                    for p in processed {
                        t.stats.frames += 1;
                        t.stats.carrier_frames += u64::from(p.carrier_db >= args.gate.carrier_prominence);
                        if let Some(GateEvent::Closed(tx)) = p.event {
                            record(&mut session, t, start, tx)?;
                        }
                    }
                }
                eprint!(
                    "\r\x1b[Kround {round}  group {}/{}  {:3.0} s left  saved {} dropped {}",
                    n + 1,
                    groups.len(),
                    dwell_frames.saturating_sub(frames) as f64 * FRAME as f64 / SAMPLE_RATE as f64,
                    session.kept,
                    session.dropped,
                );
                let _ = std::io::stderr().flush();
                if frames >= dwell_frames {
                    break;
                }
            }
            let start = start.unwrap_or_else(Utc::now);
            for (&i, r) in group.members.iter().zip(&mut receivers) {
                if let Some(tx) = r.gate.flush() {
                    record(&mut session, &mut targets[i], start, tx)?;
                }
            }
            if stop.load(Ordering::Relaxed) {
                break 'rounds;
            }
        }
        print_table(&format!("after round {round}"), &targets, args.gate.carrier_prominence);
    }
    let interrupted = stop.swap(true, Ordering::Relaxed);
    drop(rx);
    reader.join().map_err(|_| anyhow!("SDR reader thread panicked"))??;
    if interrupted {
        print_table(&format!("round {round} (interrupted)"), &targets, args.gate.carrier_prominence);
    }
    Ok(())
}

fn record(
    session: &mut Session,
    t: &mut Target,
    start: chrono::DateTime<Utc>,
    tx: crate::gate::Transmission,
) -> Result<()> {
    let seconds = tx.seconds();
    let snr = tx.peak_db - tx.floor_db;
    t.stats.transmissions += 1;
    if snr.is_finite() {
        t.stats.best_snr = Some(t.stats.best_snr.map_or(snr, |b| b.max(snr)));
    }
    if session.finish(t.mhz, start, tx)?.0 == Verdict::Kept {
        t.stats.kept += 1;
        t.stats.voice_s += seconds;
    }
    Ok(())
}

fn print_table(title: &str, targets: &[Target], prominence: f32) {
    eprintln!("\r\x1b[K");
    println!("{title}  (carrier = share of time with a carrier ≥ {prominence} dB above the channel edges)");
    println!("  {:>8}  {:<26} {:>7} {:>4} {:>6} {:>8} {:>9}", "MHz", "label", "carrier", "tx", "voice", "voice s", "best snr");
    let mut sorted: Vec<&Target> = targets.iter().collect();
    sorted.sort_by(|a, b| a.mhz.total_cmp(&b.mhz));
    for t in sorted {
        let s = &t.stats;
        let carrier = 100.0 * s.carrier_frames as f64 / s.frames.max(1) as f64;
        let snr = s.best_snr.map_or("—".to_string(), |v| format!("{v:.1} dB"));
        let heard = if s.kept > 0 { "  ← voice" } else { "" };
        println!(
            "  {:8.3}  {:<26} {carrier:6.1}% {:>4} {:>6} {:>8.1} {snr:>9}{heard}",
            t.mhz, t.label, s.transmissions, s.kept, s.voice_s
        );
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arlanda_list_fits_in_five_tunings_away_from_dc() {
        let mhz: Vec<f64> = ARLANDA.iter().map(|t| t.0).collect();
        let groups = plan_groups(&mhz, 2_400_000);
        assert_eq!(groups.len(), 5);
        let mut seen: Vec<usize> = groups.iter().flat_map(|g| g.members.clone()).collect();
        seen.sort();
        assert_eq!(seen, (0..mhz.len()).collect::<Vec<_>>());
        for g in &groups {
            for &i in &g.members {
                let off = (mhz[i] * 1e6 - g.center_hz as f64).abs();
                assert!(off >= DC_GUARD_HZ, "{} too close to DC", mhz[i]);
                assert!(off <= 2_400_000.0 * 0.4, "{} near the band edge", mhz[i]);
            }
        }
    }

    /// Tuner centre and distance of the nearest channel from it, for a 12.5 kHz grid.
    fn grid_placement(count: usize) -> (f64, f64, Vec<f64>) {
        let mhz: Vec<f64> = (0..count).map(|k| 145.000 + 0.0125 * k as f64).collect();
        let groups = plan_groups(&mhz, 2_400_000);
        assert_eq!(groups.len(), 1);
        let c = groups[0].center_hz as f64;
        let nearest = mhz.iter().map(|m| (m * 1e6 - c).abs()).fold(f64::MAX, f64::min);
        for m in &mhz {
            assert!((m * 1e6 - c).abs() < 2_400_000.0 * 0.45, "{m} outside the captured band");
        }
        (c, nearest, mhz)
    }

    #[test]
    fn dense_grid_moves_dc_clear_of_all_channels_when_there_is_room() {
        // 32 channels span 387.5 kHz: there is room to put DC beside the grid.
        let (_, nearest, _) = grid_placement(32);
        assert!(nearest >= DC_GUARD_HZ, "nearest channel {nearest} Hz from DC");
    }

    #[test]
    fn full_width_dense_grid_puts_dc_between_two_channels() {
        // 145 channels span the full 1.8 MHz: no clear spot, so split a channel gap.
        let (_, nearest, _) = grid_placement(145);
        assert!((nearest - 6250.0).abs() < 300.0, "nearest channel {nearest} Hz from DC");
    }

    #[test]
    fn parses_labels_and_rejects_nonsense() {
        let t = parse_targets(&["118.5=Tower".into(), "121.5".into()]).unwrap();
        assert_eq!((t[0].mhz, t[0].label.as_str()), (118.5, "Tower"));
        assert_eq!(t[1].label, "");
        assert!(parse_targets(&["abc".into()]).is_err());
        assert!(parse_targets(&["5000".into()]).is_err());
    }
}
