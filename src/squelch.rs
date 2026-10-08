//! Squelch for the interactive receiver. By default it opens and closes exactly like the
//! `listen` gate (see `gate.rs`): open when the channel is `--squelch-margin` dB over a
//! noise floor learned only from carrier-free frames (until the first such frame, carrier
//! prominence alone opens it), stay open while within the hysteresis, then hang. The
//! margin is adjustable from the page, or `None` to pass everything.

use crate::gate::{
    DEFAULT_CARRIER_PROMINENCE, DEFAULT_HANG_S, FLOOR_FALL, FLOOR_RISE_IDLE, FLOOR_RISE_OPEN, HYSTERESIS_DB,
    WARMUP_FRAMES,
};

/// Frames averaged for the displayed level (30 ms).
const LEVEL_FRAMES: usize = 3;
const HANG_FRAMES: u32 = (DEFAULT_HANG_S * 100.0) as u32;

pub struct Squelch {
    /// dB over the floor needed to open; `None` keeps it always open.
    pub threshold: Option<f32>,
    floor: Option<f32>,
    recent: [f32; LEVEL_FRAMES],
    frames: u64,
    open: bool,
    /// Frames since the carrier dropped while open.
    hang: u32,
}

impl Squelch {
    pub fn new(threshold: Option<f32>) -> Self {
        Self { threshold, floor: None, recent: [0.0; LEVEL_FRAMES], frames: 0, open: false, hang: 0 }
    }

    /// Learn the floor again (after a gain or span change).
    pub fn relearn(&mut self) {
        self.floor = None;
    }

    /// Move the learned floor, e.g. by the change in noise power when the bandwidth changes.
    pub fn shift_floor(&mut self, db: f32) {
        if let Some(f) = &mut self.floor {
            *f += db;
        }
    }

    /// Feed one frame's channel power and carrier prominence (pass 0 in modes without a
    /// carrier); returns whether audio should pass.
    pub fn push(&mut self, power_db: f32, carrier_db: f32) -> bool {
        self.recent[self.frames as usize % LEVEL_FRAMES] = power_db;
        self.frames += 1;
        if self.frames <= WARMUP_FRAMES {
            // Filters still settling.
            return self.threshold.is_none();
        }

        let prominent = carrier_db >= DEFAULT_CARRIER_PROMINENCE;
        let margin = self.threshold.unwrap_or(crate::gate::DEFAULT_SQUELCH_MARGIN);
        let (opens, carrier) = match self.floor {
            Some(floor) => (power_db > floor + margin, power_db > floor + margin - HYSTERESIS_DB),
            None => (prominent, prominent),
        };
        if self.open {
            if carrier {
                self.hang = 0;
            } else {
                self.hang += 1;
                if self.hang >= HANG_FRAMES {
                    self.open = false;
                }
            }
        } else if opens {
            self.open = true;
            self.hang = 0;
        }
        if !prominent {
            let rise = if self.open { FLOOR_RISE_OPEN } else { FLOOR_RISE_IDLE };
            match &mut self.floor {
                Some(f) => *f += (power_db - *f) * if power_db < *f { FLOOR_FALL } else { rise },
                None => self.floor = Some(power_db),
            }
        }
        self.threshold.is_none() || self.open
    }

    /// Smoothed channel level, dBFS.
    pub fn level_db(&self) -> f32 {
        let n = (self.frames as usize).clamp(1, LEVEL_FRAMES);
        let mean = self.recent[..n].iter().map(|db| 10f32.powf(db / 10.0)).sum::<f32>() / n as f32;
        10.0 * mean.max(1e-12).log10()
    }

    pub fn floor_db(&self) -> Option<f32> {
        self.floor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::DEFAULT_SQUELCH_MARGIN;

    fn feed(sq: &mut Squelch, db: f32, carrier: f32, n: usize) -> Vec<bool> {
        (0..n).map(|_| sq.push(db, carrier)).collect()
    }

    #[test]
    fn opens_at_the_margin_and_closes_after_hang_like_the_gate() {
        let mut sq = Squelch::new(Some(DEFAULT_SQUELCH_MARGIN));
        assert!(feed(&mut sq, -60.0, 0.0, 300).iter().all(|o| !o));
        assert!((sq.floor_db().unwrap() + 60.0).abs() < 0.1);
        // 7 dB is below the 8 dB margin: stays closed.
        assert!(feed(&mut sq, -53.0, 0.0, 50).iter().all(|o| !o));
        // 10 dB opens on the first frame.
        assert!(sq.push(-50.0, 0.0));
        // Within the 3 dB hysteresis: stays open indefinitely.
        assert!(feed(&mut sq, -54.0, 0.0, 400).iter().all(|&o| o));
        // Back to the noise: closes after the 1.5 s hang.
        let closed = feed(&mut sq, -60.0, 0.0, 300).iter().position(|o| !o).expect("closes");
        assert_eq!(closed, HANG_FRAMES as usize - 1);
    }

    #[test]
    fn carrier_on_at_tune_in_opens_and_is_not_taken_for_the_floor() {
        let mut sq = Squelch::new(Some(DEFAULT_SQUELCH_MARGIN));
        assert!(feed(&mut sq, -40.0, 30.0, 300)[10..].iter().all(|&o| o));
        assert_eq!(sq.floor_db(), None);
        // It drops: the floor is learned from the noise and the squelch closes after hang.
        assert!(feed(&mut sq, -70.0, 0.0, 300).last() == Some(&false));
        assert!((sq.floor_db().unwrap() + 70.0).abs() < 1.0);
        // A long carrier later does not drag the floor up.
        assert!(feed(&mut sq, -40.0, 30.0, 3000).iter().all(|&o| o));
        assert!((sq.floor_db().unwrap() + 70.0).abs() < 1.0, "floor {:?}", sq.floor_db());
    }

    #[test]
    fn off_passes_everything_but_still_learns_the_floor() {
        let mut sq = Squelch::new(None);
        assert!(feed(&mut sq, -62.0, 0.0, 200).iter().all(|&o| o));
        assert!(sq.push(-20.0, 0.0));
        assert!((sq.floor_db().unwrap() + 62.0).abs() < 0.5);
    }
}
