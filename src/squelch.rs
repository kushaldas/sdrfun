//! Manual squelch for the interactive receiver: opens when the channel is a set number of
//! dB above a learned noise floor, with hysteresis and hang time. Like the gate, the floor
//! is never raised by frames with a prominent carrier (AM/NFM), so a station that is
//! already on when you tune in is not taken for the noise.

/// Frames averaged for the level (30 ms).
const LEVEL_FRAMES: usize = 3;
/// Floor tracking per 10 ms frame: drops quickly, rises slowly (faster while closed).
const FLOOR_FALL: f32 = 0.3;
const FLOOR_RISE_CLOSED_DB: f32 = 0.01;
const FLOOR_RISE_OPEN_DB: f32 = 0.001;
/// Treated as a signal when no threshold is set.
const SIGNAL_DB: f32 = 6.0;
/// Close only this far below the opening threshold.
const HYSTERESIS_DB: f32 = 1.0;
/// Stay open this many frames after the level drops (0.3 s).
const HANG_FRAMES: u32 = 30;

pub struct Squelch {
    /// dB over the floor needed to open; `None` keeps it always open.
    pub threshold: Option<f32>,
    floor: Option<f32>,
    recent: [f32; LEVEL_FRAMES],
    filled: usize,
    open: bool,
    hang: u32,
}

impl Squelch {
    pub fn new(threshold: Option<f32>) -> Self {
        Self { threshold, floor: None, recent: [0.0; LEVEL_FRAMES], filled: 0, open: false, hang: 0 }
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

    /// Feed one frame's channel power and whether it has a prominent carrier (always
    /// `false` in modes without one); returns whether audio should pass.
    pub fn push(&mut self, power_db: f32, carrier: bool) -> bool {
        self.recent[self.filled % LEVEL_FRAMES] = power_db;
        self.filled += 1;
        let level = self.level_db();

        match &mut self.floor {
            // Until a carrier-free frame, a carrier alone opens.
            None if carrier => {}
            None => self.floor = Some(level),
            Some(floor) if level < *floor => *floor += (level - *floor) * FLOOR_FALL,
            Some(_) if carrier => {}
            Some(floor) => {
                // Rise slowly under a signal (so a long transmission is not taken as the
                // floor), and faster otherwise, also when the squelch is off.
                let signal = match self.threshold {
                    Some(_) => self.open,
                    None => level - *floor >= SIGNAL_DB,
                };
                let rise = if signal { FLOOR_RISE_OPEN_DB } else { FLOOR_RISE_CLOSED_DB };
                *floor += (level - *floor).min(rise);
            }
        }

        let Some(threshold) = self.threshold else {
            self.open = true;
            return true;
        };
        let above = match self.floor {
            Some(floor) => level - floor,
            None if carrier => threshold,
            None => 0.0,
        };
        if above >= threshold {
            self.open = true;
            self.hang = HANG_FRAMES;
        } else if self.open && above < threshold - HYSTERESIS_DB {
            if self.hang == 0 {
                self.open = false;
            } else {
                self.hang -= 1;
            }
        }
        self.open
    }

    /// Smoothed channel level, dBFS.
    pub fn level_db(&self) -> f32 {
        let n = self.filled.clamp(1, LEVEL_FRAMES);
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

    #[test]
    fn opens_above_threshold_and_closes_after_hang() {
        let mut sq = Squelch::new(Some(6.0));
        for _ in 0..200 {
            assert!(!sq.push(-60.0, false));
        }
        assert!((sq.floor_db().unwrap() + 60.0).abs() < 0.1);
        // 10 dB signal opens within the 3-frame level average.
        let opened = (0..5).position(|_| sq.push(-50.0, false)).expect("opens");
        assert!(opened <= 2);
        for _ in 0..100 {
            assert!(sq.push(-50.0, false), "stays open on the signal");
        }
        // Just under the threshold but within hysteresis: stays open.
        for _ in 0..100 {
            assert!(sq.push(-54.6, false));
        }
        // Back to the floor: closes after the hang time.
        let closed = (0..100).position(|_| !sq.push(-60.0, false)).expect("closes");
        assert!((HANG_FRAMES as usize..HANG_FRAMES as usize + 5).contains(&closed), "closed after {closed}");
    }

    #[test]
    fn off_threshold_always_passes() {
        let mut sq = Squelch::new(None);
        assert!(sq.push(-90.0, false));
        assert!(sq.push(-20.0, false));
    }

    #[test]
    fn floor_follows_a_noise_rise_even_with_squelch_off() {
        let mut sq = Squelch::new(None);
        for _ in 0..100 {
            sq.push(-62.0, false);
        }
        // Higher gain: the noise is now 4 dB up, below the signal level.
        for _ in 0..500 {
            sq.push(-58.0, false);
        }
        assert!((sq.floor_db().unwrap() + 58.0).abs() < 0.5, "floor {:?}", sq.floor_db());
    }

    #[test]
    fn carrier_on_at_tune_in_is_not_taken_for_the_floor() {
        let mut sq = Squelch::new(Some(6.0));
        // A station already transmitting: open on the carrier, no floor yet.
        for _ in 0..300 {
            assert!(sq.push(-40.0, true));
        }
        assert_eq!(sq.floor_db(), None);
        // It drops: the floor is learned from the noise and the squelch closes.
        let closed = (0..100).position(|_| !sq.push(-70.0, false)).expect("closes");
        assert!(closed < HANG_FRAMES as usize + 10);
        assert!((sq.floor_db().unwrap() + 70.0).abs() < 1.0);
        // A long carrier later does not drag the floor up.
        for _ in 0..3000 {
            assert!(sq.push(-40.0, true));
        }
        assert!((sq.floor_db().unwrap() + 70.0).abs() < 1.0, "floor {:?}", sq.floor_db());
    }
}
