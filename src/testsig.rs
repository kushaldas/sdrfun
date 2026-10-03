//! Synthetic speech for tests: a glottal pulse train with a moving pitch, shaped by
//! vowel formant resonators and grouped into syllables and phrases.

use std::f64::consts::PI;

/// (F1, F2, F3) in Hz for a few vowels.
const VOWELS: [[f64; 3]; 5] = [
    [730.0, 1090.0, 2440.0], // a
    [270.0, 2290.0, 3010.0], // i
    [530.0, 1840.0, 2480.0], // e
    [570.0, 840.0, 2410.0],  // o
    [300.0, 870.0, 2240.0],  // u
];
const FORMANT_BW: [f64; 3] = [90.0, 110.0, 160.0];
const SYLLABLE_S: f64 = 0.22;
const SYLLABLES_PER_PHRASE: usize = 6;
const PHRASE_GAP_S: f64 = 0.35;

/// Two-pole resonator normalised to unity gain at its centre frequency.
struct Resonator {
    a1: f64,
    a2: f64,
    gain: f64,
    y1: f64,
    y2: f64,
}

impl Resonator {
    fn new() -> Self {
        Self { a1: 0.0, a2: 0.0, gain: 0.0, y1: 0.0, y2: 0.0 }
    }

    fn tune(&mut self, f: f64, bw: f64, fs: f64) {
        let r = (-PI * bw / fs).exp();
        let w = 2.0 * PI * f / fs;
        self.a1 = 2.0 * r * w.cos();
        self.a2 = -r * r;
        self.gain = (1.0 - r) * (1.0 - 2.0 * r * (2.0 * w).cos() + r * r).sqrt();
    }

    fn process(&mut self, x: f64) -> f64 {
        let y = self.gain * x + self.a1 * self.y1 + self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// `seconds` of synthetic speech at `fs`, peak-normalised to 0.5.
pub fn speech(fs: u32, seconds: f64) -> Vec<f32> {
    let fs_f = fs as f64;
    let n = (seconds * fs_f) as usize;
    let phrase_s = SYLLABLE_S * SYLLABLES_PER_PHRASE as f64;
    let mut formants = [Resonator::new(), Resonator::new(), Resonator::new()];
    let mut phase = 0.0;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / fs_f;
        let in_phrase = t % (phrase_s + PHRASE_GAP_S);
        let phrase = (t / (phrase_s + PHRASE_GAP_S)) as usize;
        let syllable = (in_phrase / SYLLABLE_S) as usize;

        // Pitch falls across each phrase, with a little vibrato.
        let pitch = 165.0 - 45.0 * (in_phrase / phrase_s).min(1.0) + 4.0 * (2.0 * PI * 5.5 * t).sin();
        phase += pitch / fs_f;
        // Glottal pulse: a decaying sawtooth with a 1/f-ish spectrum.
        let glottal = 1.0 - 2.0 * phase.fract();

        let vowel = VOWELS[(phrase * 3 + syllable * 2) % VOWELS.len()];
        for ((res, &f), &bw) in formants.iter_mut().zip(&vowel).zip(&FORMANT_BW) {
            res.tune(f, bw, fs_f);
        }
        let voiced: f64 = formants.iter_mut().map(|r| r.process(glottal)).sum();

        let envelope = if syllable < SYLLABLES_PER_PHRASE {
            (PI * (in_phrase % SYLLABLE_S) / SYLLABLE_S).sin().powf(0.7)
        } else {
            0.0
        };
        out.push((voiced * envelope) as f32);
    }
    let peak = out.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-9);
    out.iter_mut().for_each(|x| *x *= 0.5 / peak);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clean::{CleanConfig, SAMPLE_RATE, clean_all};
    use clap::Parser;

    #[derive(Parser)]
    struct Wrapper {
        #[command(flatten)]
        cfg: CleanConfig,
    }

    #[test]
    fn rnnoise_hears_synthetic_speech_as_voice() {
        let cfg = Wrapper::parse_from(["x"]).cfg;
        let (_, stats) = clean_all(&speech(SAMPLE_RATE, 6.0), &cfg);
        let voiced = stats.vad.iter().filter(|&&v| v >= 0.5).count() as f32 / stats.vad.len() as f32;
        eprintln!("voiced fraction {voiced:.2}, mean {:.2}", stats.mean_vad().unwrap());
        assert!(voiced > 0.3, "voiced fraction {voiced}");
    }
}
