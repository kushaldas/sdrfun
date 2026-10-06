//! IMA ADPCM (4 bits per sample), as OpenWebRX uses to send audio and waterfall rows.
//! Each encoded chunk starts with the encoder state (predictor i16 LE, step index u8) so it
//! can be decoded without the chunks before it. Nibbles are packed low first.
//! `web.html` has the matching decoder.

const INDEX_TABLE: [i8; 16] = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8];

const STEP_TABLE: [i32; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408,
    449, 494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
    2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630,
    9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
    32767,
];

pub const HEADER: usize = 3;

#[derive(Default, Clone, Copy)]
struct State {
    predictor: i32,
    index: i32,
}

impl State {
    /// Apply one code; returns the new sample.
    fn step(&mut self, code: u8) -> i16 {
        let step = STEP_TABLE[self.index as usize];
        let mut diff = step >> 3;
        if code & 4 != 0 {
            diff += step;
        }
        if code & 2 != 0 {
            diff += step >> 1;
        }
        if code & 1 != 0 {
            diff += step >> 2;
        }
        self.predictor += if code & 8 != 0 { -diff } else { diff };
        self.predictor = self.predictor.clamp(i16::MIN as i32, i16::MAX as i32);
        self.index = (self.index + INDEX_TABLE[code as usize] as i32).clamp(0, 88);
        self.predictor as i16
    }
}

#[derive(Default)]
pub struct Encoder {
    state: State,
}

impl Encoder {
    pub fn encode(&mut self, samples: &[i16]) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER + samples.len().div_ceil(2));
        out.extend_from_slice(&(self.state.predictor as i16).to_le_bytes());
        out.push(self.state.index as u8);
        for pair in samples.chunks(2) {
            let lo = self.code(pair[0]);
            let hi = pair.get(1).map_or(0, |&s| self.code(s));
            out.push(lo | (hi << 4));
        }
        out
    }

    fn code(&mut self, sample: i16) -> u8 {
        let step = STEP_TABLE[self.state.index as usize];
        let mut diff = sample as i32 - self.state.predictor;
        let mut code = 0u8;
        if diff < 0 {
            code = 8;
            diff = -diff;
        }
        let mut s = step;
        for bit in [4u8, 2, 1] {
            if diff >= s {
                code |= bit;
                diff -= s;
            }
            s >>= 1;
        }
        // Track exactly what the decoder will reconstruct.
        self.state.step(code);
        code
    }
}

/// Decode one chunk made by `Encoder::encode` (`samples` of them).
#[cfg(test)]
pub fn decode(chunk: &[u8], samples: usize) -> Vec<i16> {
    let mut state = State {
        predictor: i16::from_le_bytes([chunk[0], chunk[1]]) as i32,
        index: chunk[2] as i32,
    };
    chunk[HEADER..]
        .iter()
        .flat_map(|&b| [b & 15, b >> 4])
        .take(samples)
        .map(|code| state.step(code))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(n: usize, start: usize) -> Vec<i16> {
        (start..start + n)
            .map(|i| (8000.0 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24_000.0).sin()) as i16)
            .collect()
    }

    #[test]
    fn round_trip_tracks_the_signal() {
        let mut enc = Encoder::default();
        let input = sine(2400, 0);
        let chunk = enc.encode(&input);
        assert_eq!(chunk.len(), HEADER + 1200);
        let out = decode(&chunk, input.len());
        // Skip the start-up while the step size adapts.
        let err = input[100..].iter().zip(&out[100..]).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap();
        assert!(err < 800, "max error {err}");
    }

    #[test]
    fn a_later_chunk_decodes_on_its_own() {
        let mut enc = Encoder::default();
        enc.encode(&sine(2400, 0));
        let input = sine(2400, 2400);
        let chunk = enc.encode(&input);
        let out = decode(&chunk, input.len());
        let err = input.iter().zip(&out).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap();
        assert!(err < 800, "max error {err}");
    }
}

