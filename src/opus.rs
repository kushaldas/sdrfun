//! Opus encoding for the streamed audio, with forward error correction.
//!
//! Audio is 48 kHz mono Opus in 20 ms packets. Every message carries the
//! current packet plus a copy of the previous one (the FEC redundancy): a
//! client that misses one message decodes the duplicate from the next
//! message to fill the gap, so a single dropped message costs no audio.
//! The encoder also enables Opus in-band FEC, which decoders that understand
//! it can use for the same purpose at lower cost.
//!
//! Wire format, all integers little-endian:
//! - `[0]` message kind (see `serve.rs`)
//! - `[1..5]` sequence number, starting at 0 per stream
//! - `[5..7]` length of the current packet
//! - `[7..9]` length of the previous packet (0 for the first)
//! - `[9..]` current packet, then the previous packet

use anyhow::Result;

/// Sample rate every Opus stream is encoded at.
pub const OPUS_RATE: u32 = 48_000;
/// Samples per packet: 20 ms at 48 kHz.
pub const PACKET_SAMPLES: usize = 960;
/// Size of the fixed message header.
pub const HEADER: usize = 9;

/// One encoded audio stream: queue 48 kHz audio in, complete wire messages
/// come out.
pub struct Stream {
    kind: u8,
    enc: opus::Encoder,
    kbps: i32,
    voice: bool,
    seq: u32,
    /// Audio not yet filling a whole packet.
    pending: Vec<f32>,
    frame: Vec<f32>,
    buf: Vec<u8>,
    /// Last packet, duplicated into the next message as the FEC copy.
    prev: Vec<u8>,
}

impl Stream {
    /// `kbps` target bitrate; `voice` picks the VoIP profile (with the voice
    /// signal hint) over the music one.
    pub fn new(kind: u8, kbps: i32, voice: bool) -> Result<Self> {
        Ok(Self {
            kind,
            enc: Self::encoder(kbps, voice)?,
            kbps,
            voice,
            seq: 0,
            pending: Vec::new(),
            frame: Vec::with_capacity(PACKET_SAMPLES),
            buf: vec![0; 8 * 1024],
            prev: Vec::new(),
        })
    }

    fn encoder(kbps: i32, voice: bool) -> Result<opus::Encoder> {
        let mut enc = opus::Encoder::new(
            OPUS_RATE,
            opus::Channels::Mono,
            if voice { opus::Application::Voip } else { opus::Application::Audio },
        )?;
        enc.set_bitrate(opus::Bitrate::Bits(kbps * 1000))?;
        enc.set_inband_fec(true)?;
        if voice {
            enc.set_signal(opus::Signal::Voice)?;
        }
        Ok(enc)
    }

    /// Change the profile (e.g. voice mode to broadcast FM); the encoder is
    /// rebuilt only when something actually differs.
    pub fn set_profile(&mut self, kbps: i32, voice: bool) -> Result<()> {
        if kbps == self.kbps && voice == self.voice {
            return Ok(());
        }
        self.enc = Self::encoder(kbps, voice)?;
        self.kbps = kbps;
        self.voice = voice;
        Ok(())
    }

    /// Queue 48 kHz mono audio; every complete 20 ms packet is appended to
    /// `out` as a wire message. Encoding errors skip the packet.
    pub fn push(&mut self, audio: &[f32], out: &mut Vec<Vec<u8>>) {
        self.pending.extend_from_slice(audio);
        let mut frame = std::mem::take(&mut self.frame);
        while self.pending.len() >= PACKET_SAMPLES {
            frame.clear();
            frame.extend(self.pending.drain(..PACKET_SAMPLES));
            self.encode(&frame, out);
        }
        self.frame = frame;
    }

    /// Send the buffered remainder as a short final packet, zero-padded to the
    /// next duration libopus accepts.
    pub fn flush(&mut self, out: &mut Vec<Vec<u8>>) {
        if self.pending.is_empty() {
            return;
        }
        let mut frame = std::mem::take(&mut self.frame);
        frame.clear();
        frame.append(&mut self.pending);
        const DURATIONS: [usize; 6] = [120, 240, 480, 960, 1920, 2880];
        let pad = DURATIONS.iter().copied().find(|&d| d >= frame.len()).unwrap_or(PACKET_SAMPLES);
        frame.resize(pad, 0.0);
        self.encode(&frame, out);
        self.frame = frame;
    }

    fn encode(&mut self, samples: &[f32], out: &mut Vec<Vec<u8>>) {
        let Ok(n) = self.enc.encode_float(samples, &mut self.buf) else {
            return;
        };
        let packet = self.buf[..n].to_vec();
        let mut msg = Vec::with_capacity(HEADER + packet.len() + self.prev.len());
        msg.push(self.kind);
        msg.extend_from_slice(&self.seq.to_le_bytes());
        msg.extend_from_slice(&(packet.len() as u16).to_le_bytes());
        msg.extend_from_slice(&(self.prev.len() as u16).to_le_bytes());
        msg.extend_from_slice(&packet);
        msg.extend_from_slice(&self.prev);
        self.prev = packet;
        self.seq += 1;
        out.push(msg);
    }
}

/// The packets of a message made by `Stream`: `(seq, current, previous)`.
/// Only tests (the clients decode in JavaScript) read messages back.
#[cfg(test)]
pub fn split(msg: &[u8]) -> Option<(u32, &[u8], &[u8])> {
    let seq = u32::from_le_bytes(msg[1..5].try_into().ok()?);
    let cur_len: usize = u16::from_le_bytes(msg[5..7].try_into().ok()?).into();
    let prev_len: usize = u16::from_le_bytes(msg[7..9].try_into().ok()?).into();
    let cur = msg.get(HEADER..HEADER + cur_len)?;
    let prev = msg.get(HEADER + cur_len..HEADER + cur_len + prev_len)?;
    Some((seq, cur, prev))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A decoder for tests that check what a client would hear.
    struct Decoder(opus::Decoder);

    impl Decoder {
        fn new() -> Self {
            Self(opus::Decoder::new(OPUS_RATE, opus::Channels::Mono).unwrap())
        }

        /// Decode one packet to 48 kHz float samples.
        fn decode(&mut self, packet: &[u8]) -> Vec<f32> {
            let mut out = vec![0.0f32; PACKET_SAMPLES * 2];
            let n = self.0.decode_float(packet, &mut out, false).unwrap();
            out.truncate(n);
            out
        }
    }

    fn tone(n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / OPUS_RATE as f32).sin())
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }

    #[test]
    fn packets_are_20ms_and_round_trip() {
        let mut stream = Stream::new(5, 32, true).unwrap();
        let mut out = Vec::new();
        stream.push(&tone(4800, 0.4), &mut out);
        assert_eq!(out.len(), 5, "5 x 20 ms in 100 ms");
        let mut dec = Decoder::new();
        let mut pcm = Vec::new();
        for (i, msg) in out.iter().enumerate() {
            assert_eq!(msg[0], 5);
            let (seq, cur, prev) = split(msg).unwrap();
            assert_eq!(seq, i as u32);
            assert_eq!(prev.is_empty(), i == 0, "every packet after the first carries an FEC copy");
            pcm.extend(dec.decode(cur));
        }
        assert_eq!(pcm.len(), 4800);
        // Lossy codec: check it reproduced the tone's level, not its exact samples.
        let want = 0.4 / 2.0f32.sqrt();
        let got = rms(&pcm[960..]);
        assert!((got / want).abs() > 0.6 && (got / want).abs() < 1.4, "rms {got}, want ~{want}");
    }

    #[test]
    fn a_dropped_message_is_recovered_from_the_next_one() {
        let mut stream = Stream::new(5, 32, true).unwrap();
        let mut out = Vec::new();
        stream.push(&tone(3 * PACKET_SAMPLES, 0.4), &mut out);
        assert_eq!(out.len(), 3);
        let mut dec = Decoder::new();
        let full: Vec<f32> = out
            .iter()
            .flat_map(|m| {
                let (_, cur, _) = split(m).unwrap();
                dec.decode(cur)
            })
            .collect();
        // The client misses message 1, then uses its FEC copy inside message 2.
        // A fresh decoder: decoding is stateful, as it would be after a real gap.
        let mut dec = Decoder::new();
        let (_, cur0, _) = split(&out[0]).unwrap();
        let (_, cur2, prev2) = split(&out[2]).unwrap();
        let (_, cur1, _) = split(&out[1]).unwrap();
        assert_eq!(prev2, cur1, "the FEC copy is the previous packet verbatim");
        let mut recovered = dec.decode(cur0);
        recovered.extend(dec.decode(prev2));
        recovered.extend(dec.decode(cur2));
        assert_eq!(recovered.len(), full.len());
        let err = recovered.iter().zip(&full).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-6, "recovered audio differs: {err}");
    }

    #[test]
    fn flush_sends_the_remainder_and_bitrate_can_change() {
        let mut stream = Stream::new(5, 32, true).unwrap();
        let mut out = Vec::new();
        stream.push(&tone(PACKET_SAMPLES + 160, 0.4), &mut out);
        assert_eq!(out.len(), 1);
        stream.flush(&mut out);
        assert_eq!(out.len(), 2, "the 160-sample remainder goes out on flush");
        let mut dec = Decoder::new();
        let (_, cur, _) = split(&out[1]).unwrap();
        let pcm = dec.decode(cur);
        assert_eq!(pcm.len(), 240, "padded to the next duration libopus accepts");
        assert!(rms(&pcm[..160]) > 0.01, "the short packet still carries the tone");
        stream.set_profile(64, false).unwrap();
        stream.set_profile(64, false).unwrap();
        stream.set_profile(32, true).unwrap();
    }
}
