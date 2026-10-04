//! `--serve`: a small web page plus a WebSocket that streams the cleaned audio
//! (what the speaker plays) and live status, for listening on a phone.
//!
//! WebSocket `/ws`: binary messages are mono 16-bit little-endian PCM at
//! `STREAM_RATE`; text messages are JSON events (`status`, `open`, `tx`). A new
//! client first receives every `tx` event of the run so far. Saved transmissions
//! carry a `url` (`/rec/<n>.wav`) that serves the recording, with byte ranges
//! as iOS Safari requires for audio.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, UdpSocket};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use tungstenite::Message;

use crate::clean::SAMPLE_RATE;
use crate::dsp::resample::Resampler;

/// Streamed sample rate. Older iOS Safari rejects Web Audio buffers below 22.05 kHz.
pub const STREAM_RATE: u32 = 24_000;
/// Audio is sent in pieces of this many samples (100 ms).
const CHUNK: usize = STREAM_RATE as usize / 10;
/// Messages queued per client before it is considered too slow and audio is dropped.
const CLIENT_QUEUE: usize = 64;
/// Transmission events replayed to a newly connected page.
const HISTORY: usize = 500;
const PAGE: &str = include_str!("serve.html");

#[derive(Default)]
struct Shared {
    clients: Vec<SyncSender<Message>>,
    /// `tx` events of this run, oldest first.
    history: VecDeque<String>,
    /// Recordings that may be served, indexed by the number in their URL.
    files: Vec<PathBuf>,
}

type SharedRef = Arc<Mutex<Shared>>;

fn lock(shared: &SharedRef) -> std::sync::MutexGuard<'_, Shared> {
    shared.lock().unwrap_or_else(|e| e.into_inner())
}

pub struct Streamer {
    shared: SharedRef,
    resampler: Resampler,
    scratch: Vec<f32>,
    pending: Vec<i16>,
}

impl Streamer {
    /// Start listening on `addr` (e.g. "0.0.0.0:8010"); returns the streamer and the
    /// URL to open on another device.
    pub fn start(addr: &str) -> Result<(Self, String)> {
        let listener = TcpListener::bind(addr).with_context(|| format!("binding {addr}"))?;
        let local = listener.local_addr()?;
        let shared = SharedRef::default();
        {
            let shared = shared.clone();
            thread::spawn(move || {
                for stream in listener.incoming().flatten() {
                    let shared = shared.clone();
                    thread::spawn(move || {
                        let _ = handle(stream, shared);
                    });
                }
            });
        }
        let streamer = Self {
            shared,
            resampler: Resampler::new(SAMPLE_RATE, STREAM_RATE),
            scratch: Vec::new(),
            pending: Vec::with_capacity(CHUNK),
        };
        Ok((streamer, browse_url(local)))
    }

    /// Queue 48 kHz audio for every connected listener.
    pub fn audio(&mut self, audio: &[f32]) {
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
        self.resampler.process(audio, &mut scratch);
        for &s in &scratch {
            self.pending.push((s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16);
            if self.pending.len() == CHUNK {
                self.send_pending();
            }
        }
        self.scratch = scratch;
    }

    /// Send whatever audio is buffered (at the end of a transmission).
    pub fn flush(&mut self) {
        if !self.pending.is_empty() {
            self.send_pending();
        }
    }

    /// Send a JSON event to every listener.
    pub fn event(&self, value: serde_json::Value) {
        broadcast(&mut lock(&self.shared), Message::Text(value.to_string().into()));
    }

    /// Announce a finished transmission, remembered for pages that connect later.
    /// If it was saved to `file`, the event gets a `url` that plays it.
    pub fn transmission(&self, mut event: serde_json::Value, file: Option<&Path>) {
        let mut shared = lock(&self.shared);
        if let Some(path) = file {
            shared.files.push(path.to_owned());
            event["url"] = serde_json::json!(format!("/rec/{}.wav", shared.files.len() - 1));
        }
        let text = event.to_string();
        if shared.history.len() == HISTORY {
            shared.history.pop_front();
        }
        shared.history.push_back(text.clone());
        broadcast(&mut shared, Message::Text(text.into()));
    }

    fn send_pending(&mut self) {
        let bytes: Vec<u8> = self.pending.iter().flat_map(|s| s.to_le_bytes()).collect();
        self.pending.clear();
        broadcast(&mut lock(&self.shared), Message::Binary(bytes.into()));
    }
}

fn broadcast(shared: &mut Shared, msg: Message) {
    // A full queue means a slow client: skip this message for it. A closed
    // queue means it disconnected: forget it.
    shared
        .clients
        .retain(|c| !matches!(c.try_send(msg.clone()), Err(TrySendError::Disconnected(_))));
}

/// Serve the page and recordings for plain HTTP requests, or upgrade `/ws` to a WebSocket.
fn handle(stream: TcpStream, shared: SharedRef) -> Result<()> {
    stream.set_nodelay(true)?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    let mut head = [0u8; 1024];
    let n = stream.peek(&mut head)?;
    let first = String::from_utf8_lossy(&head[..n]);
    let path = first.split_whitespace().nth(1).unwrap_or("/").to_string();

    if path.starts_with("/ws") {
        let mut ws =
            tungstenite::accept(stream).map_err(|e| anyhow::anyhow!("websocket handshake: {e}"))?;
        let (tx, rx) = sync_channel(CLIENT_QUEUE);
        // Copy the history and register in one step, so no event is missed or doubled.
        let history: Vec<String> = {
            let mut shared = lock(&shared);
            shared.clients.push(tx);
            shared.history.iter().cloned().collect()
        };
        for text in history {
            ws.send(Message::Text(text.into()))?;
        }
        pump(ws, rx);
        return Ok(());
    }

    let mut stream = stream;
    let request = read_request(&mut stream)?;
    if path == "/" || path.starts_with("/?") {
        return respond(&mut stream, "200 OK", "text/html; charset=utf-8", &[], PAGE.as_bytes());
    }
    if let Some(data) = recording(&shared, &path).and_then(|f| std::fs::read(f).ok()) {
        return serve_bytes(&mut stream, &request, "audio/wav", &data);
    }
    respond(&mut stream, "404 Not Found", "text/plain", &[], b"not found\n")
}

/// The saved recording behind `/rec/<n>.wav`, if any.
fn recording(shared: &SharedRef, path: &str) -> Option<PathBuf> {
    let n: usize = path.strip_prefix("/rec/")?.strip_suffix(".wav")?.parse().ok()?;
    lock(shared).files.get(n).cloned()
}

/// Read the request head (up to the blank line).
fn read_request(stream: &mut TcpStream) -> Result<String> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    while !buf.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte)? == 0 {
            break;
        }
        buf.push(byte[0]);
        if buf.len() > 16 * 1024 {
            bail!("request head too large");
        }
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Answer with `data`, honouring a single `Range: bytes=a-b` header.
fn serve_bytes(stream: &mut TcpStream, request: &str, kind: &str, data: &[u8]) -> Result<()> {
    let len = data.len();
    let range = request.lines().find_map(|l| {
        let (name, value) = l.split_once(':')?;
        name.trim().eq_ignore_ascii_case("range").then(|| value.trim().to_string())
    });
    let Some(spec) = range.as_deref().and_then(|r| r.strip_prefix("bytes=")) else {
        return respond(stream, "200 OK", kind, &["Accept-Ranges: bytes".into()], data);
    };
    let (a, b) = spec.split_once('-').unwrap_or((spec, ""));
    let (start, end) = match (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
        (Ok(s), Ok(e)) => (s, e.min(len.saturating_sub(1))),
        (Ok(s), Err(_)) => (s, len.saturating_sub(1)),
        // "bytes=-n": the last n bytes.
        (Err(_), Ok(n)) => (len.saturating_sub(n), len.saturating_sub(1)),
        _ => (0, len.saturating_sub(1)),
    };
    if len == 0 || start > end || start >= len {
        let headers = [format!("Content-Range: bytes */{len}")];
        return respond(stream, "416 Range Not Satisfiable", "text/plain", &headers, b"");
    }
    let headers = ["Accept-Ranges: bytes".into(), format!("Content-Range: bytes {start}-{end}/{len}")];
    respond(stream, "206 Partial Content", kind, &headers, &data[start..=end])
}

fn respond(stream: &mut TcpStream, status: &str, kind: &str, extra: &[String], body: &[u8]) -> Result<()> {
    let mut head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n",
        body.len()
    );
    for h in extra {
        head.push_str(h);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    Ok(())
}

/// Forward queued messages to one WebSocket client until it goes away.
fn pump(mut ws: tungstenite::WebSocket<TcpStream>, rx: Receiver<Message>) {
    for msg in rx {
        if ws.send(msg).is_err() {
            break;
        }
    }
}

/// A URL that another device on the LAN can open.
fn browse_url(local: SocketAddr) -> String {
    let host = if local.ip().is_unspecified() {
        // The address the default route would use; no packet is sent.
        UdpSocket::bind("0.0.0.0:0")
            .and_then(|s| s.connect("192.0.2.1:9").map(|_| s))
            .and_then(|s| s.local_addr())
            .map(|a| a.ip().to_string())
            .unwrap_or_else(|_| "localhost".into())
    } else {
        local.ip().to_string()
    };
    format!("http://{host}:{}/", local.port())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufRead;

    #[test]
    fn serves_page_and_streams_audio_and_events() {
        let (mut streamer, url) = Streamer::start("127.0.0.1:0").unwrap();
        let addr = url.trim_start_matches("http://").trim_end_matches('/').to_string();

        // Plain HTTP gets the player page.
        let mut http = TcpStream::connect(&addr).unwrap();
        write!(http, "GET / HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
        let mut reader = std::io::BufReader::new(http);
        let mut status = String::new();
        reader.read_line(&mut status).unwrap();
        assert!(status.starts_with("HTTP/1.1 200"), "{status}");
        let mut rest = String::new();
        reader.read_to_string(&mut rest).unwrap();
        assert!(rest.contains("<title>"), "page body served");

        // WebSocket gets JSON events and PCM.
        let (mut ws, _) = tungstenite::client::client(
            format!("ws://{addr}/ws"),
            TcpStream::connect(&addr).unwrap(),
        )
        .unwrap();
        // Wait until the server has registered the client.
        for _ in 0..100 {
            if !lock(&streamer.shared).clients.is_empty() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        streamer.event(serde_json::json!({"type": "open"}));
        streamer.audio(&vec![0.25; SAMPLE_RATE as usize / 5]); // 200 ms
        streamer.flush();

        let Message::Text(t) = ws.read().unwrap() else { panic!("expected text first") };
        assert!(t.contains("\"open\""));
        let mut samples = 0;
        while samples < STREAM_RATE as usize / 5 - 100 {
            match ws.read().unwrap() {
                Message::Binary(b) => {
                    assert_eq!(b.len() % 2, 0);
                    samples += b.len() / 2;
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        assert!(samples <= STREAM_RATE as usize / 5);
    }

    /// Send a raw HTTP request; return the status line, headers and body.
    fn http(addr: &str, request: &str) -> (String, String, Vec<u8>) {
        let mut s = TcpStream::connect(addr).unwrap();
        s.write_all(request.as_bytes()).unwrap();
        let mut all = Vec::new();
        s.read_to_end(&mut all).unwrap();
        let split = all.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        let head = String::from_utf8_lossy(&all[..split]).into_owned();
        let (status, headers) = head.split_once("\r\n").unwrap();
        (status.to_string(), headers.to_string(), all[split + 4..].to_vec())
    }

    #[test]
    fn late_page_gets_history_and_can_fetch_recordings_in_ranges() {
        let (streamer, url) = Streamer::start("127.0.0.1:0").unwrap();
        let addr = url.trim_start_matches("http://").trim_end_matches('/').to_string();
        let dir = std::env::temp_dir().join(format!("sdrfun-serve-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("tx.wav");
        let data: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&wav, &data).unwrap();

        // Two transmissions finish before any page is open: one saved, one dropped.
        streamer.transmission(serde_json::json!({"type": "tx", "verdict": "kept"}), Some(&wav));
        streamer.transmission(serde_json::json!({"type": "tx", "verdict": "no_voice"}), None);

        let (mut ws, _) =
            tungstenite::client::client(format!("ws://{addr}/ws"), TcpStream::connect(&addr).unwrap())
                .unwrap();
        let read_json = |ws: &mut tungstenite::WebSocket<TcpStream>| -> serde_json::Value {
            let Message::Text(t) = ws.read().unwrap() else { panic!("expected text") };
            serde_json::from_str(&t).unwrap()
        };
        let first = read_json(&mut ws);
        let second = read_json(&mut ws);
        assert_eq!(first["verdict"], "kept");
        assert_eq!(first["url"], "/rec/0.wav");
        assert_eq!(second["verdict"], "no_voice");
        assert!(second.get("url").is_none());

        let (status, headers, body) = http(&addr, "GET /rec/0.wav HTTP/1.1\r\nHost: x\r\n\r\n");
        assert!(status.contains("200"), "{status}");
        assert!(headers.contains("Accept-Ranges: bytes"));
        assert_eq!(body, data);

        let (status, headers, body) =
            http(&addr, "GET /rec/0.wav HTTP/1.1\r\nHost: x\r\nRange: bytes=0-1\r\n\r\n");
        assert!(status.contains("206"), "{status}");
        assert!(headers.contains("Content-Range: bytes 0-1/1000"), "{headers}");
        assert_eq!(body, &data[..2]);

        let (status, _, body) =
            http(&addr, "GET /rec/0.wav HTTP/1.1\r\nHost: x\r\nRange: bytes=990-\r\n\r\n");
        assert!(status.contains("206"));
        assert_eq!(body, &data[990..]);

        // Only recordings of this run are reachable.
        for path in ["/rec/1.wav", "/rec/../Cargo.toml", "/etc/passwd"] {
            let (status, _, _) = http(&addr, &format!("GET {path} HTTP/1.1\r\nHost: x\r\n\r\n"));
            assert!(status.contains("404"), "{path}: {status}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
