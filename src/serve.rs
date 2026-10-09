//! Web serving for phones. `Hub` is a small HTTP server for one page plus a WebSocket at
//! `/ws`; `Streamer` uses it for `listen --serve`, `web.rs` for `sdrfun web`.
//!
//! `Streamer`: binary messages are Opus packets with FEC (`AUDIO_OPUS`, see `opus.rs`);
//! text messages are JSON events (`status`, `open`, `tx`). A new client first receives
//! every `tx` event of the run so far. Saved transmissions carry a `url` (`/rec/<n>.wav`)
//! that serves the recording, with byte ranges as iOS Safari requires for audio. The page
//! decodes Opus with the bundled WASM decoder served at `/opus-decoder.js`.
//!
//! Clients may send JSON text messages, which are forwarded to the hub's inbound channel.
//! A hub with an inbound channel (`sdrfun web`) sends typed binary messages (the first byte
//! is the type) and handles `{"cmd": "lowdata", "on": bool}` per client: low-data clients
//! get compressed audio (`AUDIO_LOW`) and fewer waterfall rows (`WATERFALL_TYPES`), the
//! others full-quality audio (`AUDIO_FULL`).

use std::collections::{BTreeMap, VecDeque};
use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, UdpSocket};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, SyncSender, TryRecvError, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use tungstenite::Message;

/// Messages queued per client before it is considered too slow and messages are dropped.
const CLIENT_QUEUE: usize = 512;
/// Transmission events replayed to a newly connected page.
const HISTORY: usize = 500;
const PAGE: &str = include_str!("serve.html");
/// The WASM Opus decoder the pages load (see `assets/`), served with its license
/// notices (MIT for opus-decoder, BSD-3-Clause for the libopus inside it) as the
/// licenses require of every copy.
const OPUS_JS: &str = concat!(
    "/*!\n",
    include_str!("../assets/opus-decoder.LICENSE"),
    "*/\n",
    include_str!("../assets/opus-decoder.min.js")
);
/// Binary message types (first byte) that "low data" clients receive only some of.
pub const WATERFALL_TYPES: [u8; 2] = [2, 3];
/// The same audio twice: compressed for low-data clients, and full quality for the others.
pub const AUDIO_LOW: u8 = 1;
pub const AUDIO_FULL: u8 = 4;
/// `listen --serve` audio: one Opus stream for everyone.
pub const AUDIO_OPUS: u8 = 5;
/// A low-data client gets one in this many waterfall rows.
const LOWDATA_EVERY: u32 = 4;
/// How long a client thread waits for input before sending what is queued.
const POLL: Duration = Duration::from_millis(10);

/// One connected page: its outbound queue plus the waterfall row type it displays
/// (0 = all), so rows nobody looks at never enter the queue.
struct Client {
    tx: SyncSender<Message>,
    want: std::sync::atomic::AtomicU8,
}

struct Shared {
    page: &'static str,
    clients: Vec<Arc<Client>>,
    /// `tx` events of this run, oldest first.
    history: VecDeque<String>,
    /// Latest message of each kind (e.g. receiver state), sent to every new client.
    latest: BTreeMap<String, String>,
    /// Recordings that may be served, indexed by the number in their URL.
    files: Vec<PathBuf>,
    inbound: Option<Sender<serde_json::Value>>,
}

type SharedRef = Arc<Mutex<Shared>>;

fn lock(shared: &SharedRef) -> std::sync::MutexGuard<'_, Shared> {
    shared.lock().unwrap_or_else(|e| e.into_inner())
}

/// HTTP + WebSocket server for one page; clone it to share between threads.
#[derive(Clone)]
pub struct Hub {
    shared: SharedRef,
}

impl Hub {
    /// Start listening on `addr` (e.g. "0.0.0.0:8010"); returns the hub and the URL to open
    /// on another device. Client JSON messages go to `inbound`.
    pub fn start(
        addr: &str,
        page: &'static str,
        inbound: Option<Sender<serde_json::Value>>,
    ) -> Result<(Self, String)> {
        let listener = TcpListener::bind(addr).with_context(|| format!("binding {addr}"))?;
        let local = listener.local_addr()?;
        let shared = Arc::new(Mutex::new(Shared {
            page,
            clients: Vec::new(),
            history: VecDeque::new(),
            latest: BTreeMap::new(),
            files: Vec::new(),
            inbound,
        }));
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
        Ok((Self { shared }, browse_url(local)))
    }

    pub fn clients(&self) -> usize {
        lock(&self.shared).clients.len()
    }

    /// Send a JSON event to every client.
    pub fn event(&self, value: serde_json::Value) {
        broadcast(&mut lock(&self.shared), Message::Text(value.to_string().into()));
    }

    /// Send a JSON event to every client and remember it as the latest of its `kind`, so
    /// clients that connect later get it too.
    pub fn latest(&self, kind: &str, value: serde_json::Value) {
        let text = value.to_string();
        let mut shared = lock(&self.shared);
        shared.latest.insert(kind.to_string(), text.clone());
        broadcast(&mut shared, Message::Text(text.into()));
    }

    pub fn binary(&self, data: Vec<u8>) {
        broadcast(&mut lock(&self.shared), Message::Binary(data.into()));
    }
}

pub struct Streamer {
    hub: Hub,
    stream: crate::opus::Stream,
    msgs: Vec<Vec<u8>>,
}

impl Streamer {
    /// Start listening on `addr` (e.g. "0.0.0.0:8010"); returns the streamer and the
    /// URL to open on another device.
    pub fn start(addr: &str) -> Result<(Self, String)> {
        let (hub, url) = Hub::start(addr, PAGE, None)?;
        let streamer = Self {
            hub,
            stream: crate::opus::Stream::new(AUDIO_OPUS, 24, true)?,
            msgs: Vec::new(),
        };
        Ok((streamer, url))
    }

    /// Queue 48 kHz audio for every connected listener.
    pub fn audio(&mut self, audio: &[f32]) {
        self.stream.push(audio, &mut self.msgs);
        for msg in self.msgs.drain(..) {
            self.hub.binary(msg);
        }
    }

    /// Send whatever audio is buffered (at the end of a transmission).
    pub fn flush(&mut self) {
        self.stream.flush(&mut self.msgs);
        for msg in self.msgs.drain(..) {
            self.hub.binary(msg);
        }
    }

    /// Send a JSON event to every listener.
    pub fn event(&self, value: serde_json::Value) {
        self.hub.event(value);
    }

    /// Announce a finished transmission, remembered for pages that connect later.
    /// If it was saved to `file`, the event gets a `url` that plays it.
    pub fn transmission(&self, mut event: serde_json::Value, file: Option<&Path>) {
        let mut shared = lock(&self.hub.shared);
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
}

fn broadcast(shared: &mut Shared, msg: Message) {
    // A full queue means a slow client: skip this message for it. A closed
    // queue means it disconnected: forget it. Waterfall rows the client does not
    // display are skipped before queueing, so they can never crowd out audio.
    let kind = match &msg {
        Message::Binary(b) => b.first().copied().unwrap_or(0),
        _ => 0,
    };
    let is_row = matches!(kind, 2 | 3 | 6);
    shared.clients.retain(|c| {
        let want = c.want.load(std::sync::atomic::Ordering::Relaxed);
        if is_row && want != 0 && want != kind {
            return true;
        }
        !matches!(c.tx.try_send(msg.clone()), Err(TrySendError::Disconnected(_)))
    });
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
        let client = Arc::new(Client {
            tx,
            want: std::sync::atomic::AtomicU8::new(0),
        });
        // Copy the greeting and register in one step, so no event is missed or doubled.
        let (greeting, inbound) = {
            let mut shared = lock(&shared);
            shared.clients.push(client.clone());
            let greeting: Vec<String> =
                shared.latest.values().chain(shared.history.iter()).cloned().collect();
            (greeting, shared.inbound.clone())
        };
        for text in greeting {
            ws.send(Message::Text(text.into()))?;
        }
        pump(ws, rx, inbound, client);
        return Ok(());
    }

    let mut stream = stream;
    let request = read_request(&mut stream)?;
    if path == "/opus-decoder.js" {
        return respond(
            &mut stream,
            "200 OK",
            "text/javascript; charset=utf-8",
            &["Cache-Control: public, max-age=86400".into()],
            OPUS_JS.as_bytes(),
        );
    }
    if path == "/" || path.starts_with("/?") {
        let page = lock(&shared).page;
        return respond(&mut stream, "200 OK", "text/html; charset=utf-8", &[], page.as_bytes());
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

/// Exchange messages with one WebSocket client until it goes away: forward its JSON
/// messages to `inbound` and send it what is queued for it.
fn pump(
    mut ws: tungstenite::WebSocket<TcpStream>,
    rx: Receiver<Message>,
    inbound: Option<Sender<serde_json::Value>>,
    client: Arc<Client>,
) {
    if ws.get_ref().set_read_timeout(Some(POLL)).is_err() {
        return;
    }
    let mut lowdata = false;
    // Per row type: the types arrive interleaved, so one shared count would always keep
    // the same type.
    let mut rows = [0u32; WATERFALL_TYPES.len()];
    loop {
        match ws.read() {
            Ok(Message::Text(text)) => {
                let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else { continue };
                if value["cmd"] == "lowdata" {
                    lowdata = value["on"].as_bool().unwrap_or(false);
                } else if value["cmd"] == "want" {
                    client.want.store(value["rows"].as_u64().unwrap_or(0) as u8,
                        std::sync::atomic::Ordering::Relaxed);
                } else if let Some(inbound) = &inbound {
                    let _ = inbound.send(value);
                }
            }
            Ok(Message::Close(_)) => return,
            Ok(_) => {}
            Err(tungstenite::Error::Io(e))
                if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(_) => return,
        }
        loop {
            let msg = match rx.try_recv() {
                Ok(msg) => msg,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            };
            // Only typed hubs (with an inbound channel); `listen` sends untyped PCM.
            if let (Message::Binary(b), Some(_)) = (&msg, &inbound) {
                let kind = b.first().copied().unwrap_or_default();
                if kind == if lowdata { AUDIO_FULL } else { AUDIO_LOW } {
                    continue;
                }
                if lowdata && let Some(i) = WATERFALL_TYPES.iter().position(|&t| t == kind) {
                    rows[i] = rows[i].wrapping_add(1);
                    if !rows[i].is_multiple_of(LOWDATA_EVERY) {
                        continue;
                    }
                }
            }
            if ws.send(msg).is_err() {
                return;
            }
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

        // WebSocket gets JSON events and Opus packets.
        let (mut ws, _) = tungstenite::client::client(
            format!("ws://{addr}/ws"),
            TcpStream::connect(&addr).unwrap(),
        )
        .unwrap();
        // Wait until the server has registered the client.
        for _ in 0..100 {
            if streamer.hub.clients() > 0 {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        streamer.event(serde_json::json!({"type": "open"}));
        streamer.audio(&vec![0.25; 48_000 / 5]); // 200 ms at 48 kHz
        streamer.flush();

        let Message::Text(t) = ws.read().unwrap() else { panic!("expected text first") };
        assert!(t.contains("\"open\""));
        let mut dec = opus::Decoder::new(48_000, opus::Channels::Mono).unwrap();
        let mut out = [0.0f32; 2 * 960];
        let (mut samples, mut packets) = (0, 0);
        while packets < 10 {
            match ws.read().unwrap() {
                Message::Binary(b) => {
                    assert_eq!(b[0], AUDIO_OPUS);
                    let (seq, cur, prev) = crate::opus::split(&b).unwrap();
                    assert_eq!(seq, packets);
                    assert_eq!(!prev.is_empty(), packets > 0, "FEC copy from the second packet on");
                    samples += dec.decode_float(cur, &mut out, false).unwrap();
                    packets += 1;
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(samples, 48_000 / 5);
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

    #[test]
    fn hub_forwards_commands_greets_late_clients_and_thins_rows_for_low_data() {
        let (tx, rx) = std::sync::mpsc::channel();
        let (hub, url) = Hub::start("127.0.0.1:0", "<title>t</title>", Some(tx)).unwrap();
        let addr = url.trim_start_matches("http://").trim_end_matches('/').to_string();
        hub.latest("state", serde_json::json!({"type": "state", "hz": 1}));
        hub.latest("state", serde_json::json!({"type": "state", "hz": 2}));

        let connect = || {
            tungstenite::client::client(format!("ws://{addr}/ws"), TcpStream::connect(&addr).unwrap())
                .unwrap()
                .0
        };
        let mut a = connect();
        let mut b = connect();
        for ws in [&mut a, &mut b] {
            let Message::Text(t) = ws.read().unwrap() else { panic!("expected state") };
            assert!(t.contains("\"hz\":2"), "only the latest state: {t}");
        }

        a.send(Message::Text(r#"{"cmd":"tune","hz":145500000}"#.into())).unwrap();
        let cmd = rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(cmd["hz"], 145_500_000);

        b.send(Message::Text(r#"{"cmd":"lowdata","on":true}"#.into())).unwrap();
        // The lowdata command is handled by the client thread, not forwarded.
        assert!(rx.recv_timeout(Duration::from_millis(200)).is_err());
        for _ in 0..8 {
            hub.binary(vec![2, 0, 0]);
            hub.binary(vec![3, 0, 0]);
        }
        hub.binary(vec![AUDIO_LOW, 9]);
        hub.binary(vec![AUDIO_FULL, 9]);
        let count_rows = |ws: &mut tungstenite::WebSocket<TcpStream>| {
            let mut rows = (0, 0);
            loop {
                match ws.read().unwrap() {
                    Message::Binary(m) if m[0] == 2 => rows.0 += 1,
                    Message::Binary(m) if m[0] == 3 => rows.1 += 1,
                    // Each client gets exactly one of the two audio messages.
                    Message::Binary(m) if m[0] == AUDIO_LOW || m[0] == AUDIO_FULL => return (rows, m[0]),
                    _ => {}
                }
            }
        };
        assert_eq!(count_rows(&mut a), ((8, 8), AUDIO_FULL));
        let thinned = 8 / LOWDATA_EVERY as usize;
        assert_eq!(count_rows(&mut b), ((thinned, thinned), AUDIO_LOW));
    }
}
