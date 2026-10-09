# sdrfun

Listen to voice on an RTL-SDR (tested with the RTL-SDR Blog V4) or a HackRF One. `listen` and `scan`
clean up AM airband or FM voice, play it on the speaker and save each voice transmission as a WAV
file. `web` is an interactive receiver for your phone, similar to OpenWebRX but designed for a phone
screen, with a wideband sweep mode that hops the tuner across many MHz to find signals.

## Build requirements

| | Fedora | Debian / Ubuntu |
|---|---|---|
| librtlsdr | `sudo dnf install rtl-sdr rtl-sdr-devel` | `sudo apt install rtl-sdr librtlsdr-dev` |
| libhackrf (HackRF One) | `sudo dnf install hackrf hackrf-devel` | `sudo apt install hackrf libhackrf-dev` |
| ALSA (audio output) | `sudo dnf install alsa-lib-devel` | `sudo apt install libasound2-dev` |
| libopus (streamed audio) | `sudo dnf install opus-devel` | `sudo apt install libopus-dev` |
| pkg-config, C compiler, cmake | `sudo dnf install pkgconf-pkg-config gcc cmake` | `sudo apt install pkg-config build-essential cmake` |

```
cargo build --release
```

The udev rules from the `rtl-sdr` and `hackrf` packages give your user access to the radios;
re-plug them after installing.

## Docker

Everything above is packaged in the `Dockerfile`; the container needs raw USB access to the
dongle and publishes the web page on port 8010:

```
docker compose up --build -d     # build and start the receiver on http://localhost:8010/
docker compose logs -f           # follow the log
docker compose down              # stop
```

`docker-compose.yml` runs `sdrfun web --no-play` (the container has no sound card; audio
goes to the browser) and keeps bookmarks and recordings in the `sdrfun-data` volume. To run
`listen --serve` instead, change its `command:`. Without compose:

```
docker build -t sdrfun .
docker run --rm -p 8010:8010 --device /dev/bus/usb sdrfun web --no-play
```

## Usage

```
sdrfun listen                       # 120.150 MHz AM, all defaults; Ctrl-C to stop
sdrfun listen 118.700 --save-raw    # also keep the un-cleaned audio
sdrfun listen --no-play             # record only
sdrfun listen --serve               # also stream to a phone: open http://<this-pc>:8010/
sdrfun listen 145.5 --mode nfm      # amateur FM; also wfm, usb, lsb, cw
sdrfun scan                         # survey Stockholm Arlanda frequencies, report which carry voice
sdrfun scan 118.5=Tower 121.5=Guard --dwell 60 --rounds 3
sdrfun clean in.wav --play          # run the cleanup chain on a WAV file and listen
sdrfun devices                      # RTL-SDR and HackRF devices, and audio outputs
sdrfun listen --help                # every option, with its default
```

With both an RTL-SDR and a HackRF plugged in, pick which one to use with `--sdr rtlsdr|hackrf|both`
(or the `SDRFUN_SDR` environment variable); the default is whatever is attached. With `both`, the
web page gets a radio switch. The HackRF covers 10 MHz–6 GHz and hops frequency much faster, which
matters for sweep mode; the RTL-SDR has better receiver performance.

### Receiver on your phone

```
sdrfun web                          # 145.500 MHz NFM; open http://<this-pc>:8010/ on the phone
sdrfun web 99.3 --mode wfm          # start on an FM broadcast station
sdrfun web --no-play --gain auto    # sound only on the phone
```

The page shows a waterfall of the whole captured band (up to 2.4 MHz):
- **Tune:** tap the waterfall to tune, snapping to the mode's channel step. Pinch to zoom, drag to pan,
  and double-tap to zoom in or out. At deep zoom the waterfall switches to a high-resolution view around
  the tuned channel. Drag the yellow tuning line to tune by hand. Tap the frequency to type one
  (`145.5`, `7074k`, `446006.25k`), or drag it sideways to step.
- **Modes:** AM, NFM, WFM (mono), USB, LSB and CW.
- **Squelch and cleanup:** in AM and NFM the squelch starts out like `sdrfun listen`: it opens 8 dB
  over the noise floor (or on a carrier that was already on when you tuned in), stays open for 1.5 s
  after the carrier drops, and plays the 0.3 s before it opened. The SQL slider changes the margin;
  slide it fully left to hear everything, hiss included. A separate *Voice cleanup* switch applies
  the RNNoise voice cleanup in any mode; it is on by default for AM and NFM.
- **Settings:** RF gain, span, FFT size, scope range and filter bandwidth; per phone, a low-data mode and playing on with the
  screen locked. Audio is streamed as 48 kHz Opus with forward error correction (about 64 kbit/s for
  FM broadcast, 32 kbit/s for the voice modes), so a dropped packet costs no audio; low data
  compresses it to about 16 kbit/s at some cost in quality.
- **FFT size and scope range:** Settings picks the FFT size — the bins in every waterfall and spectrum
  row the Pi sends, 512 up to `--fft-max` (default 8192; `SDRFUN_FFT_MAX` in the environment does the
  same). Bigger sizes resolve closer signals but cost the Pi more CPU and every phone more data
  (about 0.5 byte per bin, 10 rows a second). `--fft` (or `SDRFUN_FFT`) is the size to start with,
  2048 by default. The Floor and Ceiling sliders fix the waterfall's dB range for everyone; *Re-scan
  range* hands it back to the automatic choice.
- **Sweep mode:** the `SWEEP` button beside the frequency display hops the
  tuner across a range far wider than the receiver's IQ bandwidth and paints one waterfall line per
  pass — like `hackrf_sweep`. Set the range by typing start and end frequencies in Settings, or just
  pinch and drag on the waterfall: in sweep mode the gestures resize the sweep itself. The HackRF
  sweeps 16 MHz per hop at 20 MS/s (a full pass of a 100 MHz range in about 5 s); the RTL-SDR hops
  2 MHz at 2.4 MS/s. Sweep is spectrum only — no audio — and tuning is paused until you switch it off.
- **Radio switch:** with `--sdr both`, chips switch the running receiver between the RTL-SDR and the
  HackRF without reloading the page; everyone connected follows.
- **Gain:** strong FM stations overload the dongle at the default 32.8 dB gain. When the meter shows
  *ADC CLIP*, lower the gain in Settings until it goes away. The HackRF uses its 0–40 dB LNA ladder.
- **Bookmarks:** shared by every phone, kept in `bookmarks.json` and drawn on the waterfall. The
  lock-screen next/previous buttons step through them.

Nothing is recorded in this mode. Everyone connected shares one receiver, and there is no password:
anyone who can reach the port can retune it, so only serve it on a trusted network (or over a VPN
such as WireGuard or Tailscale).

Below 28.8 MHz the V4 needs a librtlsdr that knows its HF upconverter (the RTL-SDR Blog fork does).

### Recordings

Recordings go to `recordings/YYYY-MM-DD/<freq>_<UTC time>_<seconds>s.wav`. One line per transmission
(kept or dropped, with the reason) is appended to `recordings/log.jsonl`.

## License

GPL-3.0-or-later, see [LICENSE](LICENSE).

sdrfun links against librtlsdr and libhackrf (both GPL-2.0-or-later), so binaries are distributed
under the GPL. Other notable dependencies: [nnnoiseless](https://github.com/jneem/nnnoiseless) (BSD-3-Clause, a port
of Xiph's RNNoise including its model), libopus (BSD-3-Clause, linked dynamically) and its
browser build [opus-decoder](https://github.com/eshaz/wasm-audio-decoders) (MIT, vendored in
`assets/`), ALSA `libasound` (LGPL-2.1-or-later, linked dynamically), cpal and hound
(Apache-2.0), and the `opus` crate's libopus bindings audiopus_sys (ISC). The remaining crates are
MIT and/or Apache-2.0 (a few also offer Unlicense or BlueOak-1.0.0; unicode-ident adds Unicode-3.0).
The opus-decoder and libopus license texts are in
[assets/opus-decoder.LICENSE](assets/opus-decoder.LICENSE) and are served with the decoder.
