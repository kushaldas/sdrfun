# sdrfun

Listen to voice on an RTL-SDR (tested with the RTL-SDR Blog V4). `listen` and `scan` clean up AM
airband or FM voice, play it on the speaker and save each voice transmission as a WAV file. `web` is
an interactive receiver for your phone, similar to OpenWebRX but designed for a phone screen.

## Build requirements

| | Fedora | Debian / Ubuntu |
|---|---|---|
| librtlsdr | `sudo dnf install rtl-sdr rtl-sdr-devel` | `sudo apt install rtl-sdr librtlsdr-dev` |
| ALSA (audio output) | `sudo dnf install alsa-lib-devel` | `sudo apt install libasound2-dev` |
| pkg-config, C compiler | `sudo dnf install pkgconf-pkg-config gcc` | `sudo apt install pkg-config build-essential` |

```
cargo build --release
```

The udev rules from the `rtl-sdr` package give your user access to the dongle; re-plug it after installing.

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
sdrfun devices                      # RTL-SDR devices and audio outputs
sdrfun listen --help                # every option, with its default
```

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
- **Settings:** RF gain, span and filter bandwidth; per phone, a low-data mode and playing on with the
  screen locked. Audio is sent uncompressed (about 770 kbit/s for FM broadcast, 380 kbit/s for the
  voice modes); low data compresses it to about 100 kbit/s at some cost in quality.
- **Gain:** strong FM stations overload the dongle at the default 32.8 dB gain. When the meter shows
  *ADC CLIP*, lower the gain in Settings until it goes away.
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

sdrfun links against librtlsdr (GPL-2.0-or-later), so binaries are distributed under the GPL.
Other notable dependencies: [nnnoiseless](https://github.com/jneem/nnnoiseless) (BSD-3-Clause, a port
of Xiph's RNNoise including its model), ALSA `libasound` (LGPL-2.1-or-later, linked dynamically),
cpal and hound (Apache-2.0). The remaining crates are MIT and/or Apache-2.0.
