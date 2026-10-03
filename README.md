# sdrfun

Listen to AM airband voice on an RTL-SDR (tested with the RTL-SDR Blog V4). The tool cleans up
the audio, plays it on the speaker and saves each voice transmission as a WAV file.

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
sdrfun clean in.wav --play          # run the cleanup chain on a WAV file and listen
sdrfun devices                      # RTL-SDR devices and audio outputs
sdrfun listen --help                # every option, with its default
```

Recordings go to `recordings/YYYY-MM-DD/<freq>_<UTC time>_<seconds>s.wav`. One line per transmission
(kept or dropped, with the reason) is appended to `recordings/log.jsonl`.

## License

GPL-3.0-or-later, see [LICENSE](LICENSE).

sdrfun links against librtlsdr (GPL-2.0-or-later), so binaries are distributed under the GPL.
Other notable dependencies: [nnnoiseless](https://github.com/jneem/nnnoiseless) (BSD-3-Clause, a port
of Xiph's RNNoise including its model), ALSA `libasound` (LGPL-2.1-or-later, linked dynamically),
cpal and hound (Apache-2.0). The remaining crates are MIT and/or Apache-2.0.
