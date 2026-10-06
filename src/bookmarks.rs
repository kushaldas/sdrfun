//! Bookmarks for `sdrfun web`: a JSON file shared by every connected phone, seeded with
//! useful defaults (Arlanda airband, Region 1 amateur calling frequencies, PMR446, marine
//! VHF, Stockholm FM broadcast) when it does not exist yet.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::channel::Mode;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bookmark {
    pub id: u32,
    pub name: String,
    pub hz: f64,
    /// Mode name as in `Mode::name`.
    pub mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bandwidth: Option<f32>,
    /// Squelch in dB over the floor; absent means the mode's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub squelch: Option<f32>,
}

pub struct Bookmarks {
    path: PathBuf,
    pub items: Vec<Bookmark>,
}

const DEFAULTS: &[(f64, &str, Mode)] = &[
    (88.300, "P4 Stockholm", Mode::Wfm),
    (89.600, "P1 Stockholm", Mode::Wfm),
    (96.200, "P2 Stockholm", Mode::Wfm),
    (99.300, "P3 Stockholm", Mode::Wfm),
    (118.500, "Arlanda TWR 01L/19R", Mode::Am),
    (119.000, "Arlanda ATIS arrival", Mode::Am),
    (121.500, "Air emergency", Mode::Am),
    (121.625, "Arlanda ATIS departure", Mode::Am),
    (121.700, "Arlanda GND W", Mode::Am),
    (123.750, "Stockholm APP/CON", Mode::Am),
    (125.125, "Arlanda TWR 01R/19L", Mode::Am),
    (144.300, "2 m SSB calling", Mode::Usb),
    (145.500, "2 m FM calling", Mode::Nfm),
    (156.800, "Marine VHF 16", Mode::Nfm),
    (433.500, "70 cm FM calling", Mode::Nfm),
    (446.00625, "PMR446 ch 1", Mode::Nfm),
];

impl Bookmarks {
    /// Load `path`, or create it with the defaults.
    pub fn open(path: &Path) -> Result<Self> {
        if !path.exists() {
            let items = DEFAULTS
                .iter()
                .zip(1..)
                .map(|(&(mhz, name, mode), id)| Bookmark {
                    id,
                    name: name.into(),
                    hz: (mhz * 1e6).round(),
                    mode: mode.name().into(),
                    bandwidth: None,
                    squelch: None,
                })
                .collect();
            let b = Self { path: path.to_owned(), items };
            b.save()?;
            return Ok(b);
        }
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let items: Vec<Bookmark> =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        Ok(Self { path: path.to_owned(), items })
    }

    pub fn add(&mut self, mut b: Bookmark) -> Result<()> {
        if Mode::from_name(&b.mode).is_none() {
            bail!("unknown mode {:?}", b.mode);
        }
        if b.name.trim().is_empty() {
            bail!("bookmark needs a name");
        }
        b.name = b.name.trim().chars().take(60).collect();
        b.id = self.items.iter().map(|i| i.id).max().unwrap_or(0) + 1;
        self.items.push(b);
        self.items.sort_by(|a, b| a.hz.total_cmp(&b.hz));
        self.save()
    }

    pub fn remove(&mut self, id: u32) -> Result<()> {
        let before = self.items.len();
        self.items.retain(|b| b.id != id);
        if self.items.len() == before {
            bail!("no bookmark {id}");
        }
        self.save()
    }

    /// Write to a temporary file and rename it over the old one, so a crash never
    /// leaves a half-written file.
    fn save(&self) -> Result<()> {
        if let Some(dir) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&self.items)? + "\n")
            .with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &self.path).with_context(|| format!("replacing {}", self.path.display()))
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({"type": "bookmarks", "items": self.items})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_defaults_then_adds_and_removes() {
        let dir = std::env::temp_dir().join(format!("sdrfun-bm-{}", std::process::id()));
        let path = dir.join("bookmarks.json");
        let _ = std::fs::remove_dir_all(&dir);

        let mut b = Bookmarks::open(&path).unwrap();
        assert_eq!(b.items.len(), DEFAULTS.len());
        assert!(b.items.iter().all(|i| Mode::from_name(&i.mode).is_some()));
        let n = b.items.len();
        b.add(Bookmark {
            id: 0,
            name: " Repeater ".into(),
            hz: 145_600_000.0,
            mode: "nfm".into(),
            bandwidth: None,
            squelch: Some(8.0),
        })
        .unwrap();
        assert!(b.add(Bookmark { id: 0, name: "x".into(), hz: 1.0, mode: "dmr".into(), bandwidth: None, squelch: None }).is_err());

        let reloaded = Bookmarks::open(&path).unwrap();
        assert_eq!(reloaded.items.len(), n + 1);
        let added = reloaded.items.iter().find(|i| i.name == "Repeater").unwrap().clone();
        assert_eq!(added.id as usize, n + 1);
        assert_eq!(added.squelch, Some(8.0));

        let mut b = reloaded;
        b.remove(added.id).unwrap();
        assert!(b.remove(9999).is_err());
        assert_eq!(Bookmarks::open(&path).unwrap().items.len(), n);
        assert!(!path.with_extension("json.tmp").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
