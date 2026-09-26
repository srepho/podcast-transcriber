//! YAML configuration.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const DEFAULT_CONFIG: &str = "config.yaml";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Feed {
    pub name: String,
    pub url: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Feed-specific context for whisper (e.g. "An NBA basketball podcast hosted by Jane Host
    /// and Sam Cohost."). Combined with the episode title and vocabulary at transcription time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct WhisperSettings {
    /// ggml model name from the whisper.cpp HF repo, e.g. tiny.en, base.en, small.en, medium.en,
    /// large-v3, large-v3-turbo. Downloaded on demand into <data_dir>/models.
    pub model: String,
    /// Explicit path to a ggml model file; overrides `model` if set.
    pub model_path: Option<PathBuf>,
    /// Language code (e.g. "en"). None = auto-detect (multilingual models only).
    pub language: Option<String>,
    /// CPU threads for inference.
    pub threads: usize,
    /// Beam size. 1 = greedy (fastest). 5 = whisper.cpp default (more accurate, slower).
    pub beam_size: usize,
    /// Optional initial prompt to bias vocabulary (names, jargon).
    pub initial_prompt: Option<String>,
}

impl Default for WhisperSettings {
    fn default() -> Self {
        Self {
            model: "base.en".into(),
            model_path: None,
            language: Some("en".into()),
            threads: default_threads(),
            beam_size: 1,
            initial_prompt: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Settings {
    /// Where audio, transcripts, models and the state DB live.
    pub data_dir: PathBuf,
    /// When a feed is first added, only queue this many newest episodes. Older ones are
    /// recorded as `skipped` and can be queued later with `podcast backfill`.
    pub max_new_per_feed: usize,
    /// Delete audio after a successful transcription.
    pub delete_audio_after_transcribe: bool,
    /// Transcript formats to write: txt, srt, json.
    pub formats: Vec<String>,
    /// When following feed pagination during backfill, stop after this many pages.
    pub max_feed_pages: usize,
    /// Similarity (0-1) a transcript fragment needs to be replaced by a vocabulary name.
    /// Lower = more aggressive. 0.8 is a good default; 0.75 catches badly split names.
    pub correction_threshold: f64,
    pub whisper: WhisperSettings,
    pub feeds: Vec<Feed>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("data"),
            max_new_per_feed: 3,
            delete_audio_after_transcribe: false,
            formats: vec!["txt".into(), "srt".into(), "json".into()],
            max_feed_pages: 50,
            correction_threshold: 0.8,
            whisper: WhisperSettings::default(),
            feeds: vec![],
        }
    }
}

impl Settings {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let settings: Settings = serde_yaml::from_str(&text)
            .with_context(|| format!("parsing config {}", path.display()))?;
        Ok(settings)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let text = serde_yaml::to_string(self)?;
        std::fs::write(path, text).with_context(|| format!("writing config {}", path.display()))
    }

    pub fn audio_dir(&self) -> PathBuf {
        self.data_dir.join("audio")
    }
    pub fn transcript_dir(&self) -> PathBuf {
        self.data_dir.join("transcripts")
    }
    pub fn model_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }
    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("podcast.db")
    }
    pub fn vocab_path(&self, feed_name: &str) -> PathBuf {
        self.data_dir.join("vocab").join(format!("{feed_name}.txt"))
    }

    pub fn model_path(&self) -> PathBuf {
        match &self.whisper.model_path {
            Some(p) => p.clone(),
            None => self
                .model_dir()
                .join(format!("ggml-{}.bin", self.whisper.model)),
        }
    }

    pub fn feed(&self, name: &str) -> Option<&Feed> {
        self.feeds.iter().find(|f| f.name == name)
    }
}

fn default_true() -> bool {
    true
}

fn default_threads() -> usize {
    // whisper.cpp scales poorly past physical cores; hyperthreads add little.
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).clamp(1, 8))
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_yaml() {
        let mut s = Settings::default();
        s.feeds.push(Feed {
            name: "x".into(),
            url: "https://example.com/rss".into(),
            enabled: true,
            prompt: None,
        });
        let text = serde_yaml::to_string(&s).unwrap();
        let back: Settings = serde_yaml::from_str(&text).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn partial_yaml_uses_defaults() {
        let s: Settings = serde_yaml::from_str("feeds:\n  - name: a\n    url: u\n").unwrap();
        assert_eq!(s.feeds.len(), 1);
        assert!(s.feeds[0].enabled);
        assert_eq!(s.whisper.model, "base.en");
        assert_eq!(
            s.model_path(),
            PathBuf::from("data/models/ggml-base.en.bin")
        );
    }
}
