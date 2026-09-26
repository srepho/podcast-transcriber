//! whisper.cpp inference and transcript writers (txt / srt / json).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::WhisperSettings;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Segment {
    /// seconds
    pub start: f64,
    pub end: f64,
    /// Text after vocabulary correction.
    pub text: String,
    /// Text exactly as whisper produced it (kept so corrections can be re-run later).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    pub feed: String,
    pub title: String,
    pub guid: String,
    pub published: Option<String>,
    pub audio_url: String,
    pub model: String,
    pub language: Option<String>,
    pub duration_secs: f64,
    /// The whisper initial prompt that was used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default)]
    pub corrections: Vec<crate::vocab::Correction>,
    pub segments: Vec<Segment>,
}

impl Transcript {
    /// Re-run vocabulary correction from the raw whisper text.
    pub fn apply_vocab(&mut self, vocab: &crate::vocab::Vocab, threshold: f64) {
        self.corrections.clear();
        for seg in &mut self.segments {
            let raw = seg.raw_text.clone().unwrap_or_else(|| seg.text.clone());
            let (fixed, corr) = vocab.correct(&raw, threshold);
            seg.raw_text = if fixed != raw { Some(raw) } else { None };
            seg.text = fixed;
            for c in corr {
                match self
                    .corrections
                    .iter_mut()
                    .find(|x| x.from == c.from && x.to == c.to)
                {
                    Some(x) => x.count += c.count,
                    None => self.corrections.push(c),
                }
            }
        }
    }

    pub fn read_json(path: &Path) -> Result<Transcript> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Ok(serde_json::from_str(&text)?)
    }

    pub fn plain_text(&self) -> String {
        // Whisper segments are sentence-ish fragments; join with spaces, paragraph every ~8.
        let mut out = String::new();
        for (i, s) in self.segments.iter().enumerate() {
            let t = s.text.trim();
            if t.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push(if i % 8 == 0 { '\n' } else { ' ' });
                if i % 8 == 0 {
                    out.push('\n');
                }
            }
            out.push_str(t);
        }
        out.push('\n');
        out
    }

    pub fn srt(&self) -> String {
        let mut out = String::new();
        for (i, s) in self.segments.iter().enumerate() {
            out.push_str(&format!(
                "{}\n{} --> {}\n{}\n\n",
                i + 1,
                srt_time(s.start),
                srt_time(s.end),
                s.text.trim()
            ));
        }
        out
    }

    /// Write the requested formats into `dir/<basename>.<fmt>`. Returns the primary (.txt if
    /// requested, else first) path.
    pub fn write(&self, dir: &Path, basename: &str, formats: &[String]) -> Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        let mut primary = None;
        for f in formats {
            let path = dir.join(format!("{basename}.{f}"));
            let body = match f.as_str() {
                "txt" => self.plain_text(),
                "srt" => self.srt(),
                "json" => serde_json::to_string_pretty(self)? + "\n",
                other => anyhow::bail!("unknown transcript format '{other}' (use txt, srt, json)"),
            };
            std::fs::write(&path, body).with_context(|| format!("writing {}", path.display()))?;
            if f == "txt" || primary.is_none() {
                primary = Some(path);
            }
        }
        primary.context("no transcript formats configured")
    }
}

pub fn srt_time(secs: f64) -> String {
    let ms = (secs * 1000.0).round().max(0.0) as u64;
    let (h, rem) = (ms / 3_600_000, ms % 3_600_000);
    let (m, rem) = (rem / 60_000, rem % 60_000);
    let (s, ms) = (rem / 1000, rem % 1000);
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// A loaded whisper model. Load once, transcribe many.
pub struct Engine {
    ctx: WhisperContext,
    settings: WhisperSettings,
    pub model_name: String,
}

impl Engine {
    pub fn load(model_path: &Path, settings: &WhisperSettings) -> Result<Self> {
        whisper_rs::install_logging_hooks(); // routes whisper.cpp/ggml stderr chatter to `log` (silent)
        let mut params = WhisperContextParameters::default();
        params.use_gpu(false);
        let ctx = WhisperContext::new_with_params(
            model_path
                .to_str()
                .context("model path is not valid UTF-8")?,
            params,
        )
        .with_context(|| format!("loading whisper model {}", model_path.display()))?;
        let model_name = model_path
            .file_stem()
            .map(|s| s.to_string_lossy().trim_start_matches("ggml-").to_string())
            .unwrap_or_default();
        Ok(Self {
            ctx,
            settings: settings.clone(),
            model_name,
        })
    }

    /// Transcribe 16 kHz mono f32 PCM. `on_progress` receives 0..=100.
    pub fn transcribe(
        &self,
        pcm: &[f32],
        prompt: Option<&str>,
        on_progress: impl FnMut(i32) + Send + 'static,
    ) -> Result<(Vec<Segment>, Option<String>)> {
        let strategy = if self.settings.beam_size <= 1 {
            SamplingStrategy::Greedy { best_of: 1 }
        } else {
            SamplingStrategy::BeamSearch {
                beam_size: self.settings.beam_size as i32,
                patience: -1.0,
            }
        };
        let mut params = FullParams::new(strategy);
        params.set_n_threads(self.settings.threads as i32);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_progress_callback_safe(on_progress);
        let lang = self.settings.language.clone();
        match (&lang, self.ctx.is_multilingual()) {
            (Some(l), true) => params.set_language(Some(l.as_str())),
            (None, true) => params.set_detect_language(true),
            (_, false) => params.set_language(Some("en")),
        }
        let prompt_owned = prompt
            .map(str::to_string)
            .or_else(|| self.settings.initial_prompt.clone());
        if let Some(p) = prompt_owned.as_deref().filter(|p| !p.trim().is_empty()) {
            params.set_initial_prompt(p);
        }

        let mut state = self.ctx.create_state().context("creating whisper state")?;
        state
            .full(params, pcm)
            .context("running whisper inference")?;

        let mut segments = Vec::with_capacity(state.full_n_segments() as usize);
        for seg in state.as_iter() {
            let text = seg.to_str_lossy()?.to_string();
            segments.push(Segment {
                start: seg.start_timestamp() as f64 / 100.0, // whisper timestamps are 10ms units
                end: seg.end_timestamp() as f64 / 100.0,
                text,
                raw_text: None,
            });
        }
        let detected = if self.ctx.is_multilingual() {
            let id = state.full_lang_id_from_state();
            whisper_rs::get_lang_str(id).map(|s| s.to_string())
        } else {
            Some("en".to_string())
        };
        Ok((segments, detected))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tr() -> Transcript {
        Transcript {
            feed: "f".into(),
            title: "t".into(),
            guid: "g".into(),
            published: None,
            audio_url: "u".into(),
            model: "base.en".into(),
            language: Some("en".into()),
            duration_secs: 3.5,
            prompt: None,
            corrections: vec![],
            segments: vec![
                Segment {
                    start: 0.0,
                    end: 1.25,
                    text: " Hello there.".into(),
                    raw_text: None,
                },
                Segment {
                    start: 1.25,
                    end: 3661.5,
                    text: " General Kenobi.".into(),
                    raw_text: None,
                },
            ],
        }
    }

    #[test]
    fn srt_time_format() {
        assert_eq!(srt_time(0.0), "00:00:00,000");
        assert_eq!(srt_time(1.25), "00:00:01,250");
        assert_eq!(srt_time(3661.5), "01:01:01,500");
        assert_eq!(srt_time(-1.0), "00:00:00,000");
    }

    #[test]
    fn srt_and_text_output() {
        let t = tr();
        assert_eq!(
            t.srt(),
            "1\n00:00:00,000 --> 00:00:01,250\nHello there.\n\n2\n00:00:01,250 --> 01:01:01,500\nGeneral Kenobi.\n\n"
        );
        assert_eq!(t.plain_text(), "Hello there. General Kenobi.\n");
    }

    #[test]
    fn write_formats() {
        let t = tr();
        let dir = tempfile::tempdir().unwrap();
        let p = t
            .write(dir.path(), "ep", &["json".into(), "txt".into()])
            .unwrap();
        assert_eq!(p, dir.path().join("ep.txt"));
        assert!(dir.path().join("ep.json").exists());
        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("ep.json")).unwrap())
                .unwrap();
        assert_eq!(parsed["segments"].as_array().unwrap().len(), 2);
        assert!(t.write(dir.path(), "ep", &["docx".into()]).is_err());
        let back = Transcript::read_json(&dir.path().join("ep.json")).unwrap();
        assert_eq!(back.segments, t.segments);
    }

    #[test]
    fn apply_vocab_is_repeatable_from_raw() {
        let mut t = tr();
        t.segments[0].text = " Hello Yokic.".into();
        let v = crate::vocab::Vocab::parse("Jokic\n");
        t.apply_vocab(&v, 0.8);
        assert_eq!(t.segments[0].text, " Hello Jokic.");
        assert_eq!(t.segments[0].raw_text.as_deref(), Some(" Hello Yokic."));
        assert_eq!(t.corrections.len(), 1);
        // second run with a different vocab starts from the raw text again
        let v2 = crate::vocab::Vocab::parse("yokic => Nikola Jokić\n");
        t.apply_vocab(&v2, 0.8);
        assert_eq!(t.segments[0].text, " Hello Nikola Jokić.");
    }
}
