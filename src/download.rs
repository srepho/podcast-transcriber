//! Streaming audio download with atomic `.part` -> final rename.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Filesystem-safe slug: keeps letters/digits, collapses everything else to a single '-'.
pub fn slug(text: &str, max_len: usize) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"[^\p{L}\p{N}]+").unwrap());
    let s = re.replace_all(text, "-").trim_matches('-').to_string();
    let mut out: String = s.chars().take(max_len).collect();
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        "episode".to_string()
    } else {
        out
    }
}

pub fn episode_basename(title: &str, published: Option<DateTime<Utc>>) -> String {
    let date = published
        .map(|d| d.format("%Y-%m-%d").to_string())
        .unwrap_or("undated".into());
    format!("{date}_{}", slug(title, 80))
}

pub fn extension_for(url: &str, content_type: Option<&str>) -> &'static str {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"\.(mp3|m4a|mp4|aac|ogg|opus|wav|flac)$").unwrap());
    let path = url
        .split(['?', '#'])
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    if let Some(m) = re.captures(&path) {
        return match m.get(1).unwrap().as_str() {
            "mp3" => "mp3",
            "m4a" => "m4a",
            "mp4" => "mp4",
            "aac" => "aac",
            "ogg" => "ogg",
            "opus" => "opus",
            "wav" => "wav",
            _ => "flac",
        };
    }
    match content_type.map(|c| {
        c.split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase()
    }) {
        Some(ct) if ct == "audio/mp4" || ct == "audio/x-m4a" => "m4a",
        Some(ct) if ct == "audio/aac" => "aac",
        Some(ct) if ct == "audio/ogg" => "ogg",
        Some(ct) if ct == "audio/opus" => "opus",
        Some(ct) if ct == "audio/wav" || ct == "audio/x-wav" => "wav",
        Some(ct) if ct == "audio/flac" => "flac",
        Some(ct) if ct == "video/mp4" => "mp4",
        _ => "mp3",
    }
}

/// Download `url` into `dir/<basename>.<ext>`. Skips if the final file already exists.
pub fn download(
    client: &reqwest::blocking::Client,
    url: &str,
    dir: &Path,
    basename: &str,
    show_progress: bool,
) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let mut resp = client
        .get(url)
        .send()
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("GET {url}"))?;
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let ext = extension_for(resp.url().as_str(), ct.as_deref());
    let final_path = dir.join(format!("{basename}.{ext}"));
    if final_path.exists() && final_path.metadata().map(|m| m.len() > 0).unwrap_or(false) {
        return Ok(final_path);
    }
    let part_path = dir.join(format!("{basename}.{ext}.part"));

    let total = resp.content_length();
    let bar = if show_progress {
        let b = match total {
            Some(t) => ProgressBar::new(t),
            None => ProgressBar::new_spinner(),
        };
        b.set_style(
            ProgressStyle::with_template(
                "  {bytes}/{total_bytes} {bar:30} {bytes_per_sec} eta {eta}",
            )
            .unwrap(),
        );
        Some(b)
    } else {
        None
    };

    {
        let mut file = std::fs::File::create(&part_path)?;
        let mut buf = vec![0u8; 1 << 16];
        loop {
            let n = resp.read(&mut buf)?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])?;
            if let Some(b) = &bar {
                b.inc(n as u64);
            }
        }
        file.flush()?;
    }
    if let Some(b) = bar {
        b.finish_and_clear();
    }
    std::fs::rename(&part_path, &final_path)?;
    Ok(final_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn slug_is_safe() {
        assert_eq!(
            slug("Hello, World! #42: The \"Return\"", 80),
            "Hello-World-42-The-Return"
        );
        assert_eq!(slug("///", 80), "episode");
        assert_eq!(slug("abcdef", 3), "abc");
        assert_eq!(slug("ab-cdef", 3), "ab");
        assert_eq!(slug("Épisode ünïcode", 80), "Épisode-ünïcode");
    }

    #[test]
    fn basename_uses_date() {
        let d = Utc.with_ymd_and_hms(2024, 3, 5, 1, 2, 3).unwrap();
        assert_eq!(episode_basename("A / B", Some(d)), "2024-03-05_A-B");
        assert_eq!(episode_basename("A", None), "undated_A");
    }

    #[test]
    fn extension_detection() {
        assert_eq!(extension_for("https://x/a.MP3?y=1", None), "mp3");
        assert_eq!(extension_for("https://x/a.m4a", Some("text/plain")), "m4a");
        assert_eq!(
            extension_for("https://x/stream", Some("audio/mp4; charset=x")),
            "m4a"
        );
        assert_eq!(extension_for("https://x/stream", Some("audio/mpeg")), "mp3");
        assert_eq!(extension_for("https://x/stream", None), "mp3");
    }
}
