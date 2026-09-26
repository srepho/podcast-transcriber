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

/// The readable prefix is not an identity: titles, dates and feed slugs can collide.
pub fn episode_filename(ep: &crate::db::Episode) -> String {
    let identity = serde_json::to_vec(&(&ep.feed_name, &ep.guid)).expect("string serialization");
    let digest = ring::digest::digest(&ring::digest::SHA256, &identity);
    let suffix: String = digest.as_ref()[..16]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    format!("{}_{}", episode_basename(&ep.title, ep.published), suffix)
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

/// Error pages and API responses served with a 200 status are markup or JSON, never audio.
fn is_non_audio_type(content_type: &str) -> bool {
    let ct = content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    // Not all of text/*: some hosts label real MP3s text/plain, and the body sniff covers markup.
    matches!(
        ct.as_str(),
        "text/html"
            | "text/xml"
            | "application/json"
            | "application/xml"
            | "application/xhtml+xml"
            | "application/rss+xml"
    )
}

/// True when the first non-whitespace byte opens markup or JSON. No audio container starts
/// with `<` or `{`, so this catches HTML error pages saved as `.mp3`.
fn looks_like_text(first_bytes: &[u8]) -> Option<bool> {
    first_bytes
        .iter()
        .find(|b| !b.is_ascii_whitespace())
        .map(|b| matches!(b, b'<' | b'{'))
}

fn existing_audio_is_usable(path: &Path) -> bool {
    let mut head = [0u8; 512];
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    match file.read(&mut head) {
        Ok(n) if n > 0 => looks_like_text(&head[..n]) != Some(true),
        _ => false,
    }
}

/// Download `url` into `dir/<basename>.<ext>`. Reuses an existing final file unless it is empty
/// or looks like a saved error page.
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
        .map_err(crate::http::safe_error)
        .with_context(|| format!("GET {}", crate::http::display_url(url)))?
        .error_for_status()
        .map_err(crate::http::safe_error)
        .with_context(|| format!("GET {}", crate::http::display_url(url)))?;
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    if let Some(ct) = ct.as_deref().filter(|ct| is_non_audio_type(ct)) {
        anyhow::bail!("server returned {ct}, not audio");
    }
    let ext = extension_for(resp.url().as_str(), ct.as_deref());
    let final_path = dir.join(format!("{basename}.{ext}"));
    if final_path.exists() && existing_audio_is_usable(&final_path) {
        return Ok(final_path);
    }
    let part_path = dir.join(format!("{basename}.{ext}.part"));
    let result = write_body(&mut resp, &part_path, show_progress)
        .and_then(|()| Ok(std::fs::rename(&part_path, &final_path)?));
    if result.is_err() {
        let _ = std::fs::remove_file(&part_path);
    }
    result.map(|()| final_path)
}

fn write_body(
    resp: &mut reqwest::blocking::Response,
    part_path: &Path,
    show_progress: bool,
) -> Result<()> {
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

    let mut file = std::fs::File::create(part_path)?;
    let mut buf = vec![0u8; 1 << 16];
    let mut written: u64 = 0;
    let mut sniffed = false;
    loop {
        // The nested I/O error may contain a signed redirect URL.
        let n = resp
            .read(&mut buf)
            .map_err(|_| anyhow::anyhow!("HTTP body transfer failed"))?;
        if n == 0 {
            break;
        }
        if !sniffed {
            match looks_like_text(&buf[..n]) {
                Some(true) => anyhow::bail!("response body is HTML/JSON, not audio"),
                Some(false) => sniffed = true,
                None => {}
            }
        }
        file.write_all(&buf[..n])?;
        written += n as u64;
        if let Some(b) = &bar {
            b.inc(n as u64);
        }
    }
    file.flush()?;
    if let Some(b) = bar {
        b.finish_and_clear();
    }
    anyhow::ensure!(written > 0, "empty response body");
    if let Some(expected) = total {
        anyhow::ensure!(
            written == expected,
            "incomplete download: {written} of {expected} bytes"
        );
    }
    Ok(())
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

    fn fetch(response: String, dir: &Path) -> Result<PathBuf> {
        use crate::test_support::serve;
        let (url, server) = serve(vec![(response, std::time::Duration::ZERO)]);
        let client = crate::http::client(5).unwrap();
        let result = download(&client, &format!("{url}/a.mp3"), dir, "ep", false);
        server.join().unwrap();
        result
    }

    fn no_part_files(dir: &Path) -> bool {
        std::fs::read_dir(dir)
            .unwrap()
            .all(|e| !e.unwrap().file_name().to_string_lossy().ends_with(".part"))
    }

    #[test]
    fn html_content_type_is_rejected() {
        use crate::test_support::response;
        let dir = tempfile::tempdir().unwrap();
        let err = fetch(
            response(200, "text/html; charset=utf-8", "<html>no</html>"),
            dir.path(),
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("not audio"));
        assert!(!dir.path().join("ep.mp3").exists());
    }

    #[test]
    fn markup_body_labelled_as_audio_is_rejected_and_cleaned_up() {
        use crate::test_support::response;
        let dir = tempfile::tempdir().unwrap();
        let body = "  \n<!DOCTYPE html><title>Expired link</title>";
        let err = fetch(response(200, "audio/mpeg", body), dir.path()).unwrap_err();
        assert!(format!("{err:#}").contains("not audio"));
        assert!(!dir.path().join("ep.mp3").exists());
        assert!(no_part_files(dir.path()));
    }

    #[test]
    fn plain_text_label_with_audio_body_is_accepted() {
        use crate::test_support::response;
        let dir = tempfile::tempdir().unwrap();
        let path = fetch(response(200, "text/plain", "ID3 audio bytes"), dir.path()).unwrap();
        assert_eq!(std::fs::read_to_string(path).unwrap(), "ID3 audio bytes");
    }

    // Invariant: a previously saved error page is replaced, while real audio already on disk
    // is reused without being rewritten.
    #[test]
    fn saved_error_page_is_replaced_but_real_audio_is_reused() {
        use crate::test_support::response;
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("ep.mp3");
        std::fs::write(&target, "<html>old error</html>").unwrap();
        fetch(response(200, "audio/mpeg", "ID3 fresh"), dir.path()).unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "ID3 fresh");
        fetch(response(200, "audio/mpeg", "ID3 different"), dir.path()).unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "ID3 fresh");
    }

    #[test]
    fn truncated_body_is_an_error_without_leftovers() {
        let dir = tempfile::tempdir().unwrap();
        let short = "HTTP/1.1 200 OK\r\nContent-Type: audio/mpeg\r\nContent-Length: 100\r\nConnection: close\r\n\r\nID3 only part";
        assert!(fetch(short.into(), dir.path()).is_err());
        assert!(!dir.path().join("ep.mp3").exists());
        assert!(no_part_files(dir.path()));
    }

    #[test]
    fn empty_body_is_an_error() {
        use crate::test_support::response;
        let dir = tempfile::tempdir().unwrap();
        assert!(fetch(response(200, "audio/mpeg", ""), dir.path()).is_err());
        assert!(!dir.path().join("ep.mp3").exists());
    }

    #[test]
    fn sniffing_ignores_leading_whitespace_and_accepts_binary() {
        assert_eq!(looks_like_text(b"  \n{\"error\":1}"), Some(true));
        assert_eq!(looks_like_text(b"ID3\x04"), Some(false));
        assert_eq!(looks_like_text(b"\xff\xfb\x90"), Some(false));
        assert_eq!(looks_like_text(b"   "), None);
    }

    #[test]
    fn filenames_distinguish_colliding_titles_and_feed_slugs() {
        let mut ep = crate::db::Episode {
            guid: "one".into(),
            feed_name: "a/b".into(),
            title: "A / B".into(),
            published: None,
            audio_url: String::new(),
            description: String::new(),
            audio_path: None,
            transcript_path: None,
            status: crate::db::Status::New,
            error: None,
        };
        let first = episode_filename(&ep);
        assert_eq!(first, episode_filename(&ep));
        ep.guid = "two".into();
        ep.title = "A: B".into();
        assert_ne!(first, episode_filename(&ep));
        ep.guid = "one".into();
        ep.feed_name = "a:b".into();
        assert_ne!(first, episode_filename(&ep));
    }
}
