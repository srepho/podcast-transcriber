//! Decode any audio file to 16 kHz mono f32 PCM via ffmpeg (piped, no temp files).

use anyhow::{bail, Context, Result};
use std::path::Path;
use std::process::{Command, Stdio};

pub const SAMPLE_RATE: u32 = 16_000;

pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

pub fn decode_to_pcm(path: &Path) -> Result<Vec<f32>> {
    let output = Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args([
            "-vn",
            "-ac",
            "1",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-f",
            "f32le",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("running ffmpeg (is it installed and on PATH?)")?;
    if !output.status.success() {
        bail!(
            "ffmpeg failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let bytes = output.stdout;
    if bytes.len() < 4 {
        bail!("ffmpeg produced no audio for {}", path.display());
    }
    let samples = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    Ok(samples)
}

pub fn duration_secs(samples: &[f32]) -> f64 {
    samples.len() as f64 / SAMPLE_RATE as f64
}
