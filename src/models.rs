//! Download ggml whisper models from the whisper.cpp Hugging Face repo.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub const KNOWN_MODELS: &[(&str, &str)] = &[
    ("tiny.en", "75 MB, fastest, rough"),
    ("base.en", "142 MB, fast, decent (default)"),
    (
        "small.en",
        "466 MB, ~3x slower than base, noticeably better",
    ),
    ("medium.en", "1.5 GB, slow on CPU, very good"),
    (
        "large-v3-turbo",
        "1.6 GB, multilingual, good speed/quality on CPU",
    ),
    ("large-v3", "3.1 GB, best quality, very slow on CPU"),
];

pub fn model_url(name: &str) -> String {
    format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin")
}

/// The configured model's path, downloading it first if it is a named model that isn't on disk.
pub fn resolve_model(settings: &crate::config::Settings) -> Result<PathBuf> {
    let path = settings.model_path();
    if path.exists() {
        return Ok(path);
    }
    if settings.whisper.model_path.is_some() {
        anyhow::bail!(
            "configured whisper.model_path does not exist: {}",
            path.display()
        );
    }
    ensure_model(
        &crate::http::client(settings.download_timeout_secs)?,
        &settings.whisper.model,
        &settings.model_dir(),
    )
}

pub fn ensure_model(
    client: &reqwest::blocking::Client,
    name: &str,
    model_dir: &Path,
) -> Result<PathBuf> {
    let path = model_dir.join(format!("ggml-{name}.bin"));
    if path.exists() {
        return Ok(path);
    }
    eprintln!("Model {name} not found; downloading from Hugging Face...");
    let url = model_url(name);
    std::fs::create_dir_all(model_dir)?;
    let downloaded =
        crate::download::download(client, &url, model_dir, &format!("ggml-{name}"), true)
            .with_context(|| format!("downloading model {name}"))?;
    // download() names by extension it infers; force our canonical name.
    if downloaded != path {
        std::fs::rename(&downloaded, &path)?;
    }
    Ok(path)
}
