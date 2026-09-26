mod audio;
mod config;
mod db;
mod download;
mod feeds;
mod http;
mod models;
mod pipeline;
#[cfg(test)]
mod test_support;
mod transcribe;
mod vocab;

use anyhow::{bail, Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use config::{Feed, Settings};
use db::{Db, Status};
use pipeline::BackfillFilter;

#[derive(Parser)]
#[command(
    name = "podcast",
    version,
    about = "Subscribe to podcast feeds, download episodes, transcribe them locally."
)]
struct Cli {
    /// Path to config file.
    #[arg(short, long, global = true, default_value = config::DEFAULT_CONFIG)]
    config: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write a default config.yaml (does nothing if it exists).
    Init,
    /// Subscribe to a feed URL. Fetches it, queues the newest episodes.
    Add {
        url: String,
        /// Short name used in paths and other commands (default: derived from feed title).
        #[arg(short, long)]
        name: Option<String>,
    },
    /// Unsubscribe from a feed and forget its episodes (audio/transcripts on disk are kept).
    Remove { name: String },
    /// List subscribed feeds.
    Feeds,
    /// Check feeds for new episodes and queue them.
    Refresh {
        /// Only this feed.
        #[arg(short, long)]
        feed: Option<String>,
    },
    /// Queue historical (older) episodes of a feed. Follows feed pagination to find everything.
    Backfill {
        feed: String,
        /// Queue every episode not yet processed.
        #[arg(long)]
        all: bool,
        /// Queue only the newest N un-queued episodes.
        #[arg(short, long)]
        limit: Option<usize>,
        /// Only episodes published on/after this date (YYYY-MM-DD).
        #[arg(long)]
        since: Option<String>,
        /// Only episodes published on/before this date (YYYY-MM-DD).
        #[arg(long)]
        until: Option<String>,
        /// Only episodes whose title contains this text (case-insensitive).
        #[arg(long)]
        title: Option<String>,
        /// Show what would be queued without changing anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Download all queued episodes.
    Download {
        #[arg(short, long)]
        feed: Option<String>,
        #[arg(short, long)]
        limit: Option<usize>,
    },
    /// Transcribe all downloaded episodes.
    Transcribe {
        #[arg(short, long)]
        feed: Option<String>,
        #[arg(short, long)]
        limit: Option<usize>,
    },
    /// refresh + download + transcribe in one go. Suitable for cron/launchd.
    Run {
        #[arg(short, long)]
        feed: Option<String>,
        /// Cap downloads/transcriptions this run.
        #[arg(short, long)]
        limit: Option<usize>,
    },
    /// Collect a bounded research sample from already discovered episodes.
    Collect {
        feed: String,
        #[arg(long)]
        title: String,
        #[arg(long, default_value_t = 5)]
        limit: usize,
    },
    /// Pipeline counts per status.
    Status {
        #[arg(short, long)]
        feed: Option<String>,
    },
    /// List episodes.
    Episodes {
        #[arg(short, long)]
        feed: Option<String>,
        /// new | downloaded | transcribed | failed | skipped
        #[arg(short, long)]
        status: Option<String>,
    },
    /// Re-queue failed episodes.
    Retry {
        #[arg(short, long)]
        feed: Option<String>,
    },
    /// Manage whisper models.
    Model {
        #[command(subcommand)]
        cmd: ModelCmd,
    },
    /// Manage a feed's name vocabulary (fixes mangled names in transcripts).
    Vocab {
        #[command(subcommand)]
        cmd: VocabCmd,
    },
    /// Re-apply a feed's vocabulary to transcripts already on disk.
    Correct { feed: String },
    /// Transcribe a local audio file directly (no feed involved).
    File {
        path: PathBuf,
        /// Output directory (default: alongside the input file).
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum VocabCmd {
    /// Show the vocabulary file and its size.
    List { feed: String },
    /// Add names, or aliases in the form "wrong words => Right Name".
    Add { feed: String, entries: Vec<String> },
    /// Append every non-empty line of a text file.
    Import { feed: String, path: PathBuf },
    /// Pull every current NBA player and head coach name from ESPN's public roster API.
    Nba { feed: String },
    /// Try the vocabulary against a piece of text and show what would change.
    Test { feed: String, text: String },
}

#[derive(Subcommand)]
enum ModelCmd {
    /// Show known models and which are downloaded.
    List,
    /// Download a model (e.g. base.en, small.en, large-v3-turbo).
    Download { name: String },
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let mut settings = Settings::load(&cli.config)?;

    match cli.cmd {
        Cmd::Init => {
            if cli.config.exists() {
                println!("{} already exists", cli.config.display());
            } else {
                Settings::default().save(&cli.config)?;
                println!("wrote {}", cli.config.display());
            }
        }

        Cmd::Add { url, name } => {
            let client = http::client(settings.feed_timeout_secs)?;
            let parsed = feeds::fetch(&client, &url)?;
            let name = name.unwrap_or_else(|| download::slug(&parsed.title, 40).to_lowercase());
            if settings.feed(&name).is_some() {
                bail!("a feed named '{name}' already exists");
            }
            let feed = Feed {
                name: name.clone(),
                url,
                enabled: true,
                prompt: None,
            };
            let db = Db::open(&settings.db_path())?;
            let r = pipeline::ingest(&db, &feed, &parsed, settings.max_new_per_feed)?;
            settings.feeds.push(feed);
            settings.save(&cli.config)?;
            println!(
                "added '{name}' ({}): {} episodes in feed, {} queued, {} older recorded as skipped",
                parsed.title, r.total_in_feed, r.queued, r.skipped
            );
            if r.skipped > 0 {
                println!("  to get history: podcast backfill {name} --all   (or --limit N / --since YYYY-MM-DD)");
            }
        }

        Cmd::Remove { name } => {
            let before = settings.feeds.len();
            settings.feeds.retain(|f| f.name != name);
            if settings.feeds.len() == before {
                bail!("no feed named '{name}'");
            }
            settings.save(&cli.config)?;
            let n = Db::open(&settings.db_path())?.delete_feed(&name)?;
            println!("removed '{name}' and {n} episode records");
        }

        Cmd::Feeds => {
            if settings.feeds.is_empty() {
                println!("no feeds. add one with: podcast add <rss-url>");
            }
            let db = Db::open(&settings.db_path())?;
            for f in &settings.feeds {
                let counts = db.counts(Some(&f.name))?;
                let summary: Vec<String> = counts
                    .iter()
                    .filter(|(_, n)| *n > 0)
                    .map(|(s, n)| format!("{n} {s}"))
                    .collect();
                println!(
                    "{:<24} {}{}  [{}]",
                    f.name,
                    if f.enabled { "" } else { "(disabled) " },
                    http::display_url(&f.url),
                    summary.join(", ")
                );
            }
        }

        Cmd::Refresh { feed } => {
            require_feed(&settings, feed.as_deref())?;
            let db = Db::open(&settings.db_path())?;
            let batch = pipeline::refresh(&settings, &db, feed.as_deref())?;
            for r in &batch.value {
                println!(
                    "{:<24} {} new{}",
                    r.feed,
                    r.queued,
                    if r.skipped > 0 {
                        format!(", {} older skipped", r.skipped)
                    } else {
                        String::new()
                    }
                );
            }
            batch.check("refresh")?;
        }

        Cmd::Backfill {
            feed,
            all,
            limit,
            since,
            until,
            title,
            dry_run,
        } => {
            let f = settings
                .feed(&feed)
                .with_context(|| format!("no feed named '{feed}'"))?
                .clone();
            let filter = BackfillFilter {
                all,
                limit,
                since: since.map(|s| parse_date(&s, false)).transpose()?,
                until: until.map(|s| parse_date(&s, true)).transpose()?,
                title_contains: title,
            };
            if filter.is_empty() {
                bail!("choose what to backfill: --all, --limit N, --since DATE, --until DATE, or --title TEXT (add --dry-run to preview)");
            }
            let db = Db::open(&settings.db_path())?;
            eprintln!("fetching full history for '{}'...", f.name);
            let selected = pipeline::backfill(&settings, &db, &f, &filter, dry_run)?;
            for e in &selected {
                println!("{}  {}", fmt_date(e.published), e.title);
            }
            println!(
                "{} {} episode(s){}",
                if dry_run { "would queue" } else { "queued" },
                selected.len(),
                if dry_run {
                    ""
                } else {
                    ". next: podcast download && podcast transcribe   (or: podcast run)"
                }
            );
        }

        Cmd::Download { feed, limit } => {
            require_feed(&settings, feed.as_deref())?;
            let db = Db::open(&settings.db_path())?;
            let n = pipeline::download_pending(&settings, &db, feed.as_deref(), limit)?;
            println!("downloaded {}, failed {}", n.value, n.failed);
            n.check("download")?;
        }

        Cmd::Transcribe { feed, limit } => {
            require_feed(&settings, feed.as_deref())?;
            let db = Db::open(&settings.db_path())?;
            let n = pipeline::transcribe_pending(&settings, &db, feed.as_deref(), limit)?;
            println!("transcribed {}, failed {}", n.value, n.failed);
            n.check("transcribe")?;
        }

        Cmd::Run { feed, limit } => {
            require_feed(&settings, feed.as_deref())?;
            let db = Db::open(&settings.db_path())?;
            // Evaluate every stage even when an earlier stage has failed items.
            let refresh = pipeline::refresh(&settings, &db, feed.as_deref()).and_then(|batch| {
                let new: usize = batch.value.iter().map(|r| r.queued).sum();
                eprintln!("refresh: {new} new, {} failed", batch.failed);
                batch.check("refresh")
            });
            let download = pipeline::download_pending(&settings, &db, feed.as_deref(), limit)
                .and_then(|batch| {
                    eprintln!(
                        "download: {} completed, {} failed",
                        batch.value, batch.failed
                    );
                    batch.check("download")
                });
            let transcribe = pipeline::transcribe_pending(&settings, &db, feed.as_deref(), limit)
                .and_then(|batch| {
                    eprintln!(
                        "transcribe: {} completed, {} failed",
                        batch.value, batch.failed
                    );
                    batch.check("transcribe")
                });
            let failures: Vec<String> = [refresh, download, transcribe]
                .into_iter()
                .filter_map(|r| r.err().map(|e| format!("{e:#}")))
                .collect();
            if !failures.is_empty() {
                bail!("run finished with failures: {}", failures.join("; "));
            }
            println!("run complete");
        }

        Cmd::Collect { feed, title, limit } => {
            require_feed(&settings, Some(&feed))?;
            let db = Db::open(&settings.db_path())?;
            pipeline::collect(&settings, &db, &feed, &title, limit)?;
        }

        Cmd::Status { feed } => {
            require_feed(&settings, feed.as_deref())?;
            let db = Db::open(&settings.db_path())?;
            for (s, n) in db.counts(feed.as_deref())? {
                println!("{:<12} {n}", s.as_str());
            }
        }

        Cmd::Episodes { feed, status } => {
            require_feed(&settings, feed.as_deref())?;
            let status = status.map(|s| s.parse::<Status>()).transpose()?;
            let db = Db::open(&settings.db_path())?;
            for e in db.list(status, feed.as_deref())? {
                let loc = e
                    .transcript_path
                    .as_ref()
                    .or(e.audio_path.as_ref())
                    .map(|p| p.display().to_string())
                    .or(e.error.clone())
                    .unwrap_or_default();
                println!(
                    "{:<11} {}  {:<20} {}  {}",
                    e.status,
                    fmt_date(e.published),
                    e.feed_name,
                    e.title,
                    loc
                );
            }
        }

        Cmd::Retry { feed } => {
            let db = Db::open(&settings.db_path())?;
            let failed = db.list(Some(Status::Failed), feed.as_deref())?;
            for e in &failed {
                // If audio is already on disk, go straight back to transcription.
                let back_to = match &e.audio_path {
                    Some(p) if p.exists() => Status::Downloaded,
                    _ => Status::New,
                };
                db.set_status(&e.feed_name, &e.guid, back_to, None)?;
            }
            println!("re-queued {}", failed.len());
        }

        Cmd::Model { cmd } => match cmd {
            ModelCmd::List => {
                for (name, desc) in models::KNOWN_MODELS {
                    let path = settings.model_dir().join(format!("ggml-{name}.bin"));
                    let mark = if path.exists() { "*" } else { " " };
                    let cur = if *name == settings.whisper.model {
                        " (configured)"
                    } else {
                        ""
                    };
                    println!("{mark} {name:<16} {desc}{cur}");
                }
                println!("\n* = downloaded to {}", settings.model_dir().display());
            }
            ModelCmd::Download { name } => {
                let p = models::ensure_model(
                    &http::client(settings.download_timeout_secs)?,
                    &name,
                    &settings.model_dir(),
                )?;
                println!("{}", p.display());
            }
        },

        Cmd::Vocab { cmd } => vocab_cmd(&settings, cmd)?,

        Cmd::Correct { feed } => {
            require_feed(&settings, Some(&feed))?;
            let db = Db::open(&settings.db_path())?;
            let (done, changed) = pipeline::correct_existing(&settings, &db, &feed)?;
            println!("re-corrected {done} transcript(s), {changed} changed");
        }

        Cmd::File { path, out } => {
            if !audio::ffmpeg_available() {
                bail!("ffmpeg not found on PATH (brew install ffmpeg)");
            }
            let model_path = models::resolve_model(&settings)?;
            let engine = transcribe::Engine::load(&model_path, &settings.whisper)?;
            let pcm = audio::decode_to_pcm(&path)?;
            let duration = audio::duration_secs(&pcm);
            eprintln!(
                "audio: {:.1} min, model {}",
                duration / 60.0,
                engine.model_name
            );
            let started = std::time::Instant::now();
            let (segments, language) = engine.transcribe(&pcm, None, |_| {})?;
            eprintln!("done in {:.0}s", started.elapsed().as_secs_f64());
            let stem = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or("transcript".into());
            let t = transcribe::Transcript {
                feed: String::new(),
                title: stem.clone(),
                guid: path.display().to_string(),
                published: None,
                audio_url: String::new(),
                model: engine.model_name.clone(),
                language,
                duration_secs: duration,
                downloaded_at: None,
                transcribed_at: Some(Utc::now().to_rfc3339()),
                producer_version: Some(env!("CARGO_PKG_VERSION").into()),
                prompt: None,
                corrections: vec![],
                segments,
            };
            let dir = out.unwrap_or_else(|| {
                path.parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."))
            });
            let p = t.write(&dir, &stem, &settings.formats)?;
            println!("{}", p.display());
        }
    }
    Ok(())
}

fn vocab_cmd(settings: &Settings, cmd: VocabCmd) -> Result<()> {
    let feed = match &cmd {
        VocabCmd::List { feed }
        | VocabCmd::Add { feed, .. }
        | VocabCmd::Import { feed, .. }
        | VocabCmd::Nba { feed }
        | VocabCmd::Test { feed, .. } => feed.clone(),
    };
    require_feed(settings, Some(&feed))?;
    let path = settings.vocab_path(&feed);
    let append = |lines: &[String]| -> Result<usize> {
        let existing = vocab::Vocab::load(&path)?;
        let mut existing_keys: std::collections::HashSet<String> =
            existing.terms.iter().map(|t| t.key.clone()).collect();
        let mut alias_keys: std::collections::HashSet<String> = existing
            .aliases
            .iter()
            .map(|a| a.from_words.join(" "))
            .collect();
        let mut out = vec![];
        for l in lines {
            let l = l.trim();
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            if let Some((from, _)) = l.split_once("=>") {
                let k: Vec<String> = from.split_whitespace().map(vocab::normalize_word).collect();
                if alias_keys.insert(k.join(" ")) {
                    out.push(l.to_string());
                }
            } else {
                let key: String = l.split_whitespace().map(vocab::normalize_word).collect();
                if !key.is_empty() && existing_keys.insert(key) {
                    out.push(l.to_string());
                }
            }
        }
        if !out.is_empty() {
            std::fs::create_dir_all(path.parent().unwrap())?;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)?;
            use std::io::Write;
            for l in &out {
                writeln!(f, "{l}")?;
            }
        }
        Ok(out.len())
    };
    match cmd {
        VocabCmd::List { .. } => {
            let v = vocab::Vocab::load(&path)?;
            println!(
                "{}: {} names, {} aliases",
                path.display(),
                v.terms.len(),
                v.aliases.len()
            );
            if path.exists() {
                print!("{}", std::fs::read_to_string(&path)?);
            }
        }
        VocabCmd::Add { entries, .. } => {
            let n = append(&entries)?;
            println!("added {n} to {}", path.display());
        }
        VocabCmd::Import { path: src, .. } => {
            let lines: Vec<String> = std::fs::read_to_string(&src)?
                .lines()
                .map(str::to_string)
                .collect();
            let n = append(&lines)?;
            println!("added {n} to {}", path.display());
        }
        VocabCmd::Nba { .. } => {
            let names = vocab::nba::fetch_names()?;
            let n = append(&names)?;
            println!(
                "fetched {} names, added {n} new to {}",
                names.len(),
                path.display()
            );
        }
        VocabCmd::Test { text, .. } => {
            let v = vocab::Vocab::load(&path)?;
            let (out, corr) = v.correct(&text, settings.correction_threshold);
            println!("{out}");
            for c in corr {
                println!("  {:?} -> {:?} (x{})", c.from, c.to, c.count);
            }
        }
    }
    Ok(())
}

fn require_feed(settings: &Settings, name: Option<&str>) -> Result<()> {
    if let Some(n) = name {
        if settings.feed(n).is_none() {
            bail!("no feed named '{n}'. run `podcast feeds` to list them");
        }
    }
    Ok(())
}

fn parse_date(s: &str, end_of_day: bool) -> Result<DateTime<Utc>> {
    let d = NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .with_context(|| format!("invalid date '{s}', expected YYYY-MM-DD"))?;
    let t = if end_of_day {
        d.and_hms_opt(23, 59, 59).unwrap()
    } else {
        d.and_hms_opt(0, 0, 0).unwrap()
    };
    Ok(DateTime::<Utc>::from_naive_utc_and_offset(t, Utc))
}

fn fmt_date(d: Option<DateTime<Utc>>) -> String {
    d.map(|d| d.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| "----------".into())
}
