//! The refresh -> download -> transcribe pipeline, plus historical backfill.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use std::time::Instant;

use crate::config::{Feed, Settings};
use crate::db::{Db, Episode, Status};
use crate::download;
use crate::feeds::{self, FeedEpisode, ParsedFeed};
use crate::transcribe::{Engine, Transcript};
use crate::vocab::{self, Vocab};

/// Re-apply a feed's vocabulary to its existing transcripts (needs the .json format).
pub fn correct_existing(settings: &Settings, db: &Db, feed_name: &str) -> Result<(usize, usize)> {
    let vocab = Vocab::load(&settings.vocab_path(feed_name))?;
    if vocab.is_empty() {
        anyhow::bail!(
            "vocabulary for '{feed_name}' is empty: {}",
            settings.vocab_path(feed_name).display()
        );
    }
    let mut done = 0;
    let mut changed = 0;
    for ep in db.list(Some(Status::Transcribed), Some(feed_name))? {
        let Some(primary) = &ep.transcript_path else {
            continue;
        };
        let json = primary.with_extension("json");
        if !json.exists() {
            eprintln!("  skipping {} (no .json transcript)", ep.title);
            continue;
        }
        let mut t = Transcript::read_json(&json)?;
        let before: usize = t.corrections.iter().map(|c| c.count).sum();
        t.apply_vocab(&vocab, settings.correction_threshold);
        let after: usize = t.corrections.iter().map(|c| c.count).sum();
        let dir = json.parent().context("transcript has no parent dir")?;
        let stem = json
            .file_stem()
            .context("bad transcript filename")?
            .to_string_lossy()
            .to_string();
        t.write(dir, &stem, &settings.formats)?;
        done += 1;
        if after != before {
            changed += 1;
        }
    }
    Ok((done, changed))
}

pub struct RefreshReport {
    pub feed: String,
    pub queued: usize,
    pub skipped: usize,
    pub total_in_feed: usize,
}

fn to_episode(feed: &Feed, e: &FeedEpisode, status: Status) -> Episode {
    Episode {
        guid: e.guid.clone(),
        feed_name: feed.name.clone(),
        title: e.title.clone(),
        published: e.published,
        audio_url: e.audio_url.clone(),
        description: e.description.clone(),
        audio_path: None,
        transcript_path: None,
        status,
        error: None,
    }
}

/// Record a parsed feed's episodes. On a feed's first refresh only the newest `max_new` are
/// queued; older ones are recorded as skipped. On later refreshes every unseen episode is queued
/// (they are genuinely new). Pure over the DB so it's testable without the network.
pub fn ingest(db: &Db, feed: &Feed, parsed: &ParsedFeed, max_new: usize) -> Result<RefreshReport> {
    let first_time = db.feed_count(&feed.name)? == 0;
    let mut queued = 0;
    let mut skipped = 0;
    // parsed.episodes is newest-first.
    for (i, e) in parsed.episodes.iter().enumerate() {
        let status = if first_time && i >= max_new {
            Status::Skipped
        } else {
            Status::New
        };
        if db.insert(&to_episode(feed, e, status))? {
            match status {
                Status::New => queued += 1,
                _ => skipped += 1,
            }
        }
    }
    Ok(RefreshReport {
        feed: feed.name.clone(),
        queued,
        skipped,
        total_in_feed: parsed.episodes.len(),
    })
}

pub fn refresh(settings: &Settings, db: &Db, only: Option<&str>) -> Result<Vec<RefreshReport>> {
    let client = feeds::http_client()?;
    let mut reports = vec![];
    for feed in settings.feeds.iter().filter(|f| f.enabled) {
        if let Some(name) = only {
            if feed.name != name {
                continue;
            }
        }
        match feeds::fetch(&client, &feed.url) {
            Ok(parsed) => reports.push(ingest(db, feed, &parsed, settings.max_new_per_feed)?),
            Err(e) => eprintln!("warning: feed '{}' failed: {e:#}", feed.name),
        }
    }
    Ok(reports)
}

/// Which historical episodes to queue.
#[derive(Debug, Default, Clone)]
pub struct BackfillFilter {
    pub all: bool,
    /// Newest N of the currently-skipped episodes.
    pub limit: Option<usize>,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
    /// Case-insensitive substring on the title.
    pub title_contains: Option<String>,
}

impl BackfillFilter {
    pub fn is_empty(&self) -> bool {
        !self.all
            && self.limit.is_none()
            && self.since.is_none()
            && self.until.is_none()
            && self.title_contains.is_none()
    }

    fn matches(&self, e: &Episode) -> bool {
        if let Some(s) = self.since {
            match e.published {
                Some(p) if p >= s => {}
                _ => return false,
            }
        }
        if let Some(u) = self.until {
            match e.published {
                Some(p) if p <= u => {}
                _ => return false,
            }
        }
        if let Some(t) = &self.title_contains {
            if !e.title.to_lowercase().contains(&t.to_lowercase()) {
                return false;
            }
        }
        true
    }
}

/// Select skipped episodes matching the filter (newest first, then apply limit).
pub fn select_backfill(db: &Db, feed_name: &str, filter: &BackfillFilter) -> Result<Vec<Episode>> {
    let mut eps = db.list(Some(Status::Skipped), Some(feed_name))?;
    eps.reverse(); // list() is oldest-first
    eps.retain(|e| filter.matches(e));
    if let Some(n) = filter.limit {
        eps.truncate(n);
    }
    Ok(eps)
}

/// Discover the feed's full history (following pagination), record unknown episodes as skipped,
/// then promote matching skipped episodes to `new`. Returns the promoted episodes.
pub fn backfill(
    settings: &Settings,
    db: &Db,
    feed: &Feed,
    filter: &BackfillFilter,
    dry_run: bool,
) -> Result<Vec<Episode>> {
    let client = feeds::http_client()?;
    let parsed = feeds::fetch_all(&client, &feed.url, settings.max_feed_pages, |page, n| {
        if page > 1 {
            eprintln!("  page {page}: {n} more episodes");
        }
    })?;
    let mut discovered = 0;
    for e in &parsed.episodes {
        if db.insert(&to_episode(feed, e, Status::Skipped))? {
            discovered += 1;
        }
    }
    if discovered > 0 {
        eprintln!("  discovered {discovered} previously unseen episodes");
    }
    let selected = select_backfill(db, &feed.name, filter)?;
    if !dry_run {
        for e in &selected {
            db.set_status(&e.guid, Status::New, None)?;
        }
    }
    Ok(selected)
}

pub fn download_pending(
    settings: &Settings,
    db: &Db,
    only: Option<&str>,
    limit: Option<usize>,
) -> Result<usize> {
    let client = feeds::http_client()?;
    let mut eps = db.list(Some(Status::New), only)?;
    if let Some(n) = limit {
        eps.truncate(n);
    }
    let mut ok = 0;
    for (i, ep) in eps.iter().enumerate() {
        eprintln!(
            "[{}/{}] downloading {} / {}",
            i + 1,
            eps.len(),
            ep.feed_name,
            ep.title
        );
        let dir = settings.audio_dir().join(download::slug(&ep.feed_name, 60));
        let basename = download::episode_basename(&ep.title, ep.published);
        match download::download(&client, &ep.audio_url, &dir, &basename, true) {
            Ok(path) => {
                db.set_audio_path(&ep.guid, &path)?;
                db.set_status(&ep.guid, Status::Downloaded, None)?;
                ok += 1;
            }
            Err(e) => {
                eprintln!("  failed: {e:#}");
                db.set_status(&ep.guid, Status::Failed, Some(&format!("download: {e:#}")))?;
            }
        }
    }
    Ok(ok)
}

pub fn transcribe_pending(
    settings: &Settings,
    db: &Db,
    only: Option<&str>,
    limit: Option<usize>,
) -> Result<usize> {
    let mut eps = db.list(Some(Status::Downloaded), only)?;
    if let Some(n) = limit {
        eps.truncate(n);
    }
    if eps.is_empty() {
        return Ok(0);
    }
    if !crate::audio::ffmpeg_available() {
        anyhow::bail!("ffmpeg not found on PATH (brew install ffmpeg)");
    }
    let model_path = crate::models::resolve_model(settings)?;
    eprintln!(
        "loading model {} ({} threads)",
        model_path.display(),
        settings.whisper.threads
    );
    let engine = Engine::load(&model_path, &settings.whisper)?;
    let mut vocabs: std::collections::HashMap<String, Vocab> = Default::default();

    let mut ok = 0;
    for (i, ep) in eps.iter().enumerate() {
        eprintln!(
            "[{}/{}] transcribing {} / {}",
            i + 1,
            eps.len(),
            ep.feed_name,
            ep.title
        );
        let vocab = match vocabs.get(&ep.feed_name) {
            Some(v) => v.clone(),
            None => {
                let v = Vocab::load(&settings.vocab_path(&ep.feed_name))?;
                vocabs.insert(ep.feed_name.clone(), v.clone());
                v
            }
        };
        match transcribe_one(settings, db, &engine, ep, &vocab) {
            Ok(path) => {
                eprintln!("  -> {}", path.display());
                ok += 1;
            }
            Err(e) => {
                eprintln!("  failed: {e:#}");
                db.set_status(
                    &ep.guid,
                    Status::Failed,
                    Some(&format!("transcribe: {e:#}")),
                )?;
            }
        }
    }
    Ok(ok)
}

fn transcribe_one(
    settings: &Settings,
    db: &Db,
    engine: &Engine,
    ep: &Episode,
    vocab: &Vocab,
) -> Result<std::path::PathBuf> {
    let audio_path = ep
        .audio_path
        .as_ref()
        .context("episode has no audio_path")?;
    if !audio_path.exists() {
        anyhow::bail!("audio file missing: {}", audio_path.display());
    }
    let pcm = crate::audio::decode_to_pcm(audio_path)?;
    let duration = crate::audio::duration_secs(&pcm);
    eprintln!("  audio: {:.1} min", duration / 60.0);

    let bar = indicatif::ProgressBar::new(100);
    bar.set_style(
        indicatif::ProgressStyle::with_template(
            "  {bar:30} {percent}% elapsed {elapsed} eta {eta}",
        )
        .unwrap(),
    );
    let bar2 = bar.clone();
    let started = Instant::now();
    let feed_prompt = settings
        .feed(&ep.feed_name)
        .and_then(|f| f.prompt.as_deref());
    let prompt = vocab::build_prompt(
        feed_prompt,
        settings.whisper.initial_prompt.as_deref(),
        &ep.title,
        &ep.description,
        vocab,
    );
    let (segments, language) = engine.transcribe(&pcm, Some(&prompt), move |p: i32| {
        bar2.set_position(p.clamp(0, 100) as u64)
    })?;
    bar.finish_and_clear();
    let secs = started.elapsed().as_secs_f64();
    eprintln!(
        "  done in {:.0}s ({:.1}x realtime)",
        secs,
        duration / secs.max(0.001)
    );

    let mut transcript = Transcript {
        feed: ep.feed_name.clone(),
        title: ep.title.clone(),
        guid: ep.guid.clone(),
        published: ep.published.map(|d| d.to_rfc3339()),
        audio_url: ep.audio_url.clone(),
        model: engine.model_name.clone(),
        language,
        duration_secs: duration,
        prompt: Some(prompt),
        corrections: vec![],
        segments,
    };
    if !vocab.is_empty() {
        transcript.apply_vocab(vocab, settings.correction_threshold);
        let n: usize = transcript.corrections.iter().map(|c| c.count).sum();
        if n > 0 {
            eprintln!("  vocab: {n} correction(s)");
        }
    }
    let dir = settings
        .transcript_dir()
        .join(download::slug(&ep.feed_name, 60));
    let basename = download::episode_basename(&ep.title, ep.published);
    let path = transcript.write(&dir, &basename, &settings.formats)?;
    db.set_transcript_path(&ep.guid, &path)?;
    db.set_status(&ep.guid, Status::Transcribed, None)?;
    if settings.delete_audio_after_transcribe {
        let _ = std::fs::remove_file(audio_path);
        db.clear_audio_path(&ep.guid)?;
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn feed() -> Feed {
        Feed {
            name: "cast".into(),
            url: "https://x/rss".into(),
            enabled: true,
            prompt: None,
        }
    }

    fn parsed(n: usize) -> ParsedFeed {
        // newest first: ep-n ... ep-1
        let episodes = (1..=n)
            .rev()
            .map(|i| FeedEpisode {
                guid: format!("ep-{i}"),
                title: format!("Episode {i}"),
                published: Some(Utc.with_ymd_and_hms(2024, 1, i as u32, 0, 0, 0).unwrap()),
                audio_url: format!("https://x/{i}.mp3"),
                description: String::new(),
            })
            .collect();
        ParsedFeed {
            title: "Cast".into(),
            episodes,
            next_page: None,
        }
    }

    #[test]
    fn first_ingest_queues_only_newest_then_later_ingests_queue_everything_new() {
        let db = Db::open_memory().unwrap();
        let r = ingest(&db, &feed(), &parsed(10), 3).unwrap();
        assert_eq!((r.queued, r.skipped, r.total_in_feed), (3, 7, 10));
        let new = db.list(Some(Status::New), None).unwrap();
        let guids: Vec<_> = new.iter().map(|e| e.guid.as_str()).collect();
        assert_eq!(guids, ["ep-8", "ep-9", "ep-10"]);

        // Same feed again: nothing new.
        let r = ingest(&db, &feed(), &parsed(10), 3).unwrap();
        assert_eq!((r.queued, r.skipped), (0, 0));

        // Feed grows by 5 -> all 5 queued even though max_new is 3.
        let r = ingest(&db, &feed(), &parsed(15), 3).unwrap();
        assert_eq!((r.queued, r.skipped), (5, 0));
    }

    #[test]
    fn backfill_selection_filters() {
        let db = Db::open_memory().unwrap();
        ingest(&db, &feed(), &parsed(10), 2).unwrap(); // ep-1..8 skipped

        let all = select_backfill(
            &db,
            "cast",
            &BackfillFilter {
                all: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(all.len(), 8);
        assert_eq!(all[0].guid, "ep-8"); // newest first

        let lim = select_backfill(
            &db,
            "cast",
            &BackfillFilter {
                limit: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        let g: Vec<_> = lim.iter().map(|e| e.guid.as_str()).collect();
        assert_eq!(g, ["ep-8", "ep-7"]);

        let since = Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0).unwrap();
        let until = Utc.with_ymd_and_hms(2024, 1, 6, 23, 0, 0).unwrap();
        let rng = select_backfill(
            &db,
            "cast",
            &BackfillFilter {
                since: Some(since),
                until: Some(until),
                ..Default::default()
            },
        )
        .unwrap();
        let g: Vec<_> = rng.iter().map(|e| e.guid.as_str()).collect();
        assert_eq!(g, ["ep-6", "ep-5"]);

        let t = select_backfill(
            &db,
            "cast",
            &BackfillFilter {
                title_contains: Some("episode 3".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(t.len(), 1);

        // Wrong feed name -> nothing.
        assert!(select_backfill(
            &db,
            "nope",
            &BackfillFilter {
                all: true,
                ..Default::default()
            }
        )
        .unwrap()
        .is_empty());
        assert!(BackfillFilter::default().is_empty());
    }
}
