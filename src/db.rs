//! SQLite state: every episode we've seen and where it is in the pipeline.

use anyhow::Result;
use chrono::{DateTime, Utc};
#[cfg(test)]
use rusqlite::OptionalExtension;
use rusqlite::{params, Connection, Row};
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Known but deliberately not queued (older history). `backfill` promotes these to New.
    Skipped,
    /// Queued for download.
    New,
    /// Audio on disk, waiting for transcription.
    Downloaded,
    /// Transcript written.
    Transcribed,
    /// Something went wrong; `error` holds the message. `retry` promotes back to New.
    Failed,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Skipped => "skipped",
            Status::New => "new",
            Status::Downloaded => "downloaded",
            Status::Transcribed => "transcribed",
            Status::Failed => "failed",
        }
    }
    pub const ALL: [Status; 5] = [
        Status::New,
        Status::Downloaded,
        Status::Transcribed,
        Status::Failed,
        Status::Skipped,
    ];
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Status {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "skipped" => Status::Skipped,
            "new" => Status::New,
            "downloaded" => Status::Downloaded,
            "transcribed" => Status::Transcribed,
            "failed" => Status::Failed,
            other => anyhow::bail!("unknown status '{other}'"),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Episode {
    pub guid: String,
    pub feed_name: String,
    pub title: String,
    pub published: Option<DateTime<Utc>>,
    pub audio_url: String,
    pub description: String,
    pub audio_path: Option<PathBuf>,
    pub transcript_path: Option<PathBuf>,
    pub status: Status,
    pub error: Option<String>,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS episodes (
    guid            TEXT PRIMARY KEY,
    feed_name       TEXT NOT NULL,
    title           TEXT NOT NULL,
    published       TEXT,
    audio_url       TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    audio_path      TEXT,
    transcript_path TEXT,
    status          TEXT NOT NULL,
    error           TEXT,
    added_at        TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status);
CREATE INDEX IF NOT EXISTS idx_episodes_feed ON episodes(feed_name);
";

pub struct Db {
    conn: Connection,
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }

    #[cfg(test)]
    pub fn open_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }

    /// Insert if unseen. Returns true if newly inserted.
    pub fn insert(&self, ep: &Episode) -> Result<bool> {
        let now = Utc::now().to_rfc3339();
        let n = self.conn.execute(
            "INSERT OR IGNORE INTO episodes
             (guid, feed_name, title, published, audio_url, description, audio_path,
              transcript_path, status, error, added_at, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?11)",
            params![
                ep.guid,
                ep.feed_name,
                ep.title,
                ep.published.map(|d| d.to_rfc3339()),
                ep.audio_url,
                ep.description,
                ep.audio_path
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                ep.transcript_path
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                ep.status.as_str(),
                ep.error,
                now,
            ],
        )?;
        Ok(n == 1)
    }

    pub fn set_status(&self, guid: &str, status: Status, error: Option<&str>) -> Result<()> {
        self.conn.execute(
            "UPDATE episodes SET status=?1, error=?2, updated_at=?3 WHERE guid=?4",
            params![status.as_str(), error, Utc::now().to_rfc3339(), guid],
        )?;
        Ok(())
    }

    pub fn set_audio_path(&self, guid: &str, path: &Path) -> Result<()> {
        self.conn.execute(
            "UPDATE episodes SET audio_path=?1, updated_at=?2 WHERE guid=?3",
            params![path.to_string_lossy(), Utc::now().to_rfc3339(), guid],
        )?;
        Ok(())
    }

    pub fn set_transcript_path(&self, guid: &str, path: &Path) -> Result<()> {
        self.conn.execute(
            "UPDATE episodes SET transcript_path=?1, updated_at=?2 WHERE guid=?3",
            params![path.to_string_lossy(), Utc::now().to_rfc3339(), guid],
        )?;
        Ok(())
    }

    pub fn clear_audio_path(&self, guid: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE episodes SET audio_path=NULL, updated_at=?1 WHERE guid=?2",
            params![Utc::now().to_rfc3339(), guid],
        )?;
        Ok(())
    }

    pub fn delete_feed(&self, feed_name: &str) -> Result<usize> {
        Ok(self.conn.execute(
            "DELETE FROM episodes WHERE feed_name=?1",
            params![feed_name],
        )?)
    }

    #[cfg(test)]
    pub fn get(&self, guid: &str) -> Result<Option<Episode>> {
        Ok(self
            .conn
            .query_row(
                "SELECT * FROM episodes WHERE guid=?1",
                params![guid],
                row_to_episode,
            )
            .optional()?)
    }

    pub fn feed_count(&self, feed_name: &str) -> Result<usize> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM episodes WHERE feed_name=?1",
            params![feed_name],
            |r| r.get(0),
        )?;
        Ok(n as usize)
    }

    /// Episodes filtered by status and/or feed. Ordered oldest-first by published date so
    /// pipelines work chronologically.
    pub fn list(&self, status: Option<Status>, feed_name: Option<&str>) -> Result<Vec<Episode>> {
        let mut sql = String::from("SELECT * FROM episodes WHERE 1=1");
        let mut args: Vec<String> = vec![];
        if let Some(s) = status {
            sql.push_str(" AND status=?");
            args.push(s.as_str().to_string());
        }
        if let Some(f) = feed_name {
            sql.push_str(" AND feed_name=?");
            args.push(f.to_string());
        }
        sql.push_str(" ORDER BY published ASC, added_at ASC");
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(args.iter()), row_to_episode)?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn counts(&self, feed_name: Option<&str>) -> Result<Vec<(Status, usize)>> {
        let mut out = vec![];
        for s in Status::ALL {
            let n: i64 = match feed_name {
                Some(f) => self.conn.query_row(
                    "SELECT COUNT(*) FROM episodes WHERE status=?1 AND feed_name=?2",
                    params![s.as_str(), f],
                    |r| r.get(0),
                )?,
                None => self.conn.query_row(
                    "SELECT COUNT(*) FROM episodes WHERE status=?1",
                    params![s.as_str()],
                    |r| r.get(0),
                )?,
            };
            out.push((s, n as usize));
        }
        Ok(out)
    }
}

fn row_to_episode(row: &Row<'_>) -> rusqlite::Result<Episode> {
    let published: Option<String> = row.get("published")?;
    let status: String = row.get("status")?;
    let audio_path: Option<String> = row.get("audio_path")?;
    let transcript_path: Option<String> = row.get("transcript_path")?;
    Ok(Episode {
        guid: row.get("guid")?,
        feed_name: row.get("feed_name")?,
        title: row.get("title")?,
        published: published
            .and_then(|p| DateTime::parse_from_rfc3339(&p).ok())
            .map(|d| d.with_timezone(&Utc)),
        audio_url: row.get("audio_url")?,
        description: row
            .get::<_, Option<String>>("description")?
            .unwrap_or_default(),
        audio_path: audio_path.map(PathBuf::from),
        transcript_path: transcript_path.map(PathBuf::from),
        status: status.parse().unwrap_or(Status::Failed),
        error: row.get("error")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(guid: &str, status: Status) -> Episode {
        Episode {
            guid: guid.into(),
            feed_name: "f".into(),
            title: format!("Episode {guid}"),
            published: Some(Utc::now()),
            audio_url: format!("https://x/{guid}.mp3"),
            description: String::new(),
            audio_path: None,
            transcript_path: None,
            status,
            error: None,
        }
    }

    #[test]
    fn insert_is_idempotent() {
        let db = Db::open_memory().unwrap();
        assert!(db.insert(&ep("a", Status::New)).unwrap());
        assert!(!db.insert(&ep("a", Status::Skipped)).unwrap());
        assert_eq!(db.get("a").unwrap().unwrap().status, Status::New);
    }

    #[test]
    fn status_transitions_and_paths() {
        let db = Db::open_memory().unwrap();
        db.insert(&ep("a", Status::New)).unwrap();
        db.set_audio_path("a", Path::new("/tmp/a.mp3")).unwrap();
        db.set_status("a", Status::Downloaded, None).unwrap();
        let got = db.get("a").unwrap().unwrap();
        assert_eq!(got.status, Status::Downloaded);
        assert_eq!(got.audio_path, Some(PathBuf::from("/tmp/a.mp3")));
        db.set_status("a", Status::Failed, Some("boom")).unwrap();
        assert_eq!(db.get("a").unwrap().unwrap().error.as_deref(), Some("boom"));
        db.clear_audio_path("a").unwrap();
        assert!(db.get("a").unwrap().unwrap().audio_path.is_none());
    }

    #[test]
    fn list_filters_and_counts() {
        let db = Db::open_memory().unwrap();
        db.insert(&ep("a", Status::New)).unwrap();
        db.insert(&ep("b", Status::Skipped)).unwrap();
        let mut c = ep("c", Status::New);
        c.feed_name = "other".into();
        db.insert(&c).unwrap();
        assert_eq!(db.list(Some(Status::New), None).unwrap().len(), 2);
        assert_eq!(db.list(Some(Status::New), Some("f")).unwrap().len(), 1);
        assert_eq!(db.list(None, Some("f")).unwrap().len(), 2);
        let counts = db.counts(None).unwrap();
        assert!(counts.contains(&(Status::New, 2)));
        assert!(counts.contains(&(Status::Skipped, 1)));
        assert_eq!(db.delete_feed("f").unwrap(), 2);
        assert_eq!(db.feed_count("f").unwrap(), 0);
    }
}
