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
    guid            TEXT NOT NULL,
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
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (feed_name, guid)
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
        let mut conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        Self::initialize(&mut conn)?;
        Ok(Self { conn })
    }

    /// Upgrade the original GUID-only key transactionally, preserving every column/path.
    fn initialize(conn: &mut Connection) -> Result<()> {
        let tx = conn.transaction()?;
        let legacy = {
            let mut stmt = tx.prepare("PRAGMA table_info(episodes)")?;
            let columns = stmt
                .query_map([], |r| Ok((r.get::<_, String>(1)?, r.get::<_, i64>(5)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            columns.iter().any(|(name, pk)| name == "guid" && *pk == 1)
                && columns
                    .iter()
                    .any(|(name, pk)| name == "feed_name" && *pk == 0)
        };
        if legacy {
            tx.execute_batch("ALTER TABLE episodes RENAME TO episodes_legacy;")?;
            tx.execute_batch(SCHEMA)?;
            tx.execute_batch(
                "INSERT INTO episodes SELECT * FROM episodes_legacy;
                              DROP TABLE episodes_legacy;",
            )?;
        }
        // Recreate indexes after dropping the legacy table (which owned their names).
        tx.execute_batch(SCHEMA)?;
        let has_download_time = {
            let mut stmt = tx.prepare("PRAGMA table_info(episodes)")?;
            let names = stmt
                .query_map([], |r| r.get::<_, String>(1))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            names.iter().any(|n| n == "downloaded_at")
        };
        if !has_download_time {
            tx.execute_batch("ALTER TABLE episodes ADD COLUMN downloaded_at TEXT;")?;
        }
        tx.commit()?;
        Ok(())
    }

    #[cfg(test)]
    pub fn open_memory() -> Result<Self> {
        let mut conn = Connection::open_in_memory()?;
        Self::initialize(&mut conn)?;
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

    pub fn set_status(
        &self,
        feed_name: &str,
        guid: &str,
        status: Status,
        error: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE episodes SET status=?1, error=?2, updated_at=?3 WHERE guid=?4 AND feed_name=?5",
            params![
                status.as_str(),
                error,
                Utc::now().to_rfc3339(),
                guid,
                feed_name
            ],
        )?;
        Ok(())
    }

    pub fn set_audio_path(&self, feed_name: &str, guid: &str, path: &Path) -> Result<()> {
        self.conn.execute(
            "UPDATE episodes SET audio_path=?1, updated_at=?2, downloaded_at=?2 WHERE guid=?3 AND feed_name=?4",
            params![
                path.to_string_lossy(),
                Utc::now().to_rfc3339(),
                guid,
                feed_name
            ],
        )?;
        Ok(())
    }

    pub fn downloaded_at(&self, feed_name: &str, guid: &str) -> Result<Option<String>> {
        Ok(self.conn.query_row(
            "SELECT downloaded_at FROM episodes WHERE feed_name=?1 AND guid=?2",
            params![feed_name, guid],
            |r| r.get(0),
        )?)
    }

    pub fn set_transcript_path(&self, feed_name: &str, guid: &str, path: &Path) -> Result<()> {
        self.conn.execute(
            "UPDATE episodes SET transcript_path=?1, updated_at=?2 WHERE guid=?3 AND feed_name=?4",
            params![
                path.to_string_lossy(),
                Utc::now().to_rfc3339(),
                guid,
                feed_name
            ],
        )?;
        Ok(())
    }

    pub fn clear_audio_path(&self, feed_name: &str, guid: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE episodes SET audio_path=NULL, updated_at=?1 WHERE guid=?2 AND feed_name=?3",
            params![Utc::now().to_rfc3339(), guid, feed_name],
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
    pub fn get(&self, feed_name: &str, guid: &str) -> Result<Option<Episode>> {
        Ok(self
            .conn
            .query_row(
                "SELECT * FROM episodes WHERE guid=?1 AND feed_name=?2",
                params![guid, feed_name],
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

    /// Newest publication date recorded for a feed, in any status.
    pub fn latest_published(&self, feed_name: &str) -> Result<Option<DateTime<Utc>>> {
        let mut stmt = self.conn.prepare(
            "SELECT published FROM episodes WHERE feed_name=?1 AND published IS NOT NULL",
        )?;
        let dates = stmt.query_map(params![feed_name], |r| r.get::<_, String>(0))?;
        let mut latest = None;
        for date in dates {
            if let Ok(d) = DateTime::parse_from_rfc3339(&date?) {
                latest = latest.max(Some(d.with_timezone(&Utc)));
            }
        }
        Ok(latest)
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
        assert_eq!(db.get("f", "a").unwrap().unwrap().status, Status::New);
    }

    #[test]
    fn status_transitions_and_paths() {
        let db = Db::open_memory().unwrap();
        db.insert(&ep("a", Status::New)).unwrap();
        db.set_audio_path("f", "a", Path::new("/tmp/a.mp3"))
            .unwrap();
        assert!(
            DateTime::parse_from_rfc3339(&db.downloaded_at("f", "a").unwrap().unwrap()).is_ok()
        );
        db.set_status("f", "a", Status::Downloaded, None).unwrap();
        let got = db.get("f", "a").unwrap().unwrap();
        assert_eq!(got.status, Status::Downloaded);
        assert_eq!(got.audio_path, Some(PathBuf::from("/tmp/a.mp3")));
        db.set_status("f", "a", Status::Failed, Some("boom"))
            .unwrap();
        assert_eq!(
            db.get("f", "a").unwrap().unwrap().error.as_deref(),
            Some("boom")
        );
        db.clear_audio_path("f", "a").unwrap();
        assert!(db.get("f", "a").unwrap().unwrap().audio_path.is_none());
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

    #[test]
    fn same_guid_in_different_feeds_has_independent_state() {
        let db = Db::open_memory().unwrap();
        let first = ep("shared", Status::New);
        let mut second = first.clone();
        second.feed_name = "other".into();
        assert!(db.insert(&first).unwrap());
        assert!(db.insert(&second).unwrap());
        db.set_audio_path("f", "shared", Path::new("audio.mp3"))
            .unwrap();
        db.set_transcript_path("f", "shared", Path::new("transcript.txt"))
            .unwrap();
        db.set_status("f", "shared", Status::Transcribed, None)
            .unwrap();
        assert_eq!(db.get("other", "shared").unwrap().unwrap(), second);
        db.clear_audio_path("f", "shared").unwrap();
        assert!(db.get("f", "shared").unwrap().unwrap().audio_path.is_none());
        db.delete_feed("f").unwrap();
        assert_eq!(db.get("other", "shared").unwrap().unwrap(), second);
    }

    #[test]
    fn legacy_database_migration_preserves_records_and_paths() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.db");
        let conn = Connection::open(&path).unwrap();
        let legacy = SCHEMA
            .replace(
                "guid            TEXT NOT NULL",
                "guid            TEXT PRIMARY KEY",
            )
            .replace(",\n    PRIMARY KEY (feed_name, guid)", "");
        conn.execute_batch(&legacy).unwrap();
        let old = Db { conn };
        let mut original = ep("shared", Status::Transcribed);
        original.audio_path = Some(PathBuf::from("original/audio.mp3"));
        original.transcript_path = Some(PathBuf::from("original/transcript.txt"));
        original.description = "Retained show notes".into();
        old.insert(&original).unwrap();
        old.conn
            .execute(
                "UPDATE episodes SET added_at='old-added', updated_at='old-updated'",
                [],
            )
            .unwrap();
        drop(old);
        let db = Db::open(&path).unwrap();
        assert_eq!(db.downloaded_at("f", "shared").unwrap(), None);
        assert_eq!(db.get("f", "shared").unwrap().unwrap(), original);
        let timestamps: (String, String) = db
            .conn
            .query_row("SELECT added_at, updated_at FROM episodes", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(timestamps, ("old-added".into(), "old-updated".into()));
        let mut other = original.clone();
        other.feed_name = "other".into();
        assert!(db.insert(&other).unwrap());
        drop(db);
        let reopened = Db::open(&path).unwrap();
        assert_eq!(reopened.list(None, None).unwrap().len(), 2);
        assert_eq!(reopened.get("f", "shared").unwrap().unwrap(), original);
    }
}
