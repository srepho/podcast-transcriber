//! One pipeline process per data directory. Overlapping runs (e.g. a scheduled `run` firing
//! while a long transcription is still going) would otherwise pick up the same queued episodes
//! and write the same `.part` files.

use anyhow::{Context, Result};
use std::fs::{File, OpenOptions, TryLockError};
use std::path::Path;

/// Held for the life of a command; the OS releases the lock when the file is closed,
/// including when the process crashes.
pub struct RunLock {
    _file: File,
}

pub fn acquire(data_dir: &Path) -> Result<RunLock> {
    std::fs::create_dir_all(data_dir)?;
    let path = data_dir.join("podcast.lock");
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    match file.try_lock() {
        Ok(()) => Ok(RunLock { _file: file }),
        Err(TryLockError::WouldBlock) => anyhow::bail!(
            "another podcast command is already working in {}; try again when it finishes",
            data_dir.display()
        ),
        Err(TryLockError::Error(e)) => {
            Err(e).with_context(|| format!("locking {}", path.display()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_holder_is_refused_until_the_first_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let first = acquire(dir.path()).unwrap();
        let err = acquire(dir.path()).err().unwrap();
        assert!(format!("{err:#}").contains("already working"));
        drop(first);
        acquire(dir.path()).unwrap();
    }

    #[test]
    fn separate_data_dirs_do_not_contend() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let _first = acquire(a.path()).unwrap();
        acquire(b.path()).unwrap();
    }

    #[test]
    fn creates_missing_data_dir() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("not/yet");
        acquire(&nested).unwrap();
        assert!(nested.join("podcast.lock").exists());
    }
}
