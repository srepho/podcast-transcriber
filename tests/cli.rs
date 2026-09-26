use std::path::Path;
use std::process::{Command, Output};

fn invoke(config: &Path, command: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_podcast"))
        .arg("--config")
        .arg(config)
        .arg(command)
        .output()
        .unwrap()
}

fn config(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("config.yaml");
    std::fs::write(&path, format!(
        "data_dir: {}\nfeeds:\n  - name: demo\n    url: invalid://FAKE_SECRET/feed?token=FAKE_SECRET\n",
        dir.join("data").display())).unwrap();
    path
}

#[test]
fn failed_refresh_exits_nonzero_and_never_prints_private_url() {
    let dir = tempfile::tempdir().unwrap();
    let config = config(dir.path());
    let output = invoke(&config, "refresh");
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("1 item(s) failed"));
    assert!(!stderr.contains("FAKE_SECRET"));
    let feeds = invoke(&config, "feeds");
    assert!(feeds.status.success());
    assert!(!String::from_utf8_lossy(&feeds.stdout).contains("FAKE_SECRET"));
}

#[test]
fn run_processes_queue_after_refresh_failure_and_reports_both_failures() {
    let dir = tempfile::tempdir().unwrap();
    let config = config(dir.path());
    assert!(invoke(&config, "status").status.success());
    let db = rusqlite::Connection::open(dir.path().join("data/podcast.db")).unwrap();
    db.execute("INSERT INTO episodes (guid, feed_name, title, audio_url, status, added_at, updated_at)
        VALUES ('episode', 'demo', 'Test episode', 'invalid://FAKE_SECRET/audio', 'new', 'now', 'now')", []).unwrap();
    let output = invoke(&config, "run");
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("refresh: 1 item(s) failed"), "{stderr}");
    assert!(stderr.contains("download: 1 item(s) failed"), "{stderr}");
    assert!(stderr.contains("transcribe: 0 completed"), "{stderr}");
    assert!(!stderr.contains("FAKE_SECRET"));
    let status: String = db
        .query_row("SELECT status FROM episodes", [], |r| r.get(0))
        .unwrap();
    assert_eq!(status, "failed");
    assert!(invoke(&config, "retry").status.success());
    assert_eq!(invoke(&config, "download").status.code(), Some(1));
}

#[test]
fn empty_queue_exits_successfully() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config.yaml");
    std::fs::write(
        &config,
        format!("data_dir: {}\n", dir.path().join("data").display()),
    )
    .unwrap();
    assert!(invoke(&config, "run").status.success());
}
