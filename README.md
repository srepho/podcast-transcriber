# podcast

Subscribe to podcast RSS feeds, download episodes (new or historical), and transcribe them
locally with whisper.cpp. Single static Rust binary; no Python, no cloud, no API keys.

## Requirements

- Rust toolchain (`cargo`), `cmake` (builds whisper.cpp), `ffmpeg` on PATH (`brew install ffmpeg`).
- Runs on CPU. On a 4-core Intel Mac, `base.en` transcribes at roughly 8x realtime
  (a 60-minute episode in about 7-8 minutes). `small.en` is ~3x slower and noticeably better.

## Build

```bash
cargo build --release
./target/release/podcast --help      # optionally: cp target/release/podcast ~/.local/bin/
```

## Quick start

```bash
podcast init                         # writes config.yaml
podcast add https://example.com/feed.rss --name myshow
podcast run                          # refresh feeds, download queued, transcribe downloaded
```

Transcripts land in `data/transcripts/<feed>/<date>_<title>_<id>.{txt,srt,json}`.
The ID suffix is a stable hash of the subscription name and episode GUID, so episodes
with the same date/title have separate files. Existing files remain at their recorded paths.
Audio is kept in `data/audio/<feed>/` unless `delete_audio_after_transcribe: true`.

Existing databases are upgraded transactionally on first use to identify episodes by
subscription and GUID together. Records, statuses and paths are preserved; keep a backup
before using an older binary with an upgraded database.

## Historical episodes

Adding a feed only queues the newest `max_new_per_feed` (default 3) episodes; everything
older is recorded as `skipped` so nothing is downloaded by surprise. Later refreshes queue
unseen episodes published after the newest one already known. Unseen episodes that are older
(typically a feed migration that changed every GUID or audio URL) are recorded as `skipped`
with a warning instead of queuing the whole back catalogue. Pull history on demand:

```bash
podcast backfill myshow --dry-run --since 2025-01-01      # preview
podcast backfill myshow --since 2025-01-01                # queue
podcast backfill myshow --limit 20                        # newest 20 not yet processed
podcast backfill myshow --title "Season Outlook"          # title substring
podcast backfill myshow --all                             # everything
podcast download && podcast transcribe                    # or: podcast run
```

Backfill follows `rel="next"` feed pagination, so it finds episodes beyond the first page
where the feed supports it.

## Getting names right

Generic speech-to-text mangles proper nouns ("Yokic", "Antetokumpo", "Sean Sharani").
Three layers address this:

1. **Prompting.** Each episode is transcribed with a whisper prompt built from the feed's
   `prompt` (set it in `config.yaml`: hosts, subject matter), proper nouns found in the
   episode's show notes, vocabulary names mentioned in the show notes, and the episode
   title. whisper.cpp keeps only the last ~224 tokens of the prompt, so the most
   episode-specific material goes last.
2. **Vocabulary correction.** `data/vocab/<feed>.txt` holds one name per line. After
   transcription every name is fuzzy-matched against the text (letters-only comparison with
   a phonetic fold, so y/j and c/k confusions count as equal) and mangled spellings are
   replaced. Guard rails keep it from touching ordinary prose: a match must contain a
   capitalized word, cannot start or end on a stopword (unless the name itself does, as in
   "Will Hardy"), cannot cross a sentence boundary, and for two-word names the surname must
   hold up on its own. Matching runs across whisper's segment boundaries, so a name split
   between two segments is still found; the corrected name goes in the first segment.
3. **Aliases.** For recurring garbles the fuzzy matcher can't reach, add an explicit line:
   `nicole yokage => Nikola Jokić`.

```bash
podcast vocab nba myshow                 # every current NBA player, coach and team (ESPN)
podcast vocab add myshow "Jane Host" "sean sharani => Shams Charania"
podcast vocab import myshow names.txt    # one name per line
podcast vocab test myshow "Yarnis Antetokumpo and Nikola Yokic were great"
podcast correct myshow                   # re-apply vocab to existing transcripts
```

Every correction is recorded in the `.json` transcript (`corrections`), and the raw whisper
text is preserved per segment (`raw_text`), so `podcast correct` can always start over
from the original output after the vocabulary changes. Tune aggressiveness with
`correction_threshold` (default 0.8; lower is more aggressive).

`scripts/ctg_names.py` pulls two-word names out of any nested JSON stats dump (it looks
for keys named like `name` or `player`) to feed `vocab import` with historical players.

## Commands

For a bounded prediction-model evidence sample, see [the research pilot guide](docs/RESEARCH_PILOT.md).
It covers timestamped provenance, raw/corrected evidence, manual claim review and strict
decision-time filtering. The optional exporter uses Python's standard library; collection
and transcription remain in the Rust binary.

| Command | Purpose |
|---|---|
| `init`, `add`, `remove`, `feeds` | manage subscriptions |
| `refresh`, `download`, `transcribe`, `run` | the pipeline, separately or in one go |
| `backfill <feed> ...` | queue older episodes |
| `status`, `episodes [--status s] [--feed f]` | inspect the queue |
| `retry [--redownload]` | re-queue failed episodes; `--redownload` deletes their audio first |
| `model list`, `model download <name>` | whisper models (auto-downloaded on first use) |
| `vocab ...`, `correct <feed>` | name correction |
| `collect <feed> --title TEXT --limit 5` | collect a bounded sample without processing unrelated queued work |
| `file <audio>` | transcribe a local file with no feed |

Every command takes `--config <path>` (default `./config.yaml`) and `--limit N` where
it makes sense, so `podcast run --limit 2` is a safe cron/launchd job.

Commands that change state take a lock on `data/podcast.lock`, so an overlapping scheduled
run exits with an error instead of processing the same episodes twice. `status` and
`episodes` never wait for it.

Downloads that are not audio (an HTML or JSON error page, an empty or truncated body) fail
instead of being saved. If a file on disk is corrupt, `podcast retry --redownload` fetches it
again; plain `retry` reuses audio already on disk.

Pipeline commands exit nonzero when any attempted item fails, while retaining successful
work. `run` attempts all three stages even if an earlier one reports failures. Inspect
`status` and use `retry` after resolving the cause. Empty transcription output is a failure:
the audio is retained. Transcript files are staged before publication and replaced atomically
per file; replacing several formats is not one transaction.

Feed listings and HTTP diagnostics hide URL paths, queries and credentials. Configuration,
the state database and JSON transcript metadata still contain private URLs; keep these private.

## Config reference

```yaml
data_dir: data
max_new_per_feed: 3            # queued when a feed is first added; older = skipped
delete_audio_after_transcribe: false
formats: [txt, srt, json]      # keep json if you want `correct` to work later
max_feed_pages: 50             # pagination cap for backfill
feed_timeout_secs: 60          # total deadline for each feed request
download_timeout_secs: 3600    # total deadline for each audio/model download
correction_threshold: 0.8
whisper:
  model: base.en               # tiny.en base.en small.en medium.en large-v3-turbo large-v3
  model_path: null             # explicit ggml file, overrides model
  language: en                 # null = auto-detect (multilingual models only)
  threads: 4
  beam_size: 1                 # 1 = greedy (fast); 5 = beam search (slower, a bit better)
  initial_prompt: null         # global prompt prefix
feeds:
  - name: myshow
    url: https://...
    enabled: true
    prompt: "An NBA podcast hosted by Jane Host and Sam Cohost."
```

## Scheduling on macOS

```bash
# every 6 hours, at most 3 episodes per run
cat > ~/Library/LaunchAgents/com.user.podcast.plist <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.user.podcast</string>
  <key>ProgramArguments</key><array>
    <string>/ABSOLUTE/PATH/podcast/target/release/podcast</string>
    <string>--config</string><string>/ABSOLUTE/PATH/podcast/config.yaml</string>
    <string>run</string><string>--limit</string><string>3</string>
  </array>
  <key>WorkingDirectory</key><string>/ABSOLUTE/PATH/podcast</string>
  <key>StartInterval</key><integer>21600</integer>
  <key>StandardOutPath</key><string>/tmp/podcast.log</string>
  <key>StandardErrorPath</key><string>/tmp/podcast.log</string>
</dict></plist>
PLIST
launchctl load ~/Library/LaunchAgents/com.user.podcast.plist
```

## License

MIT. See `LICENSE`.

## Development

```bash
cargo test --release
cargo clippy --release --all-targets
cargo fmt --check
```
