# Contributing

Issues and pull requests are welcome.

- **Bugs:** open an issue with the command you ran, the feed URL if it is public, and the
  error output. Private feed URLs contain access tokens; never paste them.
- **Changes:** run `cargo test --release`, `cargo clippy --release --all-targets` and
  `cargo fmt --check` before opening a PR. Add a unit test for any behaviour change in
  `feeds.rs`, `pipeline.rs` or `vocab.rs`; these are pure and easy to test without network.
- **Vocabulary heuristics** (`src/vocab.rs`): precision beats recall. A false correction
  in ordinary prose is worse than a missed name, so any loosening of the matcher needs a
  regression case in `no_false_positives_on_ordinary_prose`.
