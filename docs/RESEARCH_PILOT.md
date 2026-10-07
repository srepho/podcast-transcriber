# Podcast evidence pilot

The initial goal is to assess whether episode evidence can be extracted accurately and
later add predictive value. This pilot does not infer market probabilities or place orders.
Core collection remains a Rust CLI; the optional research exporter uses Python 3.10+
standard-library modules only (development tools: `uv sync`). The default keyword extractor sends
nothing to an external service; the optional LLM extractor (below) sends each transcript to the chosen provider.

## Collect a bounded sample

```sh
podcast refresh --feed myshow
podcast collect myshow --title 'Daily' --limit 5
python3 scripts/pilot_dataset.py build --feed myshow --title Daily --limit 5 \
  --out data/pilots/sample-001
```

Collection chooses up to ten matching episodes, newest first, from the discovered database.
It reuses transcribed episodes and retries selected failures; unrelated queued episodes are
untouched. The export selects its sampling frame before checking transcript availability,
so missing/failed episodes remain in `episodes.jsonl`. A short sample may be smaller than the
requested limit; check the manifest. Collection and export should use the same title/limit.

Each bundle is a new directory containing:

- `manifest.json`: schema/extractor versions, extractor and vocabulary hashes, sampling frame,
  counts and file hashes.
- `episodes.jsonl`: hashed episode ID, publication/download/transcription/observation times,
  transcript model and source hash; unknown legacy timestamps remain null.
- `segments.jsonl`: segment IDs, second offsets, raw ASR wording and corrected wording.
- `candidates.jsonl`: keyword hints with adjacent-segment context, possible entities and
  uncertainty words. These are candidate passages, not verified factual claims.
- `review.json`: editable dispositions, claim text, type, certainty, entity mappings, expiry,
  audio-check flag and notes. Every candidate starts pending.
- `REVIEW.md`: readable evidence with timestamps for review.

Credentials, enclosure URLs and raw GUIDs are excluded from research bundles. Transcript
content is still private source material: keep everything under gitignored `data/`.

## Review before modelling

Review all segments of the pilot, not just keyword hits, and record missed actionable claims
in a separate local evaluation note. Keyword precision/recall is not established by software tests.
For candidates, listen around the timestamps and check negation, who is speaking, attribution,
conditional wording, dates and names. A corrected name may itself be wrong.

In `review.json`, set a reviewer and each disposition to `accepted`, `rejected` or `pending`.
Acceptance requires:

- A supported `claim_text` and explanatory `notes`.
- `claim_type`: injury, rotation, matchup, transaction, contract or form.
- `certainty`: reported_fact, opinion, speculation, conditional or unknown.
- At least one entity: `{"type":"player","name":"Alex Example","model_entity_id":"espn:123"}`
  (or type `team`). Resolve against the target model's entity catalogue; no fuzzy ID join is
  performed. Ambiguous mappings stay pending.
- `temporal_scope`: current, upcoming, historical or unknown. Only current/upcoming
  claims can be selected as signals; a new episode can discuss an old event.
- An explicit timezone-aware `valid_until` expiry, later than the episode's publication.
- An honest boolean `audio_checked`; acceptance can record transcript-only review, but that
  limitation must remain visible when assessing extraction quality.

```sh
python3 scripts/pilot_dataset.py finalize --bundle data/pilots/sample-001 \
  --review data/pilots/sample-001/review.json --out data/pilots/reviewed-001
python3 scripts/pilot_dataset.py select --dataset data/pilots/reviewed-001 \
  --decision-time 2026-10-01T10:00:00Z --out data/pilots/cutoff-001
```

`finalize` verifies bundle hashes and freezes a copy of every disposition, including rejected
and pending records. `select` writes both `eligible.jsonl` and a full `audit.jsonl` with exclusion
reasons. Finalized records cannot be silently edited; revised reviews create a new version.
Every command writes to a staging directory and renames it into place, so an interrupted run
leaves no partial output.

Before trusting any row, `select` checks that the claims, the frozen review copy and the
manifest agree (hashes, dispositions, availability times and counts). That catches accidental
edits, but a manifest can always be rewritten to match. For an experiment, record the
finalized manifest's SHA-256 in the experiment config and pass it as
`--expected-manifest-sha256`; the selection manifest records the hash it verified.

## Time and model contract

All cutoffs require timezones. An accepted record's `model_ready_at` is the finalization time,
a conservative observed availability after collection, transcription, extraction and review.
Eligibility requires `model_ready_at < decision_time`, publication before that cutoff when
known, acceptance, current/upcoming temporal scope, and an unexpired claim. Equality at the cutoff is excluded. Legacy download
and transcription times are never guessed from publication, filesystem times or DB updated_at.
Re-correction produces a different transcript hash and requires a new bundle/review.

For an explicitly retrospective experiment,
`select --historical-delay-hours 24 --historical-validity-days 7` substitutes publication plus
the declared processing delay. The manifest is marked `research_only: true` and every row has
`timing_basis: historical_assumption`. This is not evidence that the data was actually
available then. Run several predeclared delay scenarios; do not mix assumed and observed
records in one performance claim.

In historical mode the reviewer's `valid_until` is ignored: it was written after the fact and can
encode the outcome (for example, an expiry set to the day a player actually returned). Expiry is
instead the assumed availability plus `--historical-validity-days`, declared before evaluation,
and rows carry `expiry_basis: historical_policy` and `expires_at`. Hindsight can still leak through
`temporal_scope`, `certainty` and the choice to accept a claim; review historical episodes without
looking up what happened next.

The target model can read JSONL with pandas and join explicitly mapped entities to fixtures,
then use that fixture's saved `decision_time`. This repository does not modify the prediction
model. Episode mentions must not be treated as independent matches or independent samples.

## Pilot gate

1. Review five to ten complete episodes; record candidate correctness, entity/correction errors,
   missed claims, audio verification and processing latency. Keep rejected/missing records.
2. Select a narrow feature definition before measuring outcomes (for example a reviewed injury
   report with an explicit expiry). Freeze entity mapping, aggregation and timing policy.
3. Evaluate baseline versus baseline plus podcast features on identical fixtures with chronological
   splits, validation-only selection and a held-out period. Report coverage and uncertainty.
4. A five-episode sample validates the extraction workflow; it cannot establish a predictive edge.

## Validation

```sh
cargo test --release --locked
cargo clippy --release --all-targets -- -D warnings
cargo fmt --check
uv sync && uv run ruff check && uv run mypy && uv run pytest
```

## Audio checks and expanded evidence

`select --require-audio-checked` excludes otherwise eligible records unless the reviewer has
set the boolean `audio_checked` to true after human listening. Local ASR re-transcription,
including a different decoding strategy using the same model, is a consistency check and
must leave this flag false. Shared-model passes can repeat the same error.

If a candidate omits its subject or important context, add `additional_segment_ids` to that
entry in the editable review. Finalization resolves these against the hashed segment file
and requires the same episode and transcript version. It preserves the original candidate
and separately exports `reviewed_segment_ids`, reviewed raw/corrected evidence and its time
span. Do not silently rewrite source wording or infer a missing subject.

For an offseason sample, prioritize contracts, transactions, roster continuity and explicit
conditional cap scenarios. Scarce injury or rotation reporting is expected, not evidence of
poor extraction recall. Keep proposed offers distinct from signed contracts, and hypothetical
roster scenarios distinct from completed transactions. Resolve identity separately from
current team membership: a podcast can report a change before a saved roster reflects it.
Hold unresolved roster transitions for corroboration rather than automatically overwriting
model rosters or declaring the report false.

Expiry is a feature-freshness requirement, not a claim that an injury has healed or a contract
has ceased to exist. Define it before fixture evaluation; do not extend it merely to obtain
eligible observations. Longer-lived offseason state features need explicit update/supersession
rules and a separate validation policy.

## Scheduled collection and LLM-proposed claims

`scripts/daily.py FEED PUBLISHED_AFTER` runs `podcast run` (refresh, download, transcribe) and then

```sh
python scripts/pilot_dataset.py build --feed FEED --unprocessed --published-after 2026-10-01T00:00:00Z \
  --extractor llm --provider anthropic --athletes athletes.jsonl --out data/pilots/auto-<UTC stamp>
```

`--unprocessed` selects transcribed episodes published after the floor that no earlier unprocessed
build has extracted (`data/pilots/extracted.json`), and writes nothing when there are none. An episode
whose extraction fails stays in the bundle as `extraction_failed`, is listed in the manifest's
`extraction_failures`, and is retried by the next run.

The LLM extractor (`scripts/llm_claims.py`, `uv sync --group extract`) makes one structured-output
request per transcript and turns each proposed claim into an ordinary candidate whose review entry
is pre-filled but still `pending`: claim text, type, certainty, temporal scope, entities, and an
expiry of publication plus a fixed per-type window (`EXPIRY_DAYS`). Proposals citing segments that
do not exist, spans longer than `MAX_SPAN`, or invalid labels are dropped, never repaired. Entity IDs
come only from an exact (case-, accent- and punctuation-insensitive) match against `--athletes` and
the built-in ESPN team table; ambiguous or unknown names stay unmapped for the reviewer.

Providers: `anthropic` (default model `claude-opus-5-5`, schema enforced, server-side refusal
fallback), `openai` (strict JSON schema), and OpenAI-compatible `deepseek`, `qwen`, `moonshot`,
`zhipu` or `compatible` with `--base-url`. Non-Anthropic providers need `--model`. Per-provider request
options live in `PROVIDERS`: DeepSeek runs at `reasoning_effort: low` with a 32k output limit,
because unbounded reasoning used the whole output budget on a full transcript and reasoning off
produced confidently wrong names. The
compatible providers only guarantee JSON, so the schema is stated in the prompt and the validator
does the rest. Provider, model and endpoint are part of the manifest's `extractor_sha256` and each
candidate's `extractor_model`; compare extractors on the same episodes before switching. Keys come
from each provider's usual environment variable, the keychain service `podcast-<provider>`, or
`ENV_FILE` (a dotenv file from which only that variable is read).

Review with `python3 scripts/review_claims.py BUNDLE --finalize`: accept, reject, edit text,
certainty or expiry, or set an entity's model ID. Acceptance still requires mapped IDs and an
expiry, and records `audio_checked: false`. `scripts/launchd/com.podcast.daily.plist` is a template
for running the job every six hours through the venv's Python (launchd cannot run a bash script
under `~/Downloads`); keep its log path outside `~/Downloads`, `~/Documents` and
`~/Desktop`.

Sending transcripts to a provider subjects private source material to that provider's retention
policy and jurisdiction. Check the provider's terms before scheduling it.
