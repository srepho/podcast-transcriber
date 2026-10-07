#!/usr/bin/env python3
"""Local research bundles: build -> edit review.json -> finalize -> select.

Standard library only. Hints are review candidates, never automatic factual claims.
"""
import argparse
import hashlib
import json
import math
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Callable
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
EXTRACTOR_VERSION = "keyword-context-v1"
KINDS = {
    "injury": r"\b(injur\w*|surgery|rehab|sprain\w*|fractur\w*|concussion|out indefinitely)\b",
    "rotation": r"\b(rotation|minutes|starting lineup|starter|bench|playing time)\b",
    "matchup": r"\b(matchup|match-up|defend\w*|switch\w*|pick.and.roll)\b",
    "transaction": r"\b(trad\w*|signing|waiv\w*|roster|free agent)\b",
    "contract": r"\b(contract|extension|salary|cap space)\b",
    "form": r"\b(shooting|efficien\w*|improv\w*|declin\w*)\b",
}
UNCERTAINTY = r"\b(?:if|might|may|could|would|likely|reportedly|expect|think)\b"
CERTAINTIES = {"reported_fact", "opinion", "speculation", "conditional", "unknown"}
MAX_DELAY_HOURS = 24 * 365
MAX_VALIDITY_DAYS = 365
# chrono's to_rfc3339() writes 1-9 fraction digits; Python 3.10 only parses exactly 3 or 6.
FRACTION = re.compile(r"\.(\d+)(?=[+-]\d\d:\d\d$|$)")


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError(f"timestamps must be ISO 8601 strings, got {type(value).__name__}")
    text = FRACTION.sub(lambda m: "." + (m.group(1) + "00000")[:6], value.replace("Z", "+00:00"))
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"invalid timestamp: {value!r}") from None
    if dt.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return dt.astimezone(timezone.utc)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_json(path, value):
    path.write_bytes(encoded(value))


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in rows))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@contextmanager
def new_directory(path):
    # A new directory per run/version prevents silent replacement of research evidence.
    # Files are staged beside it and published by rename, so a crash leaves no partial output.
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        yield staging
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")
        staging.rename(path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def string(value, field):
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def check_time(value):
    if value is not None:
        timestamp(value)
    return value


def extracted_path(data_dir):
    return data_dir / "pilots" / "extracted.json"


def build(data_dir, feed, title, limit, out, extractor="keyword", unprocessed=False, published_after=None,
          catalogue=None, call=None, provider="anthropic", model=None, base_url=None):
    """Export a review bundle. `unprocessed` selects transcribed episodes not yet extracted by any earlier
    unprocessed build (tracked in pilots/extracted.json) and returns None when there are none."""
    if not 1 <= limit <= 10:
        raise ValueError("pilot limit must be 1..10")
    if extractor not in ("keyword", "llm"):
        raise ValueError("extractor must be keyword or llm")
    if unprocessed and published_after is None:
        raise ValueError("--unprocessed needs --published-after so the backlog is never extracted by accident")
    generated = now()
    with closing(sqlite3.connect(data_dir.joinpath("podcast.db").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        # Select the cohort before checking transcription availability; missing records stay visible.
        episodes = [dict(r) for r in db.execute(
            "SELECT * FROM episodes WHERE feed_name=? ORDER BY published DESC, guid", (feed,)
        ) if title.casefold() in (r["title"] or "").casefold()]
    if unprocessed:
        state = json.loads(extracted_path(data_dir).read_text()) if extracted_path(data_dir).exists() else {}
        floor = timestamp(published_after)
        episodes = [ep for ep in episodes if ep["status"] == "transcribed" and ep["published"]
                    and timestamp(ep["published"]) >= floor and digest(encoded([feed, ep["guid"]])) not in state]
        if not episodes:
            return None
    episodes = episodes[:limit]
    if not episodes:
        raise ValueError("no matching episodes")
    llm: Any = None
    engine: Any = None
    if extractor == "llm":
        import llm_claims  # needs the `extract` dependency group; keyword mode stays stdlib-only

        llm = llm_claims
        engine = llm.Extractor(provider, model, base_url)
        catalogue = catalogue if catalogue is not None else llm.Catalogue(players={})
    failures: list[dict[str, str]] = []
    terms = []
    vocab = data_dir / "vocab" / f"{feed}.txt"
    vocab_sha256 = digest(vocab.read_bytes()) if vocab.exists() else None
    if vocab.exists():
        for line in vocab.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                terms.append(line.split("=>")[-1].strip())
    term_patterns = [(term, re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.I)) for term in set(terms)]
    episode_rows, segments, candidates, reviews = [], [], [], []
    for ep in episodes:
        episode_id = digest(encoded([feed, ep["guid"]]))
        row = {"schema_version": SCHEMA_VERSION, "episode_id": episode_id,
               "feed": feed, "title": ep["title"], "published_at": check_time(ep["published"]),
               "downloaded_at": check_time(ep.get("downloaded_at")), "transcribed_at": None,
               "observed_at": generated, "model_ready_at": None, "source_status": ep["status"],
               "audit_status": "missing_transcript", "transcript_sha256": None}
        path = Path(ep["transcript_path"]).with_suffix(".json") if ep["transcript_path"] else None
        if path is not None and path.exists():
            source = path.read_bytes()
            transcript = json.loads(source)
            if transcript["guid"] != ep["guid"] or transcript["feed"] != feed:
                raise ValueError("transcript identity does not match database record")
            row.update(transcript_sha256=digest(source), model=transcript["model"],
                       language=transcript.get("language"), producer_version=transcript.get("producer_version"),
                       transcribed_at=check_time(transcript.get("transcribed_at")),
                       downloaded_at=check_time(transcript.get("downloaded_at") or ep.get("downloaded_at")),
                       audit_status="ready_for_review")
            for field in ("downloaded_at", "transcribed_at"):
                if row[field] and timestamp(row[field]) > timestamp(generated):
                    raise ValueError(f"future {field}; check system clock")
            if row["downloaded_at"] and row["transcribed_at"] and timestamp(row["transcribed_at"]) < timestamp(row["downloaded_at"]):
                raise ValueError("transcription precedes download")
            items = transcript["segments"]
            for index, segment in enumerate(items):
                start, end = segment["start"], segment["end"]
                if not all(isinstance(n, (int, float)) and math.isfinite(n) for n in (start, end)) or not 0 <= start <= end:
                    raise ValueError("invalid segment timing")
                string(segment["text"], "segment text")
                raw = segment.get("raw_text")
                if raw is not None:
                    string(raw, "segment raw_text")
                segments.append({"schema_version": SCHEMA_VERSION, "episode_id": episode_id,
                                 "segment_id": f"{episode_id}:{index}", "index": index,
                                 "start_secs": start, "end_secs": end,
                                 "raw_text": raw if raw is not None else segment["text"],
                                 "corrected_text": segment["text"],
                                 "transcript_sha256": row["transcript_sha256"]})
                kinds = [kind for kind, pattern in KINDS.items() if re.search(pattern, segment["text"], re.I)]
                if extractor != "keyword" or not kinds:
                    continue
                lo, hi = max(0, index - 1), min(len(items), index + 2)
                text = " ".join(s["text"].strip() for s in items[lo:hi])
                raw_text = " ".join((s.get("raw_text") or s["text"]).strip() for s in items[lo:hi])
                candidate_id = digest(encoded([episode_id, row["transcript_sha256"], index, EXTRACTOR_VERSION]))
                candidates.append({"schema_version": SCHEMA_VERSION, "candidate_id": candidate_id,
                    "episode_id": episode_id, "transcript_sha256": row["transcript_sha256"],
                    "segment_ids": [f"{episode_id}:{i}" for i in range(lo, hi)],
                    "start_secs": items[lo]["start"], "end_secs": items[hi - 1]["end"],
                    "raw_evidence": raw_text, "corrected_evidence": text, "suggested_types": kinds,
                    "entity_suggestions": sorted({term for term, pattern in term_patterns if pattern.search(text)}),
                    "uncertainty_markers": sorted(set(re.findall(UNCERTAINTY, text.lower()))),
                    "extractor_version": EXTRACTOR_VERSION})
                reviews.append({"candidate_id": candidate_id, "status": "pending", "claim_type": kinds[0],
                    "claim_text": "", "certainty": "unknown", "temporal_scope": "unknown", "entities": [], "valid_until": None,
                    "audio_checked": False, "notes": ""})
            if llm is not None:
                episode_segments = [s for s in segments if s["episode_id"] == episode_id]
                try:
                    found, proposed = llm.episode_candidates(
                        episode_id, row["transcript_sha256"], ep["title"], row["published_at"], episode_segments,
                        catalogue, lambda value: digest(encoded(value)), timestamp,
                        call if call is not None else engine, engine.name)
                except Exception as error:  # one failed episode must not discard the others; it is retried later
                    failures.append({"episode_id": episode_id, "error": f"{type(error).__name__}: {error}"})
                    row["audit_status"] = "extraction_failed"
                else:
                    candidates += [{"schema_version": SCHEMA_VERSION, **c} for c in found]
                    reviews += proposed
        episode_rows.append(row)
    files = {"episodes.jsonl": episode_rows, "segments.jsonl": segments, "candidates.jsonl": candidates}
    with new_directory(out) as staging:
        for name, rows in files.items():
            write_rows(staging / name, rows)
        fingerprint = engine.fingerprint() if engine is not None else {"kinds": KINDS, "uncertainty": UNCERTAINTY}
        manifest = {"schema_version": SCHEMA_VERSION,
                    "extractor_version": llm.EXTRACTOR_VERSION if llm is not None else EXTRACTOR_VERSION,
                    "extractor_model": engine.name if engine is not None else None,
                    "extractor_sha256": digest(encoded(fingerprint)), "extraction_failures": failures,
                    "vocab_sha256": vocab_sha256,
                    "generated_at": generated, "sampling": {"feed": feed, "title_contains": title,
                    "requested": limit, "selected": len(episodes), "order": "newest_publication_first",
                    "unprocessed_only": unprocessed, "published_after": published_after},
                    "counts": {name: len(rows) for name, rows in files.items()},
                    "files": {name: digest((staging / name).read_bytes()) for name in files},
                    "availability_policy": "reviewed records become available when finalized; publication is not availability"}
        write_json(staging / "manifest.json", manifest)
        write_json(staging / "review.json", {"schema_version": SCHEMA_VERSION,
                   "manifest_sha256": digest((staging / "manifest.json").read_bytes()),
                   "reviewer": "", "reviews": reviews})
        write_review_notes(staging / "REVIEW.md", episode_rows, candidates)
    if unprocessed:
        failed = {f["episode_id"] for f in failures}
        state = json.loads(extracted_path(data_dir).read_text()) if extracted_path(data_dir).exists() else {}
        state.update({row["episode_id"]: out.name for row in episode_rows
                      if row["audit_status"] == "ready_for_review" and row["episode_id"] not in failed})
        extracted_path(data_dir).parent.mkdir(parents=True, exist_ok=True)
        write_json(extracted_path(data_dir), state)
    return manifest


def write_review_notes(path, episode_rows, candidates):
    lines = ["# Pilot evidence review", "", "Edit review.json; each candidate starts pending. Suggestions are not verified facts.",
             "Check the audio around each timestamp, resolve entity IDs, and set an expiry before accepting.",
             "Read all segments.jsonl to look for missed claims; keyword hints do not measure recall.", ""]
    titles = {row["episode_id"]: row["title"] for row in episode_rows}
    for candidate in candidates:
        lines.extend([f"## {candidate['candidate_id']}", "",
                      f"{titles[candidate['episode_id']]} | {candidate['start_secs']:.1f}–{candidate['end_secs']:.1f} seconds"
                      f" | {', '.join(candidate['suggested_types'])}", "",
                      "Raw: " + candidate["raw_evidence"], "",
                      "Corrected: " + candidate["corrected_evidence"], "",
                      "Entity suggestions: " + ", ".join(candidate["entity_suggestions"]), ""])
    path.write_text("\n".join(lines))


def load_bundle(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported bundle schema")
    for name in ("episodes.jsonl", "segments.jsonl", "candidates.jsonl"):
        if digest((bundle / name).read_bytes()) != manifest["files"][name]:
            raise ValueError(f"bundle content changed: {name}")
    return manifest


def check_review_item(item, reviewer):
    if not isinstance(item, dict) or not isinstance(item.get("candidate_id"), str):
        raise ValueError("each review must be an object with a candidate_id")
    status = item.get("status")
    if status not in ("pending", "accepted", "rejected"):
        raise ValueError("invalid review status")
    if status != "pending" and not reviewer.strip():
        raise ValueError("reviewer required")
    if item.get("valid_until") is not None:
        check_time(item["valid_until"])
    if status == "accepted":
        if item.get("claim_type") not in KINDS or item.get("certainty") not in CERTAINTIES:
            raise ValueError("invalid claim type or certainty")
        if item.get("temporal_scope") not in ("current", "upcoming", "historical", "unknown"):
            raise ValueError("explicit temporal_scope required")
        if not string(item.get("claim_text", ""), "claim_text").strip() or not string(item.get("notes", ""), "notes").strip():
            raise ValueError("accepted claims require claim_text and review notes")
        entities = item.get("entities")
        if not isinstance(entities, list) or not entities or any(
            not isinstance(e, dict) or e.get("type") not in ("player", "team")
            or not isinstance(e.get("name"), str) or not e["name"].strip()
            or not isinstance(e.get("model_entity_id"), str) or not e["model_entity_id"].strip() for e in entities
        ):
            raise ValueError("accepted claims require explicit player/team model entity IDs")
        if not item.get("valid_until"):
            raise ValueError("accepted claims require an explicit expiry")
        if not isinstance(item.get("audio_checked"), bool):
            raise ValueError("audio_checked must be boolean")
    return status


def finalize(bundle, review_path, out):
    manifest = load_bundle(bundle)
    review = json.loads(review_path.read_text())
    if not isinstance(review, dict) or review.get("manifest_sha256") != digest((bundle / "manifest.json").read_bytes()):
        raise ValueError("review belongs to another bundle")
    reviewer = string(review.get("reviewer", ""), "reviewer")
    candidates = {c["candidate_id"]: c for c in read_rows(bundle / "candidates.jsonl")}
    episodes = {e["episode_id"]: e for e in read_rows(bundle / "episodes.jsonl")}
    segments = {s["segment_id"]: s for s in read_rows(bundle / "segments.jsonl")}
    entries = review.get("reviews")
    if not isinstance(entries, list):
        raise ValueError("reviews must be a list")
    statuses = [check_review_item(item, reviewer) for item in entries]
    if len(entries) != len(candidates) or {r["candidate_id"] for r in entries} != set(candidates):
        raise ValueError("review must contain every candidate exactly once")
    finalized = now()
    if timestamp(finalized) < timestamp(manifest["generated_at"]):
        raise ValueError("finalization precedes bundle creation")
    rows = []
    for item, status in zip(entries, statuses, strict=True):
        candidate = candidates[item["candidate_id"]]
        episode = episodes[candidate["episode_id"]]
        published = episode["published_at"]
        if status == "accepted" and published:
            if timestamp(published) > timestamp(finalized):
                raise ValueError("accepted claim's episode is published after finalization; check the clock")
            if timestamp(item["valid_until"]) <= timestamp(published):
                raise ValueError(f"valid_until must be after publication ({published}): {item['candidate_id']}")
        additional = item.get("additional_segment_ids", [])
        if not isinstance(additional, list) or not all(isinstance(s, str) for s in additional):
            raise ValueError("additional_segment_ids must be a list of segment IDs")
        evidence_ids = set(candidate["segment_ids"] + additional)
        if any(s not in segments or segments[s]["episode_id"] != candidate["episode_id"]
               or segments[s]["transcript_sha256"] != candidate["transcript_sha256"] for s in evidence_ids):
            raise ValueError("review evidence must come from the same frozen episode transcript")
        evidence = sorted((segments[s] for s in evidence_ids), key=lambda s: s["index"])
        rows.append({"schema_version": SCHEMA_VERSION, **candidate,
                     "reviewed_segment_ids": [s["segment_id"] for s in evidence],
                     "reviewed_raw_evidence": " ".join(s["raw_text"].strip() for s in evidence),
                     "reviewed_corrected_evidence": " ".join(s["corrected_text"].strip() for s in evidence),
                     "reviewed_start_secs": min(s["start_secs"] for s in evidence),
                     "reviewed_end_secs": max(s["end_secs"] for s in evidence),
                     "published_at": published, "downloaded_at": episode["downloaded_at"],
                     "transcribed_at": episode["transcribed_at"], "transcript_model": episode.get("model"),
                     "bundle_generated_at": manifest["generated_at"], "reviewed_at": finalized if status != "pending" else None,
                     "model_ready_at": finalized if status == "accepted" else None,
                     "reviewer": review.get("reviewer"), "review": item})
    with new_directory(out) as staging:
        write_rows(staging / "claims.jsonl", rows)
        write_json(staging / "review.json", review)
        write_json(staging / "manifest.json", {"schema_version": SCHEMA_VERSION, "finalized_at": finalized,
                   "source_manifest_sha256": review["manifest_sha256"], "review_sha256": digest(encoded(review)),
                   "claims_sha256": digest((staging / "claims.jsonl").read_bytes()),
                   "counts": {s: sum(r["review"]["status"] == s for r in rows) for s in ("accepted", "rejected", "pending")}})
    return rows


def load_dataset(dataset, expected_manifest_sha256=None):
    """Verify a finalized dataset is internally consistent before any row is trusted.

    Internal checks catch accidental edits. Only a manifest hash recorded outside the dataset
    (for example in the experiment config) catches a consistent rewrite of every file.
    """
    manifest_bytes = (dataset / "manifest.json").read_bytes()
    manifest_sha256 = digest(manifest_bytes)
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("finalized manifest does not match the expected hash")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported dataset schema")
    claims = dataset / "claims.jsonl"
    if digest(claims.read_bytes()) != manifest["claims_sha256"]:
        raise ValueError("finalized claims changed")
    review_bytes = (dataset / "review.json").read_bytes()
    if digest(review_bytes) != manifest["review_sha256"]:
        raise ValueError("finalized review changed")
    review = json.loads(review_bytes)
    if review["manifest_sha256"] != manifest["source_manifest_sha256"]:
        raise ValueError("finalized review belongs to another bundle")
    rows = read_rows(claims)
    items = {item["candidate_id"]: item for item in review["reviews"]}
    finalized = manifest["finalized_at"]
    if len(rows) != len(items) or {r["candidate_id"] for r in rows} != set(items):
        raise ValueError("finalized claims do not match the review")
    for row in rows:
        status = row["review"]["status"]
        if (row["review"] != items[row["candidate_id"]] or row["reviewer"] != review.get("reviewer")
                or row["model_ready_at"] != (finalized if status == "accepted" else None)
                or row["reviewed_at"] != (finalized if status != "pending" else None)):
            raise ValueError(f"finalized claim is inconsistent with its review: {row['candidate_id']}")
    counts = {s: sum(r["review"]["status"] == s for r in rows) for s in ("accepted", "rejected", "pending")}
    if counts != manifest["counts"]:
        raise ValueError("finalized counts changed")
    return manifest_sha256, manifest, rows


def select(dataset, decision_time, out, historical_delay_hours=None, require_audio_checked=False,
           historical_validity_days=None, expected_manifest_sha256=None):
    cutoff = timestamp(decision_time)
    historical = historical_delay_hours is not None
    if historical:
        if not math.isfinite(historical_delay_hours) or not 0 <= historical_delay_hours <= MAX_DELAY_HOURS:
            raise ValueError(f"historical delay must be between 0 and {MAX_DELAY_HOURS} hours")
        # Reviewer expiries were written after the fact and can encode the outcome; replace them
        # with a validity window declared before evaluation.
        if historical_validity_days is None:
            raise ValueError("historical selection requires a predeclared validity window (days)")
        if not math.isfinite(historical_validity_days) or not 0 < historical_validity_days <= MAX_VALIDITY_DAYS:
            raise ValueError(f"historical validity must be between 0 and {MAX_VALIDITY_DAYS} days")
    elif historical_validity_days is not None:
        raise ValueError("a validity window applies only to historical selection")
    manifest_sha256, manifest, rows = load_dataset(dataset, expected_manifest_sha256)
    audit, eligible = [], []
    for row in rows:
        reason, availability, expires = None, row["model_ready_at"], row["review"].get("valid_until")
        if row["review"]["status"] != "accepted":
            reason = "not_accepted"
        elif row["review"].get("temporal_scope") not in ("current", "upcoming"):
            reason = "non_current_claim"
        elif historical:
            availability = expires = None
            if row["published_at"]:
                available = timestamp(row["published_at"]) + timedelta(hours=historical_delay_hours)
                availability = available.isoformat()
                expires = (available + timedelta(days=historical_validity_days)).isoformat()
        if reason is None:
            if availability is None:
                reason = "unknown_availability"
            elif timestamp(availability) >= cutoff:
                reason = "not_available_before_cutoff"
            elif row["published_at"] and timestamp(row["published_at"]) >= cutoff:
                reason = "not_published_before_cutoff"
            elif expires and cutoff >= timestamp(expires):
                reason = "expired"
        if reason is None and require_audio_checked and row["review"].get("audio_checked") is not True:
            reason = "audio_not_verified"
        result = {**row, "decision_time": cutoff.isoformat(), "available_at": availability, "expires_at": expires,
                  "timing_basis": "historical_assumption" if historical else "observed",
                  "expiry_basis": "historical_policy" if historical else "reviewer",
                  "eligible": reason is None, "exclusion_reason": reason}
        audit.append(result)
        if reason is None:
            eligible.append(result)
    with new_directory(out) as staging:
        write_rows(staging / "eligible.jsonl", eligible)
        write_rows(staging / "audit.jsonl", audit)
        write_json(staging / "manifest.json", {"schema_version": SCHEMA_VERSION, "decision_time": cutoff.isoformat(),
                   "source_manifest_sha256": manifest_sha256, "source_claims_sha256": manifest["claims_sha256"],
                   "historical_delay_hours": historical_delay_hours, "historical_validity_days": historical_validity_days,
                   "research_only": historical, "require_audio_checked": require_audio_checked,
                   "eligible": len(eligible), "total": len(audit)})
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    build_parser.add_argument("--feed", required=True)
    build_parser.add_argument("--title", default="")
    build_parser.add_argument("--limit", type=int, default=5)
    build_parser.add_argument("--out", type=Path, required=True)
    build_parser.add_argument("--extractor", choices=["keyword", "llm"], default="keyword",
                              help="llm sends each transcript to --provider to propose claims")
    build_parser.add_argument("--provider", default="anthropic",
                              help="anthropic, openai, deepseek, qwen, moonshot, zhipu or compatible")
    build_parser.add_argument("--model", help="Required for every provider except anthropic")
    build_parser.add_argument("--base-url", help="Override the provider endpoint (required for compatible)")
    build_parser.add_argument("--unprocessed", action="store_true",
                              help="Only transcribed episodes not yet extracted; exits quietly when there are none")
    build_parser.add_argument("--published-after", help="Required with --unprocessed (ISO 8601 with timezone)")
    build_parser.add_argument("--athletes", type=Path,
                              help="athletes.jsonl (athlete_id, name, active) for exact ESPN id mapping")
    finish = commands.add_parser("finalize")
    finish.add_argument("--bundle", type=Path, required=True)
    finish.add_argument("--review", type=Path, required=True)
    finish.add_argument("--out", type=Path, required=True)
    select_parser = commands.add_parser("select")
    select_parser.add_argument("--dataset", type=Path, required=True)
    select_parser.add_argument("--decision-time", required=True)
    select_parser.add_argument("--historical-delay-hours", type=float)
    select_parser.add_argument("--historical-validity-days", type=float,
                               help="Required with --historical-delay-hours; replaces reviewer expiries")
    select_parser.add_argument("--expected-manifest-sha256",
                               help="Pin the finalized dataset; record this hash outside the dataset")
    select_parser.add_argument("--require-audio-checked", action="store_true",
                               help="Exclude claims without human audio verification; ASR cross-checks do not qualify")
    select_parser.add_argument("--out", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "finalize":
        args["review_path"] = args.pop("review")
    if command == "build":
        athletes = args.pop("athletes")
        if args["extractor"] == "llm":
            import llm_claims
            args["catalogue"] = llm_claims.Catalogue.load(athletes)
    try:
        handlers: dict[str, Callable[..., Any]] = {"build": build, "finalize": finalize, "select": select}
        result = handlers[command](**args)
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as error:
        parser.exit(1, f"error: {error}\n")
    if result is None:
        print(json.dumps({"new_episodes": 0}))
        return
    print(json.dumps(result.get("counts", {}) if isinstance(result, dict) else {"records": len(result)}))


if __name__ == "__main__":
    main()
