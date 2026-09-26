#!/usr/bin/env python3
"""Local research bundles: build -> edit review.json -> finalize -> select.

Standard library only. Hints are review candidates, never automatic factual claims.
"""
from contextlib import closing
import argparse
import hashlib
import json
import math
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
CERTAINTIES = {"reported_fact", "opinion", "speculation", "conditional", "unknown"}


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
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


def new_directory(path):
    # A new directory per run/version prevents silent replacement of research evidence.
    path.mkdir(parents=True, exist_ok=False)


def check_time(value):
    if value is not None:
        timestamp(value)
    return value


def build(data_dir, feed, title, limit, out):
    if not 1 <= limit <= 10:
        raise ValueError("pilot limit must be 1..10")
    generated = now()
    with closing(sqlite3.connect(data_dir.joinpath("podcast.db").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        # Select the cohort before checking transcription availability; missing records stay visible.
        episodes = [dict(r) for r in db.execute(
            "SELECT * FROM episodes WHERE feed_name=? ORDER BY published DESC, guid", (feed,)
        ) if title.casefold() in r["title"].casefold()][:limit]
    if not episodes:
        raise ValueError("no matching episodes")
    terms = []
    vocab = data_dir / "vocab" / f"{feed}.txt"
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
                raw = segment.get("raw_text")
                segments.append({"schema_version": SCHEMA_VERSION, "episode_id": episode_id,
                                 "segment_id": f"{episode_id}:{index}", "index": index,
                                 "start_secs": start, "end_secs": end,
                                 "raw_text": raw if raw is not None else segment["text"],
                                 "corrected_text": segment["text"],
                                 "transcript_sha256": row["transcript_sha256"]})
                kinds = [kind for kind, pattern in KINDS.items() if re.search(pattern, segment["text"], re.I)]
                if not kinds:
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
                    "uncertainty_markers": sorted(set(re.findall(r"\b(?:if|might|may|could|would|likely|reportedly|expect|think)\b", text.lower()))),
                    "extractor_version": EXTRACTOR_VERSION})
                reviews.append({"candidate_id": candidate_id, "status": "pending", "claim_type": kinds[0],
                    "claim_text": "", "certainty": "unknown", "temporal_scope": "unknown", "entities": [], "valid_until": None,
                    "audio_checked": False, "notes": ""})
        episode_rows.append(row)
    new_directory(out)
    files = {"episodes.jsonl": episode_rows, "segments.jsonl": segments, "candidates.jsonl": candidates}
    for name, rows in files.items():
        write_rows(out / name, rows)
    manifest = {"schema_version": SCHEMA_VERSION, "extractor_version": EXTRACTOR_VERSION,
                "generated_at": generated, "sampling": {"feed": feed, "title_contains": title,
                "requested": limit, "selected": len(episodes), "order": "newest_publication_first"},
                "counts": {name: len(rows) for name, rows in files.items()},
                "files": {name: digest((out / name).read_bytes()) for name in files},
                "availability_policy": "reviewed records become available when finalized; publication is not availability"}
    write_json(out / "manifest.json", manifest)
    write_json(out / "review.json", {"schema_version": SCHEMA_VERSION,
               "manifest_sha256": digest((out / "manifest.json").read_bytes()),
               "reviewer": "", "reviews": reviews})
    lines = ["# Pilot evidence review", "", "Edit review.json; each candidate starts pending. Suggestions are not verified facts.",
             "Check the audio around each timestamp, resolve entity IDs, and set an expiry before accepting.",
             "Read all segments.jsonl to look for missed claims; keyword hints do not measure recall.", ""]
    titles = {row["episode_id"]: row["title"] for row in episode_rows}
    for candidate in candidates:
        lines.extend([f"## {candidate['candidate_id']}", "",
                      f"{titles[candidate['episode_id']]} | {candidate['start_secs']:.1f}–{candidate['end_secs']:.1f} seconds | {', '.join(candidate['suggested_types'])}", "",
                      "Raw: " + candidate["raw_evidence"], "",
                      "Corrected: " + candidate["corrected_evidence"], "",
                      "Entity suggestions: " + ", ".join(candidate["entity_suggestions"]), ""])
    (out / "REVIEW.md").write_text("\n".join(lines))
    return manifest


def load_bundle(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported bundle schema")
    for name in ("episodes.jsonl", "segments.jsonl", "candidates.jsonl"):
        if digest((bundle / name).read_bytes()) != manifest["files"][name]:
            raise ValueError(f"bundle content changed: {name}")
    return manifest


def finalize(bundle, review_path, out):
    manifest = load_bundle(bundle)
    review = json.loads(review_path.read_text())
    if review.get("manifest_sha256") != digest((bundle / "manifest.json").read_bytes()):
        raise ValueError("review belongs to another bundle")
    candidates = {c["candidate_id"]: c for c in read_rows(bundle / "candidates.jsonl")}
    episodes = {e["episode_id"]: e for e in read_rows(bundle / "episodes.jsonl")}
    segments = {s["segment_id"]: s for s in read_rows(bundle / "segments.jsonl")}
    entries = review["reviews"]
    if len(entries) != len(candidates) or {r["candidate_id"] for r in entries} != set(candidates):
        raise ValueError("review must contain every candidate exactly once")
    finalized = now()
    if timestamp(finalized) < timestamp(manifest["generated_at"]):
        raise ValueError("finalization precedes bundle creation")
    rows = []
    for item in entries:
        status = item["status"]
        if status not in ("pending", "accepted", "rejected"):
            raise ValueError("invalid review status")
        if status != "pending" and not str(review.get("reviewer", "")).strip():
            raise ValueError("reviewer required")
        if status == "accepted":
            if item.get("claim_type") not in KINDS or item.get("certainty") not in CERTAINTIES:
                raise ValueError("invalid claim type or certainty")
            if item.get("temporal_scope") not in ("current", "upcoming", "historical", "unknown"):
                raise ValueError("explicit temporal_scope required")
            if not item.get("claim_text", "").strip() or not item.get("notes", "").strip():
                raise ValueError("accepted claims require claim_text and review notes")
            if not item.get("entities") or any(
                e.get("type") not in ("player", "team") or not e.get("name", "").strip()
                or not e.get("model_entity_id", "").strip() for e in item["entities"]
            ):
                raise ValueError("accepted claims require explicit player/team model entity IDs")
            if not item.get("valid_until"):
                raise ValueError("accepted claims require an explicit expiry")
            if not isinstance(item.get("audio_checked"), bool):
                raise ValueError("audio_checked must be boolean")
        check_time(item.get("valid_until"))
        candidate = candidates[item["candidate_id"]]
        episode = episodes[candidate["episode_id"]]
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
                     "published_at": episode["published_at"], "downloaded_at": episode["downloaded_at"],
                     "transcribed_at": episode["transcribed_at"], "transcript_model": episode.get("model"),
                     "bundle_generated_at": manifest["generated_at"], "reviewed_at": finalized if status != "pending" else None,
                     "model_ready_at": finalized if status == "accepted" else None,
                     "reviewer": review.get("reviewer"), "review": item})
    new_directory(out)
    write_rows(out / "claims.jsonl", rows)
    write_json(out / "review.json", review)
    write_json(out / "manifest.json", {"schema_version": SCHEMA_VERSION, "finalized_at": finalized,
               "source_manifest_sha256": review["manifest_sha256"], "review_sha256": digest(encoded(review)),
               "claims_sha256": digest((out / "claims.jsonl").read_bytes()),
               "counts": {s: sum(r["review"]["status"] == s for r in rows) for s in ("accepted", "rejected", "pending")}})
    return rows


def select(dataset, decision_time, out, historical_delay_hours=None, require_audio_checked=False):
    cutoff = timestamp(decision_time)
    manifest = json.loads((dataset / "manifest.json").read_text())
    claims = dataset / "claims.jsonl"
    if digest(claims.read_bytes()) != manifest["claims_sha256"]:
        raise ValueError("finalized claims changed")
    if historical_delay_hours is not None and (not math.isfinite(historical_delay_hours) or historical_delay_hours < 0):
        raise ValueError("historical delay must be finite and nonnegative")
    audit, eligible = [], []
    for row in read_rows(claims):
        reason, availability = None, row["model_ready_at"]
        if row["review"]["status"] != "accepted":
            reason = "not_accepted"
        elif row["review"].get("temporal_scope") not in ("current", "upcoming"):
            reason = "non_current_claim"
        elif historical_delay_hours is not None:
            availability = (timestamp(row["published_at"]) + timedelta(hours=historical_delay_hours)).isoformat() if row["published_at"] else None
        if reason is None:
            if availability is None:
                reason = "unknown_availability"
            elif timestamp(availability) >= cutoff:
                reason = "not_available_before_cutoff"
            elif row["published_at"] and timestamp(row["published_at"]) >= cutoff:
                reason = "not_published_before_cutoff"
            elif row["review"].get("valid_until") and cutoff >= timestamp(row["review"]["valid_until"]):
                reason = "expired"
        if reason is None and require_audio_checked and row["review"].get("audio_checked") is not True:
            reason = "audio_not_verified"
        result = {**row, "decision_time": cutoff.isoformat(), "available_at": availability,
                  "timing_basis": "historical_assumption" if historical_delay_hours is not None else "observed",
                  "eligible": reason is None, "exclusion_reason": reason}
        audit.append(result)
        if reason is None:
            eligible.append(result)
    new_directory(out)
    write_rows(out / "eligible.jsonl", eligible)
    write_rows(out / "audit.jsonl", audit)
    write_json(out / "manifest.json", {"schema_version": SCHEMA_VERSION, "decision_time": cutoff.isoformat(),
               "source_claims_sha256": manifest["claims_sha256"], "historical_delay_hours": historical_delay_hours,
               "research_only": historical_delay_hours is not None, "require_audio_checked": require_audio_checked,
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
    finish = commands.add_parser("finalize")
    finish.add_argument("--bundle", type=Path, required=True)
    finish.add_argument("--review", type=Path, required=True)
    finish.add_argument("--out", type=Path, required=True)
    select_parser = commands.add_parser("select")
    select_parser.add_argument("--dataset", type=Path, required=True)
    select_parser.add_argument("--decision-time", required=True)
    select_parser.add_argument("--historical-delay-hours", type=float)
    select_parser.add_argument("--require-audio-checked", action="store_true",
                               help="Exclude claims without human audio verification; ASR cross-checks do not qualify")
    select_parser.add_argument("--out", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "finalize":
        args["review_path"] = args.pop("review")
    try:
        result = {"build": build, "finalize": finalize, "select": select}[command](**args)
    except (ValueError, KeyError, OSError, sqlite3.Error) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result.get("counts", {}) if isinstance(result, dict) else {"records": len(result)}))


if __name__ == "__main__":
    main()
