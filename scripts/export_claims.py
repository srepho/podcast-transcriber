#!/usr/bin/env python3
"""Export every LLM-proposed claim, reviewed or not, as one feed a downstream model can choose from.

    python3 scripts/export_claims.py [--pilots data/pilots] [--out data/exports]

Writes claims_feed.jsonl and claims_feed.manifest.json (row count, SHA-256, counts) atomically.
This is a separate, explicitly unreviewed input: the finalized-bundle contract is unchanged, and
each row says whether a person (or a delegated reviewer) accepted, rejected or has not yet seen it.

Timing: `available_at` is when the pipeline first held the claim (the bundle's extraction time),
not the episode's publication. Backfilled episodes therefore become available when they were
extracted. `published_at` is kept for retrospective research, which must declare its own delay.
`advisory_valid_until` is the extractor's fixed freshness window; nothing is filtered by it here.
Standard library only.
"""
import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

FEED_SCHEMA_VERSION = 1
FEED = "claims_feed.jsonl"
MANIFEST = "claims_feed.manifest.json"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_bundles(pilots):
    """Bundles holding LLM proposals (keyword-hint bundles carry no claims), oldest first."""
    for bundle in sorted(pilots.iterdir()):
        manifest = bundle / "manifest.json"
        if (bundle / "candidates.jsonl").exists() and manifest.exists():
            if str(json.loads(manifest.read_text()).get("extractor_version", "")).startswith("llm-claims"):
                yield bundle


def finalized(pilots, bundle):
    """candidate_id -> frozen claim row from the current finalized version of a bundle, if any."""
    versions = [p for p in pilots.iterdir() if p.is_dir() and p.name.startswith(bundle.name + "-reviewed")
                and (p / "claims.jsonl").exists()]
    if not versions:
        return {}
    latest = max(versions, key=lambda p: json.loads((p / "manifest.json").read_text())["finalized_at"])
    return {row["candidate_id"]: row for row in rows(latest / "claims.jsonl")}


def subject_key(claim_type, entities):
    """Same type and same mapped players (else teams): a grouping hint for duplicates across episodes."""
    ids = sorted({e["model_entity_id"] for e in entities if e.get("model_entity_id")})
    players = [i for i in ids if i.startswith("espn:athlete:")]
    return f"{claim_type}|{','.join(players or ids)}" if ids else None


def feed_rows(pilots):
    out = []
    for bundle in source_bundles(pilots):
        manifest = json.loads((bundle / "manifest.json").read_text())
        episodes = {e["episode_id"]: e for e in rows(bundle / "episodes.jsonl")}
        review = json.loads((bundle / "review.json").read_text())
        reviews = {r["candidate_id"]: r for r in review["reviews"]}
        frozen = finalized(pilots, bundle)
        for candidate in rows(bundle / "candidates.jsonl"):
            item = reviews[candidate["candidate_id"]]
            episode = episodes[candidate["episode_id"]]
            done = frozen.get(candidate["candidate_id"])
            proposal = candidate.get("proposal", {})
            out.append({
                "feed_schema_version": FEED_SCHEMA_VERSION,
                "candidate_id": candidate["candidate_id"], "bundle": bundle.name,
                "episode_id": candidate["episode_id"], "episode_title": episode["title"],
                "published_at": episode["published_at"], "transcribed_at": episode.get("transcribed_at"),
                "available_at": manifest["generated_at"], "extractor_model": candidate.get("extractor_model"),
                "claim_type": item["claim_type"], "certainty": item["certainty"],
                "temporal_scope": item["temporal_scope"], "claim_text": item["claim_text"],
                "entities": item["entities"], "subject_key": subject_key(item["claim_type"], item["entities"]),
                "advisory_valid_until": item.get("valid_until"), "asr_uncertain": bool(proposal.get("asr_uncertain")),
                "review_status": item["status"], "reviewer": review.get("reviewer") or None,
                "review_notes": item.get("notes") or None,
                "finalized_status": done["review"]["status"] if done else None,
                "model_ready_at": done.get("model_ready_at") if done else None,
                "proposal": proposal,
                "start_secs": candidate["start_secs"], "end_secs": candidate["end_secs"],
                "corrected_evidence": candidate["corrected_evidence"], "raw_evidence": candidate["raw_evidence"],
            })
    return sorted(out, key=lambda r: (r["available_at"], r["published_at"] or "", r["candidate_id"]))


def write_atomic(path, data):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def export(pilots, out):
    feed = feed_rows(pilots)
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in feed).encode()
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for r in feed:
        counts[r["review_status"]] = counts.get(r["review_status"], 0) + 1
    manifest = {"feed_schema_version": FEED_SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
                "rows": len(feed), "sha256": hashlib.sha256(data).hexdigest(), "review_status_counts": counts,
                "bundles": sorted({r["bundle"] for r in feed}),
                "timing": "available_at is extraction time; published_at is not availability"}
    write_atomic(out / FEED, data)  # feed first: a reader that sees the new manifest finds the matching feed
    write_atomic(out / MANIFEST, (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilots", type=Path, default=Path("data/pilots"))
    parser.add_argument("--out", type=Path, default=Path("data/exports"))
    args = parser.parse_args()
    manifest = export(args.pilots, args.out)
    print(json.dumps({"rows": manifest["rows"], "review_status_counts": manifest["review_status_counts"]}))


if __name__ == "__main__":
    main()
