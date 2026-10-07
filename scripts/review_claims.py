#!/usr/bin/env python3
"""Review pending claim proposals in a pilot bundle from the terminal, then optionally finalize.

    python3 scripts/review_claims.py data/pilots/auto-20261008T0600Z [--reviewer NAME] [--finalize]

Each pending entry shows the proposal, its transcript evidence, the mapped model entity IDs and
related claims already decided (same type and same players, in this or any other bundle).
Unmapped entities are re-resolved against the current catalogue when the bundle is opened.
Commands:  a accept | r reject | d reject as duplicate of the closest related claim | s skip | q quit
           t TEXT  replace claim text        c CERTAINTY   set certainty
           x DAYS  expiry = publication + DAYS              i N ID   set entity N's model id (e.g. espn:athlete:1966)
review.json is rewritten after every decision. Accepting records a transcript-only review
(audio_checked stays false); the pilot's acceptance rules still apply at finalize.
"""
import argparse
import difflib
import importlib.util
import json
import os
import sys
import tempfile
import textwrap
from datetime import timedelta
from pathlib import Path

_spec = importlib.util.spec_from_file_location("pilot_dataset", Path(__file__).with_name("pilot_dataset.py"))
assert _spec and _spec.loader
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)

ACCEPT_NOTE = "Accepted a model-proposed claim after reading the transcript evidence; audio not checked."
REJECT_NOTE = "Rejected model-proposed claim."
DECIDED = ("accepted", "rejected")


def remap(data, catalogue):
    """Fill unmapped entities of pending entries from the catalogue. Returns how many were filled."""
    filled = 0
    for item in data["reviews"]:
        if item["status"] != "pending":
            continue
        for entity in item.get("entities", []):
            if not entity.get("model_entity_id"):
                entity_id = catalogue.resolve(entity["type"], entity["name"])
                if entity_id:
                    entity["model_entity_id"] = entity_id
                    filled += 1
    return filled


def subject(item):
    """What a claim is about: its mapped players, or its mapped teams when it names no player."""
    ids = {e.get("model_entity_id") for e in item.get("entities", []) if e.get("model_entity_id")}
    players = {i for i in ids if i.startswith("espn:athlete:")}
    return frozenset(players or ids)


def related(item, decided, limit=2):
    """Decided claims with the same type and subject, most similar text first."""
    key = (item.get("claim_type"), subject(item))
    if not key[1]:
        return []
    matches = [d for d in decided if d["candidate_id"] != item["candidate_id"]
               and (d.get("claim_type"), subject(d)) == key]
    ratio = lambda d: difflib.SequenceMatcher(None, item["claim_text"].lower(), d["claim_text"].lower()).ratio()  # noqa: E731
    return sorted(matches, key=ratio, reverse=True)[:limit]


def decided_elsewhere(bundle):
    """Decided entries from every other bundle beside this one, once per candidate."""
    seen, out = set(), []
    for path in sorted(bundle.parent.glob("*/review.json")):
        if path.parent == bundle:
            continue
        for item in json.loads(path.read_text()).get("reviews", []):
            if item.get("status") in DECIDED and item["candidate_id"] not in seen:
                seen.add(item["candidate_id"])
                out.append(item)
    return out


def save(path, review):
    """Atomic rewrite so an interrupted session never leaves half a review file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".review-", suffix=".json")
    with os.fdopen(fd, "wb") as f:
        f.write(pilot.encoded(review))
    os.replace(tmp, path)


def missing_for_accept(item):
    """Why an entry cannot be accepted yet (empty when it can)."""
    problems = []
    if not item.get("claim_text", "").strip():
        problems.append("claim text")
    if not item.get("entities"):
        problems.append("at least one entity")
    problems += [f"model id for {e['name']}" for e in item.get("entities", []) if not e.get("model_entity_id")]
    if not item.get("valid_until"):
        problems.append("expiry (x DAYS)")
    return problems


def apply(item, command, published, reviewer, similar=()):
    """Apply one command to a review entry. Returns (done, message)."""
    verb, _, rest = command.strip().partition(" ")
    if verb == "d":
        if not similar:
            return False, "no related decided claim to mark this a duplicate of"
        original = similar[0]
        item.update(status="rejected", notes=f"Duplicate of {original['candidate_id']}: {original['claim_text']}")
        return True, "rejected as duplicate"
    if verb == "a":
        problems = missing_for_accept(item)
        if problems:
            return False, "cannot accept, needs: " + ", ".join(problems)
        item.update(status="accepted", notes=item.get("notes") or ACCEPT_NOTE)
        return True, "accepted"
    if verb == "r":
        item.update(status="rejected", notes=item.get("notes") or REJECT_NOTE)
        return True, "rejected"
    if verb == "t" and rest.strip():
        item["claim_text"] = rest.strip()
        return False, "claim text updated"
    if verb == "c" and rest.strip() in pilot.CERTAINTIES:
        item["certainty"] = rest.strip()
        return False, "certainty updated"
    if verb == "x" and published:
        try:
            days = float(rest)
        except ValueError:
            return False, "x needs a number of days"
        if days <= 0:
            return False, "expiry must be after publication"
        item["valid_until"] = (pilot.timestamp(published) + timedelta(days=days)).isoformat()
        return False, f"expires {item['valid_until']}"
    if verb == "i":
        index, _, entity_id = rest.strip().partition(" ")
        try:
            entity = item["entities"][int(index)]
        except (ValueError, IndexError):
            return False, "i needs an entity number and an id"
        if not entity_id.startswith(("espn:athlete:", "espn:team:")):
            return False, "ids look like espn:athlete:<id> or espn:team:<id>"
        entity["model_entity_id"] = entity_id
        return False, f"{entity['name']} -> {entity_id}"
    return False, "unknown command"


def show(item, candidate, episode, position, total, similar=()):
    start = int(candidate["start_secs"])
    proposal = candidate.get("proposal", {})
    print(f"\n[{position}/{total}] {episode['title']} @ {start // 60}:{start % 60:02d}")
    print(f"  {item['claim_type']} | {item['certainty']} | {item['temporal_scope']}"
          + ("  | ASR UNCERTAIN" if proposal.get("asr_uncertain") else ""))
    print("  Claim: " + item["claim_text"])
    for n, e in enumerate(item.get("entities", [])):
        print(f"  [{n}] {e['type']} {e['name']}: {e.get('model_entity_id') or 'UNMAPPED'}")
    print(f"  Expires: {item.get('valid_until')}")
    print(textwrap.indent(textwrap.fill(candidate["corrected_evidence"], 100), "  > "))
    for other in similar:
        print(f"  Related [{other['status']}]: {other['claim_text']}")


def review(bundle, reviewer, read=input, catalogue=None):
    review_path = bundle / "review.json"
    data = json.loads(review_path.read_text())
    data["reviewer"] = data.get("reviewer") or reviewer
    if catalogue is not None:
        filled = remap(data, catalogue)
        if filled:
            print(f"mapped {filled} more entities from the current catalogue")
            save(review_path, data)
    decided = decided_elsewhere(bundle) + [r for r in data["reviews"] if r["status"] in DECIDED]
    candidates = {c["candidate_id"]: c for c in pilot.read_rows(bundle / "candidates.jsonl")}
    episodes = {e["episode_id"]: e for e in pilot.read_rows(bundle / "episodes.jsonl")}
    pending = [r for r in data["reviews"] if r["status"] == "pending"]
    for position, item in enumerate(pending, 1):
        candidate = candidates[item["candidate_id"]]
        episode = episodes[candidate["episode_id"]]
        similar = related(item, decided)
        show(item, candidate, episode, position, len(pending), similar)
        while True:
            command = read("  > ").strip()
            if command == "q":
                save(review_path, data)
                return data
            if command == "s":
                break
            done, message = apply(item, command, episode["published_at"], reviewer, similar)
            print("  " + message)
            if done:
                decided.append(item)
                save(review_path, data)
                break
    save(review_path, data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--reviewer", default=os.environ.get("USER", ""))
    parser.add_argument("--finalize", action="store_true", help="Freeze the review into <bundle>-reviewed afterwards")
    parser.add_argument("--athletes", type=Path, action="append", help="Catalogue JSONL for re-mapping; repeatable")
    parser.add_argument("--entity-aliases", type=Path)
    args = parser.parse_args()
    catalogue = None
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import daily  # default catalogue locations
        import llm_claims

        catalogue = llm_claims.Catalogue.load(args.athletes or daily.DEFAULT_ATHLETES,
                                              args.entity_aliases or daily.DEFAULT_ALIASES)
    except ImportError:
        pass  # re-mapping is a convenience; review works without it
    data = review(args.bundle, args.reviewer, catalogue=catalogue)
    counts = {s: sum(r["status"] == s for r in data["reviews"]) for s in ("accepted", "rejected", "pending")}
    print(json.dumps(counts))
    if args.finalize:
        out = args.bundle.with_name(args.bundle.name + "-reviewed")
        try:
            pilot.finalize(args.bundle, args.bundle / "review.json", out)
        except (ValueError, OSError) as error:
            sys.exit(f"finalize failed: {error}")
        print(f"finalized -> {out}")


if __name__ == "__main__":
    main()
