#!/usr/bin/env python3
"""Scheduled collection: refresh, download and transcribe new episodes, then have an LLM propose
claims for episodes not yet extracted, into a new pending bundle under data/pilots/auto-<UTC stamp>.
Nothing reaches a model until a person reviews and finalizes the bundle (scripts/review_claims.py).

    .venv/bin/python scripts/daily.py FEED[,FEED...] PUBLISHED_AFTER     e.g. myshow,othershow 2026-10-01T00:00:00Z

Every listed feed is extracted into its own bundle, data/pilots/auto-<feed>-<UTC stamp>; a failure on
one feed does not stop the others.

PROVIDER (default anthropic), MODEL (required except for anthropic) and BASE_URL choose the extractor.
FALLBACK_PROVIDER and FALLBACK_MODEL name a second extractor: when the first fails for any episode (or
the whole feed), the episodes still unprocessed are extracted again with it, into
auto-<feed>-<stamp>-<fallback provider>.
The provider's key comes from its usual environment variable, the login keychain (service
"podcast-<provider>"), or ENV_FILE, a dotenv file from which only that one variable is read.

Run by launchd through the venv's Python rather than /bin/bash: macOS privacy protection grants
folder access per executable, and a bash script under ~/Downloads fails with "Operation not permitted".
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import export_claims  # noqa: E402
import llm_claims  # noqa: E402
import pilot_dataset  # noqa: E402

# The ESPN athlete history, then current rosters (recent rookies the history lacks); both optional.
DEFAULT_ATHLETES = [REPO.parent / "PredictionMarkets/data/basketball/nba/players/athletes.jsonl",
                    REPO / "data/vocab/espn_rosters.jsonl"]
DEFAULT_ALIASES = REPO / "data/vocab/entity_aliases.txt"
MODEL_REPO = REPO.parent / "PredictionMarkets"


def refresh_rosters():
    """Re-export current rosters when the prediction model's copy is newer. Never fatal: a stale
    catalogue only leaves more names for the reviewer to map."""
    python = MODEL_REPO / ".venv/bin/python"
    rosters = MODEL_REPO / "data/basketball/sdv/nba/rosters"
    if not python.exists() or not rosters.is_dir():
        return
    done = subprocess.run([str(python), str(REPO / "scripts/export_rosters.py"), str(rosters),
                           str(DEFAULT_ATHLETES[1])], capture_output=True, text=True, check=False)
    print((done.stdout or done.stderr).strip() or f"roster export exited {done.returncode}", flush=True)


def log(message):
    print(f"== {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {message}", flush=True)


def notify(message):
    script = f"display notification {json.dumps(message)} with title \"Podcast claims\""
    subprocess.run(["osascript", "-e", script], capture_output=True, check=False)


def dotenv_value(path, name):
    """The last assignment of `name` in a dotenv file; nothing else is read into the environment."""
    value = None
    for line in Path(path).read_text().splitlines():
        key, sep, raw = line.strip().removeprefix("export ").partition("=")
        if sep and key.strip() == name:
            value = raw.strip().strip("\"'")
    return value or None


def load_key(provider, env_file=None):
    """Put the provider's key in the environment. Returns whether one was found."""
    name = llm_claims.PROVIDERS[provider]["key_env"]
    if os.environ.get(name):
        return True
    found = subprocess.run(["security", "find-generic-password", "-s", f"podcast-{provider}", "-w"],
                           capture_output=True, text=True, check=False)
    value = found.stdout.strip() if found.returncode == 0 else ""
    if not value and env_file and Path(env_file).exists():
        value = dotenv_value(env_file, name) or ""
    if value:
        os.environ[name] = value
    return bool(value)


def export_feed():
    """Refresh the unreviewed claims feed (data/exports). Never fatal: the bundles are the record."""
    try:
        manifest = export_claims.export(REPO / "data/pilots", REPO / "data/exports")
        log(f"claims feed: {manifest['rows']} rows {manifest['review_status_counts']}")
    except Exception as error:
        print(f"claims feed export failed: {type(error).__name__}: {error}", file=sys.stderr)


def extract_feed(feed, stamp, extractors, published_after, catalogue, build=None):
    """Extract a feed's unprocessed episodes with the first extractor; while episodes fail (or the whole
    build does), retry what is still unprocessed with the next. Returns (all succeeded, summary lines)."""
    build = build or pilot_dataset.build
    lines: list[str] = []
    for attempt, (provider, model, base_url) in enumerate(extractors):
        suffix = "" if attempt == 0 else f"-{provider}"
        out = REPO / "data/pilots" / f"auto-{feed}-{stamp}{suffix}"
        log(f"extract {feed} with {provider}:{model or 'default'} -> {out.relative_to(REPO)}")
        try:
            manifest = build(REPO / "data", feed, "", 10, out, extractor="llm", unprocessed=True,
                             published_after=published_after, catalogue=catalogue,
                             provider=provider, model=model, base_url=base_url)
        except Exception as error:  # the log and a notification are the only places a scheduled failure shows
            print(f"extraction failed for {feed} ({provider}): {type(error).__name__}: {error}", file=sys.stderr)
            continue
        if manifest is None:
            if attempt == 0:
                log(f"{feed}: no new episodes")
            return True, lines
        lines.append(f"{feed}: {summary(manifest)}" + (f" via {provider}" if attempt else ""))
        log(lines[-1])
        if not manifest["extraction_failures"]:
            return True, lines
    return False, lines


def summary(manifest):
    failed = len(manifest["extraction_failures"])
    counts = manifest["counts"]
    return (f"{counts['candidates.jsonl']} claims from {counts['episodes.jsonl']} episodes"
            + (f", {failed} failed" if failed else ""))


def main(argv=None):
    try:
        return run(argv)
    finally:
        export_feed()  # every run, so review decisions reach the feed even when nothing new arrived


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("feeds", help="Comma-separated feed names")
    parser.add_argument("published_after")
    parser.add_argument("--provider", default=os.environ.get("PROVIDER", "anthropic"))
    parser.add_argument("--model", default=os.environ.get("MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL"))
    parser.add_argument("--env-file", default=os.environ.get("ENV_FILE"))
    parser.add_argument("--fallback-provider", default=os.environ.get("FALLBACK_PROVIDER"))
    parser.add_argument("--fallback-model", default=os.environ.get("FALLBACK_MODEL"))
    parser.add_argument("--athletes", type=Path, action="append",
                        help="Catalogue JSONL; repeatable (default: ESPN athletes, then data/vocab/espn_rosters.jsonl)")
    parser.add_argument("--entity-aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--skip-collect", action="store_true", help="Only extract; do not run `podcast run`")
    args = parser.parse_args(argv)
    os.chdir(REPO)  # transcript paths in the database are relative to the repository

    if not args.skip_collect:
        log("collect")
        # A failed refresh or download must not block extraction of episodes already transcribed.
        code = subprocess.run([str(REPO / "target/release/podcast"), "run"], check=False).returncode
        if code:
            print(f"podcast run failed (exit {code}); continuing with what is transcribed", file=sys.stderr)
            notify(f"Collection failed (exit {code}); see ~/Library/Logs/podcast/daily.log")

    if not load_key(args.provider, args.env_file):
        print(f"no {args.provider} API key; skipping extraction", file=sys.stderr)
        notify(f"No {args.provider} API key; claims not extracted")
        return 1
    if not args.athletes:
        refresh_rosters()
    catalogue = llm_claims.Catalogue.load(args.athletes or DEFAULT_ATHLETES, args.entity_aliases)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    extractors = [(args.provider, args.model, args.base_url)]
    if args.fallback_provider:
        if load_key(args.fallback_provider, args.env_file):
            extractors.append((args.fallback_provider, args.fallback_model, None))
        else:
            print(f"no {args.fallback_provider} API key; running without a fallback", file=sys.stderr)
    failed, made = [], []
    for feed in [f.strip() for f in args.feeds.split(",") if f.strip()]:
        ok, lines = extract_feed(feed, stamp, extractors, args.published_after, catalogue)
        made += lines
        if not ok:
            failed.append(feed)
    if failed:
        notify(f"Claim extraction failed for {', '.join(failed)}; see ~/Library/Logs/podcast/daily.log")
    if made:
        notify("; ".join(made) + " (scripts/review_claims.py data/pilots/auto-* --finalize)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
