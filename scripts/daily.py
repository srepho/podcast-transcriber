#!/usr/bin/env python3
"""Scheduled collection: refresh, download and transcribe new episodes, then have an LLM propose
claims for episodes not yet extracted, into a new pending bundle under data/pilots/auto-<UTC stamp>.
Nothing reaches a model until a person reviews and finalizes the bundle (scripts/review_claims.py).

    .venv/bin/python scripts/daily.py FEED PUBLISHED_AFTER     e.g. myshow 2026-10-01T00:00:00Z

PROVIDER (default anthropic), MODEL (required except for anthropic) and BASE_URL choose the extractor.
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
import llm_claims  # noqa: E402
import pilot_dataset  # noqa: E402

# The ESPN athlete history, then current rosters (recent rookies the history lacks); both optional.
DEFAULT_ATHLETES = [REPO.parent / "PredictionMarkets/data/basketball/nba/players/athletes.jsonl",
                    REPO / "data/vocab/espn_rosters.jsonl"]
DEFAULT_ALIASES = REPO / "data/vocab/entity_aliases.txt"


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


def summary(manifest):
    failed = len(manifest["extraction_failures"])
    counts = manifest["counts"]
    return (f"{counts['candidates.jsonl']} claims from {counts['episodes.jsonl']} episodes"
            + (f", {failed} failed" if failed else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("feed")
    parser.add_argument("published_after")
    parser.add_argument("--provider", default=os.environ.get("PROVIDER", "anthropic"))
    parser.add_argument("--model", default=os.environ.get("MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL"))
    parser.add_argument("--env-file", default=os.environ.get("ENV_FILE"))
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
    out = REPO / "data/pilots" / f"auto-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%MZ')}"
    log(f"extract with {args.provider}:{args.model or 'default'} -> {out.relative_to(REPO)}")
    try:
        manifest = pilot_dataset.build(
            REPO / "data", args.feed, "", 10, out, extractor="llm", unprocessed=True,
            published_after=args.published_after,
            catalogue=llm_claims.Catalogue.load(args.athletes or DEFAULT_ATHLETES, args.entity_aliases),
            provider=args.provider, model=args.model, base_url=args.base_url)
    except Exception as error:  # the log and a notification are the only places a scheduled failure shows
        print(f"extraction failed: {type(error).__name__}: {error}", file=sys.stderr)
        notify("Claim extraction failed; see ~/Library/Logs/podcast/daily.log")
        return 1
    if manifest is None:
        log("no new episodes")
        return 0
    text = summary(manifest)
    log(text)
    notify(f"{text} to review: scripts/review_claims.py {out.relative_to(REPO)} --finalize")
    return 0


if __name__ == "__main__":
    sys.exit(main())
