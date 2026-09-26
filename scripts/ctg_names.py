#!/usr/bin/env python3
"""Extract two-word names from a nested JSON stats dump for `podcast vocab import`.

Looks for string values under player-name keys (see NAME_KEYS) that look like
"First Last", including accented names. Works on most per-season player stat exports.

Usage: python3 scripts/ctg_names.py players.json > names.txt
"""
import json
import re
import sys

# Unicode letters, so "Nikola Jokić" and "Luka Dončić" are kept.
NAME_RE = re.compile(r"^[^\W\d_]+(?:[ '.\-]+[^\W\d_]+)+\.?$")
# Exact keys: a substring match on "name" would also collect team_name, arena_name, ...
NAME_KEYS = {"name", "player", "player_name", "playername", "full_name", "display_name"}


def is_name(value: str) -> bool:
    value = value.strip()
    return bool(NAME_RE.match(value)) and value[0].isupper() and " " in value


def walk(obj, names):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and key.casefold() in NAME_KEYS and is_name(value):
                names.add(value.strip())
            else:
                walk(value, names)
    elif isinstance(obj, list):
        for item in obj:
            walk(item, names)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with open(sys.argv[1]) as fh:
        data = json.load(fh)
    names: set[str] = set()
    walk(data, names)
    sys.stdout.write("\n".join(sorted(names)) + "\n")
    print(f"{len(names)} names", file=sys.stderr)


if __name__ == "__main__":
    main()
