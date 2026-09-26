#!/usr/bin/env python3
"""Extract two-word names from a nested JSON stats dump for `podcast vocab import`.

Looks for string values under keys named like "name" or "player" that look like
"First Last". Works on most per-season player stat exports.

Usage: python3 scripts/ctg_names.py players.json > names.txt
"""
import json
import re
import sys

NAME_RE = re.compile(r"^[A-Z][A-Za-z'.\- ]+ [A-Z][A-Za-z'.\- ]+$")


def walk(obj, names):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and re.search(r"(player|name)", key, re.I) and NAME_RE.match(value):
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
