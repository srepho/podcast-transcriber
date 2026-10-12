#!/usr/bin/env python3
"""Export current NBA rosters (sportsdataverse parquet) to the JSONL catalogue claim mapping reads.

    <python with pandas> scripts/export_rosters.py ROSTERS_DIR OUT.jsonl

Uses the newest rosters_<season>.parquet in ROSTERS_DIR and rewrites OUT atomically, only when that
file is newer than OUT. Needs pandas and pyarrow, so daily.py runs it with the prediction model's
Python; the podcast tooling itself stays free of them.
"""
import json
import os
import sys
import tempfile
from pathlib import Path


def newest(rosters_dir: Path) -> Path | None:
    files = sorted(rosters_dir.glob("rosters_*.parquet"), key=lambda p: p.stem.split("_")[-1])
    return files[-1] if files else None


def export(rosters_dir: Path, out: Path) -> int | None:
    """Rows written, or None when OUT is already up to date (or there is nothing to export)."""
    source = newest(rosters_dir)
    if source is None or (out.exists() and out.stat().st_mtime >= source.stat().st_mtime):
        return None
    import pandas as pd  # type: ignore[import-untyped]  # runs under the prediction model's Python

    rows = pd.read_parquet(source, columns=["athlete_id", "full_name"]).drop_duplicates("athlete_id")
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=".rosters-", suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for row in rows.itertuples(index=False):
            f.write(json.dumps({"athlete_id": str(row.athlete_id), "name": row.full_name, "active": True}) + "\n")
    os.replace(tmp, out)
    return len(rows)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    written = export(Path(sys.argv[1]), Path(sys.argv[2]))
    print("rosters up to date" if written is None else f"exported {written} roster players")
