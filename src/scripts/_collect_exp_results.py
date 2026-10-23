#!/usr/bin/env python3
"""Parse autodl-tmp experiment result_*.txt files into CSV rows."""

from __future__ import annotations

import csv
import re
from pathlib import Path

DIR_RE = re.compile(
    r"^(?P<setting>.+)_his(?P<his>\d+)d_pred(?P<horizon>\d+)d_sd(?P<sd>\d+)_"
    r"(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$"
)
RESULT_RE = re.compile(
    r"^\[(?P<event>[^\]]+)\]\s+TEST_at_VAL_best\b.*?forecast:\s*"
    r"rmse=(?P<rmse>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*"
    r"mae=(?P<mae>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.MULTILINE,
)

SKIP_DIR_PARTS = {".ipynb_checkpoints", "archive", ".Trash-0"}


def iter_run_dirs(exp_root: Path):
    for path in sorted(exp_root.iterdir()):
        if not path.is_dir():
            continue
        if path.name in SKIP_DIR_PARTS:
            continue
        match = DIR_RE.match(path.name)
        if match:
            yield path, match.groupdict()


def parse_result_file(path: Path):
    text = path.read_text(encoding="utf-8")
    match = RESULT_RE.search(text)
    if not match:
        return None
    return {
        "event": match.group("event"),
        "rmse": float(match.group("rmse")),
        "mae": float(match.group("mae")),
    }


def collect_rows(exp_root: Path, settings=None):
    """Return one row per (setting, event, horizon, sd), keeping the latest run."""
    latest = {}
    skipped = []

    for run_dir, meta in iter_run_dirs(exp_root):
        setting = meta["setting"]
        if settings is not None and setting not in settings:
            continue

        for result_path in sorted(run_dir.glob("event*/result_*.txt")):
            if any(part in SKIP_DIR_PARTS for part in result_path.parts):
                continue
            parsed = parse_result_file(result_path)
            if parsed is None:
                skipped.append(str(result_path))
                continue
            key = (setting, parsed["event"], int(meta["horizon"]), int(meta["sd"]))
            row = {
                "setting": setting,
                "event": parsed["event"],
                "horizon": int(meta["horizon"]),
                "sd": int(meta["sd"]),
                "rmse": parsed["rmse"],
                "mae": parsed["mae"],
                "_ts": meta["ts"],
                "_run": run_dir.name,
            }
            prev = latest.get(key)
            if prev is None or meta["ts"] >= prev["_ts"]:
                latest[key] = row

    rows = [latest[k] for k in sorted(latest)]
    return rows, skipped


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
