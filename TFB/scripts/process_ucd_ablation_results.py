#!/usr/bin/env python3
"""Aggregate UCDGPT ablation result files for ucd-d24 / ucd-d48.

Writes per-horizon CSVs under each results directory and a merged
``ablation_results_avg.csv`` that averages MAE/RMSE across the requested
horizons for each (event, variant) pair.

Usage:
    python TFB/scripts/process_ucd_ablation_results.py
    python TFB/scripts/process_ucd_ablation_results.py --horizons d24 d48
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "TFB" / "results"
DEFAULT_HORIZONS = ("d24", "d48")
DEFAULT_MERGED_OUTPUT = RESULTS_ROOT / "ucd-ablation" / "ablation_results_avg.csv"

EVENT_PATTERN = re.compile(r"event(?P<event>\d+)$")
RESULT_PATTERN = re.compile(
    r"\[(?P<event>event\d+)\]\s+"
    r"(?P<status>TEST_best|VAL_best)\s+epoch:(?P<epoch>\d+)\s+\|\s+"
    r"forecast_full:\s*rmse=(?P<rmse>[-+0-9.eE]+),\s*"
    r"mae=(?P<mae>[-+0-9.eE]+)\s*\|\s*train_time:(?P<train_time>[-+0-9.eE]+|N/A)min"
)

# Short keys match ``src/figure/ablation_metric_curves.py``.
VARIANT_KEYS = ("full", "no_contra", "no_temporal", "no_spatial", "no_random")
VARIANT_DISPLAY = {
    "full": "MetroMAE (full)",
    "no_contra": "w/o contrastive loss",
    "no_temporal": "w/o temporal meta mask",
    "no_spatial": "w/o spatial meta mask",
    "no_random": "w/o random base mask",
}


def variant_key(directory_name: str) -> str:
    """Map experiment directory names to short ablation keys."""
    if "mscomb" in directory_name and "cw0p5" in directory_name:
        return "full"
    if "mscomb" in directory_name and "cw0_" in directory_name:
        return "no_contra"
    if "msnrand" in directory_name:
        return "no_random"
    if "msnspat" in directory_name:
        return "no_spatial"
    if "msntemp" in directory_name:
        return "no_temporal"
    raise ValueError(f"Unrecognized UCD ablation directory: {directory_name}")


def horizon_results_dir(horizon: str) -> Path:
    path = RESULTS_ROOT / f"ucd-{horizon}"
    if not path.is_dir():
        raise FileNotFoundError(f"Missing results directory: {path}")
    return path


def collect_horizon_rows(results_dir: Path, horizon: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    directories = sorted(
        path
        for path in results_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if len(directories) != 5:
        raise ValueError(
            f"Expected 5 ablation directories under {results_dir}, found {len(directories)}"
        )

    seen_variants: set[str] = set()
    for directory in directories:
        key = variant_key(directory.name)
        if key in seen_variants:
            raise ValueError(f"Duplicate variant {key} under {results_dir}")
        seen_variants.add(key)

        event_dirs = sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_dir() and EVENT_PATTERN.fullmatch(path.name)
            ),
            key=lambda path: int(EVENT_PATTERN.fullmatch(path.name).group("event")),
        )
        if len(event_dirs) != 8:
            raise ValueError(f"{directory}: expected 8 event directories, found {len(event_dirs)}")

        for event_dir in event_dirs:
            result_path = event_dir / f"result_{event_dir.name}.txt"
            if not result_path.is_file():
                raise FileNotFoundError(f"Missing result file: {result_path}")
            match = RESULT_PATTERN.fullmatch(result_path.read_text(encoding="utf-8").strip())
            if match is None:
                raise ValueError(f"Could not parse result file: {result_path}")
            if match.group("event") != event_dir.name:
                raise ValueError(f"Event name mismatch in {result_path}")
            train_time = match.group("train_time")
            rows.append(
                {
                    "horizon": horizon,
                    "variant": key,
                    "variant_label": VARIANT_DISPLAY[key],
                    "variant_directory": directory.name,
                    "event": event_dir.name,
                    "result_status": match.group("status"),
                    "best_epoch": int(match.group("epoch")),
                    "MAE": float(match.group("mae")),
                    "RMSE": float(match.group("rmse")),
                    "train_time_min": "" if train_time == "N/A" else float(train_time),
                    "result_file": result_path.relative_to(results_dir).as_posix(),
                }
            )

    missing = set(VARIANT_KEYS) - seen_variants
    if missing:
        raise ValueError(f"{results_dir}: missing variants {sorted(missing)}")
    if len(rows) != 40:
        raise ValueError(f"Expected 40 rows (5 variants x 8 events), collected {len(rows)}")
    return rows


def write_avg_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the long-format CSV consumed by ``ablation_metric_curves.py``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["event", "variant", "RMSE", "MAE"]
    ordered = sorted(
        rows,
        key=lambda row: (VARIANT_KEYS.index(str(row["variant"])), str(row["event"])),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered:
            writer.writerow(
                {
                    "event": row["event"],
                    "variant": row["variant"],
                    "RMSE": f"{float(row['RMSE']):.6f}",
                    "MAE": f"{float(row['MAE']):.6f}",
                }
            )


def write_full_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the wide transposed CSV kept for quick inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (VARIANT_KEYS.index(str(row["variant"])), str(row["event"])),
    )
    events = [str(row["event"]) for row in ordered]
    variants = [str(row["variant"]) for row in ordered]
    rmses = [f"{float(row['RMSE']):.6f}" for row in ordered]
    maes = [f"{float(row['MAE']):.6f}" for row in ordered]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event", *events])
        writer.writerow(["variant", *variants])
        writer.writerow(["RMSE", *rmses])
        writer.writerow(["MAE", *maes])


def write_detail_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "horizon",
        "variant",
        "variant_label",
        "variant_directory",
        "event",
        "result_status",
        "best_epoch",
        "MAE",
        "RMSE",
        "train_time_min",
        "result_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_horizon_rows(all_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        grouped[(str(row["event"]), str(row["variant"]))].append(row)

    merged: list[dict[str, object]] = []
    for event, variant in sorted(
        grouped,
        key=lambda key: (VARIANT_KEYS.index(key[1]), key[0]),
    ):
        items = grouped[(event, variant)]
        if len(items) < 1:
            continue
        merged.append(
            {
                "event": event,
                "variant": variant,
                "RMSE": mean(float(item["RMSE"]) for item in items),
                "MAE": mean(float(item["MAE"]) for item in items),
            }
        )
    expected = 8 * len(VARIANT_KEYS)
    if len(merged) != expected:
        raise ValueError(f"Expected {expected} merged rows, got {len(merged)}")
    return merged


def mean_by_variant(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    summary: list[dict[str, object]] = []
    for variant in VARIANT_KEYS:
        items = grouped[variant]
        summary.append(
            {
                "variant": variant,
                "variant_label": VARIANT_DISPLAY[variant],
                "MAE": mean(float(item["MAE"]) for item in items),
                "RMSE": mean(float(item["RMSE"]) for item in items),
            }
        )
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["variant", "variant_label", "MAE", "RMSE"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variant": row["variant"],
                    "variant_label": row["variant_label"],
                    "MAE": f"{float(row['MAE']):.6f}",
                    "RMSE": f"{float(row['RMSE']):.6f}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Horizons to process (directories named ucd-<horizon>)",
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=DEFAULT_MERGED_OUTPUT,
        help="Path for the horizon-averaged per-event CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, object]] = []

    for horizon in args.horizons:
        results_dir = horizon_results_dir(horizon)
        rows = collect_horizon_rows(results_dir, horizon)
        all_rows.extend(rows)

        write_avg_csv(results_dir / "ablation_results_avg.csv", rows)
        write_full_csv(results_dir / "ablation_results_full.csv", rows)
        write_detail_csv(results_dir / f"ablation_{horizon}_results.csv", rows)
        write_summary_csv(
            results_dir / "ablation_results_summary.csv", mean_by_variant(rows)
        )
        print(f"[{horizon}] wrote {len(rows)} event-level rows under {results_dir}")

    merged = merge_horizon_rows(all_rows)
    write_avg_csv(args.merged_output, merged)
    write_summary_csv(
        args.merged_output.with_name("ablation_results_summary.csv"),
        mean_by_variant(merged),
    )
    write_detail_csv(
        args.merged_output.with_name("ablation_results_detail.csv"),
        all_rows,
    )
    print(f"Wrote merged {len(merged)} rows to {args.merged_output}")


if __name__ == "__main__":
    main()
