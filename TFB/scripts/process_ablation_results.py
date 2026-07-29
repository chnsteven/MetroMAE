#!/usr/bin/env python3
"""Process benchmark archives into horizon and overall CSVs."""

from __future__ import annotations

import csv
import shutil
import tarfile
from pathlib import Path
from statistics import mean

HORIZONS = ("d12", "d24", "d36", "d48")
EVENTS = tuple(f"event_{i}.csv" for i in range(8))
FIELDS = (
    "model_name",
    "strategy_args",
    "model_params",
    "mae",
    "rmse",
    "file_name",
    "fit_time",
    "inference_time",
    "actual_data",
    "inference_data",
    "log_info",
)


def read_archive_row(archive_path: Path) -> dict[str, str]:
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".csv"):
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            rows = list(csv.DictReader(extracted.read().decode("utf-8").splitlines()))
            if not rows:
                raise ValueError(f"No rows in {archive_path}")
            return rows[0]
    raise ValueError(f"No csv in {archive_path}")


def mean_row(rows: list[dict[str, str]]) -> dict[str, str]:
    def avg(key: str) -> str:
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        return str(mean(values)) if values else ""

    return {
        "model_name": "",
        "strategy_args": "",
        "model_params": "",
        "mae": avg("mae"),
        "rmse": avg("rmse"),
        "file_name": "",
        "fit_time": avg("fit_time"),
        "inference_time": avg("inference_time"),
        "actual_data": "",
        "inference_data": "",
        "log_info": "",
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def process_results_dir(
    results_dir: Path,
    *,
    remove_archives: bool = False,
    horizon_output_dir: Path | None = None,
) -> None:
    hourly_dir = results_dir / "hourly"
    horizon_output_dir = horizon_output_dir or hourly_dir
    if not hourly_dir.exists():
        raise FileNotFoundError(f"Missing hourly directory: {hourly_dir}")

    for checkpoint in results_dir.rglob(".ipynb_checkpoints"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)

    overall_rows: list[dict[str, str]] = []

    for horizon in HORIZONS:
        horizon_dir = hourly_dir / horizon
        if not horizon_dir.exists():
            raise FileNotFoundError(f"Missing horizon directory: {horizon_dir}")

        for csv_path in horizon_dir.glob("test_report*.csv"):
            csv_path.unlink()

        archives = sorted(horizon_dir.glob("*.csv.tar.gz"))
        if len(archives) != 8:
            raise RuntimeError(
                f"{horizon_dir}: expected 8 archives, found {len(archives)}"
            )

        by_event: dict[str, dict[str, str]] = {}
        for archive in archives:
            row = read_archive_row(archive)
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(path=horizon_dir, filter="data")
            file_name = (row.get("file_name") or "").strip()
            if not file_name:
                raise ValueError(f"Missing file_name in {archive}")
            by_event[file_name] = row

        event_rows = [by_event[event] for event in EVENTS]
        horizon_rows = event_rows + [mean_row(event_rows)]
        horizon_csv = horizon_output_dir / f"{horizon}.csv"
        write_csv(horizon_csv, horizon_rows)
        overall_rows.extend(horizon_rows)

        if remove_archives:
            for archive in archives:
                archive.unlink()

    overall_rows.append(mean_row([row for row in overall_rows if row.get("file_name")]))
    write_csv(results_dir / "overall.csv", overall_rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "results"
    ablation_dirs = sorted(root.glob("UCDGPT-ablation-*"))
    if not ablation_dirs:
        raise FileNotFoundError("No UCDGPT-ablation-* directories found")

    for ablation_dir in ablation_dirs:
        print(f"Processing {ablation_dir.name} ...")
        process_results_dir(ablation_dir)
        print(f"  wrote {ablation_dir / 'overall.csv'}")

    stmtm_dir = root / "STMTM"
    if stmtm_dir.exists():
        print("Processing STMTM ...")
        process_results_dir(
            stmtm_dir,
            remove_archives=True,
            horizon_output_dir=stmtm_dir,
        )
        print(f"  wrote {stmtm_dir / 'overall.csv'}")


if __name__ == "__main__":
    main()
