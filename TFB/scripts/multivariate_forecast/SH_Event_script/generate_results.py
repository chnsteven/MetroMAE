#!/usr/bin/env python3
"""Generate SH_Event result summary CSV files from benchmark archives."""

from __future__ import annotations

import argparse
import csv
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


HORIZONS = ("d12", "d24", "d36", "d48")
EVENTS = tuple(f"event_{i}.csv" for i in range(8))
SUMMARY_FIELDS = ("row_type", "horizon", "dataset", "mae", "rmse", "n_results")
MERGED_FIELDS = ("model",) + SUMMARY_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SH_Event overall, per-event, and average CSV tables."
    )
    parser.add_argument(
        "--root",
        default="results/",
        help="results directory. Default: %(default)s",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Also generate LaTeX tables for event_0..7 and average results.",
    )
    return parser.parse_args()


def read_archive_rows(archive_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".csv"):
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            text = extracted.read().decode("utf-8")
            rows.extend(csv.DictReader(text.splitlines()))
    return rows


def horizon_dirs(model_dir: Path) -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    for horizon in HORIZONS:
        dirs[horizon] = model_dir / "hourly" / horizon
    return dirs


def fmt_float(value: float) -> str:
    return f"{value:.12g}"


def generate_model_summary(model_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    values: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    issues: list[str] = []

    dirs = horizon_dirs(model_dir)
    for horizon in HORIZONS:
        horizon_dir = dirs.get(horizon)
        if horizon_dir is None:
            issues.append(f"{model_dir.name} {horizon}: missing horizon directory")
            continue

        archives = sorted(horizon_dir.glob("*.csv.tar.gz"))
        if not archives:
            issues.append(f"{model_dir.name} {horizon}: no csv.tar.gz archives")
            continue

        for archive in archives:
            for row in read_archive_rows(archive):
                file_name = (row.get("file_name") or "").strip()
                if not file_name:
                    continue
                try:
                    mae = float(row["mae"])
                    rmse = float(row["rmse"])
                except (KeyError, TypeError, ValueError):
                    continue
                values[(horizon, file_name)].append((mae, rmse))

    event_rows: list[dict[str, str]] = []
    horizon_rows: list[dict[str, str]] = []
    horizon_avgs: list[tuple[float, float]] = []

    for horizon in HORIZONS:
        event_vals: list[tuple[float, float]] = []
        for event in EVENTS:
            vals = values.get((horizon, event), [])
            if vals:
                mae = mean(v[0] for v in vals)
                rmse = mean(v[1] for v in vals)
                event_vals.append((mae, rmse))
                event_rows.append(
                    {
                        "row_type": "event",
                        "horizon": horizon,
                        "dataset": event,
                        "mae": fmt_float(mae),
                        "rmse": fmt_float(rmse),
                        "n_results": str(len(vals)),
                    }
                )
            else:
                issues.append(f"{model_dir.name} {horizon}: missing {event}")
                event_rows.append(
                    {
                        "row_type": "event",
                        "horizon": horizon,
                        "dataset": event,
                        "mae": "",
                        "rmse": "",
                        "n_results": "0",
                    }
                )

        if event_vals:
            mae = mean(v[0] for v in event_vals)
            rmse = mean(v[1] for v in event_vals)
            horizon_avgs.append((mae, rmse))
            horizon_rows.append(
                {
                    "row_type": "horizon_avg",
                    "horizon": horizon,
                    "dataset": "AVERAGE",
                    "mae": fmt_float(mae),
                    "rmse": fmt_float(rmse),
                    "n_results": str(len(event_vals)),
                }
            )
        else:
            horizon_rows.append(
                {
                    "row_type": "horizon_avg",
                    "horizon": horizon,
                    "dataset": "AVERAGE",
                    "mae": "",
                    "rmse": "",
                    "n_results": "0",
                }
            )

    if horizon_avgs:
        overall_mae = mean(v[0] for v in horizon_avgs)
        overall_rmse = mean(v[1] for v in horizon_avgs)
        overall_row = {
            "row_type": "overall_avg",
            "horizon": "ALL",
            "dataset": "AVERAGE",
            "mae": fmt_float(overall_mae),
            "rmse": fmt_float(overall_rmse),
            "n_results": str(len(horizon_avgs)),
        }
    else:
        overall_row = {
            "row_type": "overall_avg",
            "horizon": "ALL",
            "dataset": "AVERAGE",
            "mae": "",
            "rmse": "",
            "n_results": "0",
        }

    rows = event_rows + horizon_rows + [overall_row]
    if len(rows) != 37:
        raise RuntimeError(f"{model_dir.name}: expected 37 rows, got {len(rows)}")
    return rows, issues


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_latex(root: Path, models: list[str]) -> None:
    for event_id in range(8):
        csv_path = root / f"event_{event_id}_result.csv"
        tex_path = root / f"event_{event_id}_result.tex"
        make_latex_table(
            csv_path,
            tex_path,
            models,
            ("d12", "d24", "d36", "d48"),
            f"SH Event event\\_{event_id} forecasting results",
            f"tab:sh-event-event-{event_id}",
        )

    make_latex_table(
        root / "average_result.csv",
        root / "average_result.tex",
        models,
        ("d12", "d24", "d36", "d48", "ALL"),
        "SH Event average forecasting results",
        "tab:sh-event-average",
    )


def make_latex_table(
    csv_path: Path,
    tex_path: Path,
    models: list[str],
    horizons: tuple[str, ...],
    caption: str,
    label: str,
) -> None:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_model = {model: {} for model in models}
    for row in rows:
        if row["model"] in by_model:
            by_model[row["model"]][row["horizon"]] = row

    def metric(value: str) -> str:
        if value == "":
            return "--"
        return f"{float(value):.4f}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{'l' + 'rr' * len(horizons)}}}",
        r"\toprule",
    ]
    header = ["Model"]
    for horizon in horizons:
        header.extend([f"{horizon} MAE", f"{horizon} RMSE"])
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for model in models:
        cells = [model]
        for horizon in horizons:
            row = by_model.get(model, {}).get(horizon)
            if row is None:
                cells.extend(["--", "--"])
            else:
                cells.extend([metric(row["mae"]), metric(row["rmse"])])
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    tex_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    model_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and horizon_dirs(path)
    )

    merged_rows: list[dict[str, str]] = []
    all_issues: list[str] = []
    for model_dir in model_dirs:
        rows, issues = generate_model_summary(model_dir)
        write_csv(model_dir / "overall_result.csv", SUMMARY_FIELDS, rows)
        merged_rows.extend({"model": model_dir.name, **row} for row in rows)
        all_issues.extend(issues)

    write_csv(root / "overall_result.csv", MERGED_FIELDS, merged_rows)

    for event in EVENTS:
        event_rows = [
            row
            for row in merged_rows
            if row["row_type"] == "event" and row["dataset"] == event
        ]
        event_id = event.removeprefix("event_").removesuffix(".csv")
        write_csv(root / f"event_{event_id}_result.csv", MERGED_FIELDS, event_rows)

    average_rows = [
        row for row in merged_rows if row["row_type"] in ("horizon_avg", "overall_avg")
    ]
    write_csv(root / "average_result.csv", MERGED_FIELDS, average_rows)

    if args.latex:
        generate_latex(root, [path.name for path in model_dirs])

    print(f"models: {', '.join(path.name for path in model_dirs)}")
    print(f"overall rows: {len(merged_rows)}")
    print("per-model rows:", dict(sorted(Counter(row["model"] for row in merged_rows).items())))
    if all_issues:
        print("issues:")
        for issue in all_issues:
            print(f"  - {issue}")


if __name__ == "__main__":
    main()
