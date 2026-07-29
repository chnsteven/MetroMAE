#!/usr/bin/env python3
"""Convert ``overall_results.csv`` to a single-column LaTeX table.

Input: per-event results (eight event categories per method and horizon).
Output: ``AAAI27/Tables/per_dataset.tex``

Usage:
    python src/figure/convert_overall_results_to_latex.py
    python src/figure/convert_overall_results_to_latex.py \\
        --csv TFB/results/overall_results.csv \\
        --output AAAI27/Tables/per_dataset.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

FIGURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURE_DIR.parents[1]
if str(FIGURE_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE_DIR))

from generate_tfb_latex_tables import (
    DEFAULT_OUTPUT_DIR,
    OVERALL_RESULTS_CSV,
    load_overall_results_csv,
    render_per_event_results_table,
)

DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "per_dataset.tex"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=OVERALL_RESULTS_CSV,
        help="Input CSV path (default: TFB/results/overall_results.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .tex path (default: AAAI27/Tables/per_dataset.tex)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv if args.csv.is_absolute() else REPO_ROOT / args.csv
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = render_per_event_results_table(
        load_overall_results_csv(csv_path),
        single_column=False,
        full_width=True,
        float_top=True,
    )
    output_path.write_text(table, encoding="utf-8")
    print(f"Wrote {output_path} from {csv_path}")


if __name__ == "__main__":
    main()
