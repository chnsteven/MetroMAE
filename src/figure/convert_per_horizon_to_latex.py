#!/usr/bin/env python3
"""Convert ``per_horizon_results.csv`` to a single-column LaTeX table.

Input: horizon-wise means (one row per method).
Output: ``AAAI27/Tables/per_horizon.tex``

Usage:
    python src/figure/convert_per_horizon_to_latex.py
    python src/figure/convert_per_horizon_to_latex.py \\
        --csv TFB/results/per_horizon_results.csv \\
        --output AAAI27/Tables/per_horizon.tex
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
    PER_HORIZON_RESULTS_FILE,
    load_per_horizon_csv,
    render_overall_table,
)

DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "per_horizon.tex"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=PER_HORIZON_RESULTS_FILE,
        help="Input CSV path (default: TFB/results/per_horizon_results.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .tex path (default: AAAI27/Tables/per_horizon.tex)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv if args.csv.is_absolute() else REPO_ROOT / args.csv
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A top-of-page, two-column float keeps the complete metric-column layout
    # compact enough for the paper body.
    table = render_overall_table(load_per_horizon_csv(csv_path), single_column=False)
    output_path.write_text(table, encoding="utf-8")
    print(f"Wrote {output_path} from {csv_path}")


if __name__ == "__main__":
    main()
