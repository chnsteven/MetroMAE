#!/usr/bin/env python3
"""Export full-model experiment results to CSV.

Reads result_*.txt under autodl-tmp/experiments and writes
event, horizon, sd, rmse, mae from TEST_at_VAL_best.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _collect_exp_results import collect_rows, write_csv

DEFAULT_EXP_ROOT = Path("/root/autodl-tmp/experiments")
DEFAULT_OUTPUT = DEFAULT_EXP_ROOT / "full_results.csv"
FIELDS = ["event", "horizon", "sd", "rmse", "mae"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-root", type=Path, default=DEFAULT_EXP_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows, skipped = collect_rows(args.exp_root, settings={"full"})
    write_csv(args.output, rows, FIELDS)
    print(f"Wrote {len(rows)} rows to {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} result files without TEST_at_VAL_best")


if __name__ == "__main__":
    main()
