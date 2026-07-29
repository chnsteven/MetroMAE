#!/usr/bin/env python3
"""Backward-compatible wrapper around ``process_ucd_ablation_results.py``.

Usage:
    python TFB/scripts/process_ucd_d24_ablation_results.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("process_ucd_ablation_results.py")
    runpy.run_path(str(target), run_name="__main__")
