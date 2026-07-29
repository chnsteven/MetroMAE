"""Expose UcdGPT core modules from the monorepo ``src/`` directory."""

from __future__ import annotations

import sys
from pathlib import Path

_UCDGPT_SRC = Path(__file__).resolve().parents[4] / "src"
_UCDGPT_SRC_STR = str(_UCDGPT_SRC)


def ensure_ucdgpt_src() -> Path:
    """Add ``<repo>/src`` to ``sys.path`` so TFB can import UcdGPT modules."""
    if not _UCDGPT_SRC.is_dir():
        raise ImportError(f"UcdGPT src directory not found: {_UCDGPT_SRC}")
    if _UCDGPT_SRC_STR not in sys.path:
        sys.path.insert(0, _UCDGPT_SRC_STR)
    return _UCDGPT_SRC
