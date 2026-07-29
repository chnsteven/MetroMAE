"""Expose MetroMAE core modules from the monorepo ``src/`` directory."""

from __future__ import annotations

import sys
from pathlib import Path

_METROMAE_SRC = Path(__file__).resolve().parents[4] / "src"
_METROMAE_SRC_STR = str(_METROMAE_SRC)


def ensure_metromae_src() -> Path:
    """Add ``<repo>/src`` to ``sys.path`` so TFB can import MetroMAE modules."""
    if not _METROMAE_SRC.is_dir():
        raise ImportError(f"MetroMAE src directory not found: {_METROMAE_SRC}")
    if _METROMAE_SRC_STR not in sys.path:
        sys.path.insert(0, _METROMAE_SRC_STR)
    return _METROMAE_SRC
