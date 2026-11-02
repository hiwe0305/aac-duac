#!/usr/bin/env python3
"""LIBERO evaluation wrapper for pi0.5 (delegates to flowbased_aac/duac.py)."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FLOWBASED_ROOT = ROOT / "flowbased_aac"
for p in (ROOT, FLOWBASED_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from duac import main as duac_main  # noqa: E402


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("eval-libero", "eval-liberopro", "serve-pi05", "doctor", "test"):
        argv = ["eval-libero", *argv]
    sys.exit(duac_main(argv))
