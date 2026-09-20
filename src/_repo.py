"""Repo-root and import-path helpers for the paper analysis package.

All modules under src/ should import this first so peer imports work when
scripts are run as ``python src/<module>.py`` and repo-root helpers remain
importable (geometry writers, etc.).
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent

for _p in (str(SRC_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
