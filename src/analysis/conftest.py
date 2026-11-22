"""Pytest bootstrap for the ``analysis`` package.

Adds the repository root to ``sys.path`` so ``pytest analysis`` works from the
project root even when the tests are collected without the root package being
importable (no editable install is used in this project).
"""

from __future__ import annotations

import sys
from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
