"""Pytest root configuration.

Adds the repository root to ``sys.path`` so tests can ``import src`` without an
editable install, and exposes a few repo-level fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
