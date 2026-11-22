"""Shared fixtures for the ``analysis`` test suite (synthetic inputs only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ANALYSIS_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import artifacts, synthetic  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def bundle():
    """Session-scoped synthetic bundle (one model keeps the suite fast)."""
    return synthetic.make_bundle(models=("egraphsage",))


@pytest.fixture()
def layout(tmp_path: Path) -> artifacts.ArtifactLayout:
    """Isolated results layout so tests never touch the real results tree."""
    return artifacts.ArtifactLayout(tmp_path / "results")
