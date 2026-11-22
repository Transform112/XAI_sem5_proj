"""XDRIFT: calibrated measurement of explanation drift in continual GNN NIDS.

This package is the single source of truth for every executable stage of the
study (data audit, graph construction, training, explainers, drift analysis).
It must stay operating-system neutral: no absolute paths, every path resolved
through :mod:`src.utils.paths`, every run described by a manifest from
:mod:`src.utils.manifest`.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
