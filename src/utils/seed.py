"""Deterministic seeding helpers.

Every training/explanation stage must start from the same state for a given
seed. :func:`seed_everything` sets Python, NumPy and (when installed) PyTorch
seeds, enables deterministic algorithms where the installed torch version
supports it, and returns a small record of what was applied so the run manifest
can store it.

Notes
-----
* ``PYTHONHASHSEED`` only matters before interpreter start-up; the lane launcher
  sets it for child processes. If torch is already imported, CUDA workspace
  settings may be too late, so this module sets ``CUBLAS_WORKSPACE_CONFIG``
  before importing torch.
* ``torch.use_deterministic_algorithms(True)`` can make some ops raise on CUDA;
  ``warn_only=True`` keeps the run alive while still surfacing a warning, which
  is the project convention for the exploratory phase. Freeze a stricter value
  in the protocol if a comparison run requires bitwise reproducibility.
"""

from __future__ import annotations

import os
import random
from typing import Any


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    warn_only: bool = True,
    cudnn_deterministic: bool = True,
    set_cublas_workspace: bool = True,
) -> dict[str, Any]:
    """Seed all RNGs used by the study and return the applied settings."""
    if seed is None:
        raise ValueError("seed must be an integer")
    seed = int(seed)
    settings: dict[str, Any] = {
        "seed": seed,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "deterministic_algorithms": bool(deterministic),
        "cudnn_deterministic": bool(cudnn_deterministic and deterministic),
        "numpy": False,
        "torch": False,
        "torch_version": None,
    }

    # Must be set before the CUDA context exists; harmless otherwise.
    if set_cublas_workspace:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)

    try:  # NumPy is a hard dependency of the data stages, optional for syntax tests.
        import numpy as np

        np.random.seed(seed % (2**32))
        settings["numpy"] = True
    except ImportError:  # pragma: no cover - minimal environments
        settings["numpy"] = False

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = settings["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = False
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)
            except (AttributeError, TypeError):  # pragma: no cover - old torch
                torch.use_deterministic_algorithms(True)
        settings["torch"] = True
        settings["torch_version"] = torch.__version__
        settings["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - CPU analysis environments
        settings["torch"] = False
        settings["cuda_available"] = False

    return settings


def describe_determinism_state() -> dict[str, Any]:
    """Report current torch determinism flags without changing them."""
    state: dict[str, Any] = {"torch": False}
    try:
        import torch
    except ImportError:
        return state
    state.update(
        {
            "torch": True,
            "torch_version": torch.__version__,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": getattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: None)(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_available": torch.cuda.is_available(),
        }
    )
    return state


__all__ = ["describe_determinism_state", "seed_everything"]
