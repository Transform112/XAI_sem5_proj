"""Optional torch / PyTorch Geometric support detection.

The canonical lane (RTX 5060) and the Kaggle lane have working CUDA torch
builds; the legacy development PC may have a torch wheel that cannot load at
all (missing MSVC runtime -> ``OSError: [WinError 126] c10.dll``). Data-stage
code must therefore never require torch, and torch-dependent code/tests must
skip cleanly instead of crashing.

``import torch`` can raise more than ImportError on a broken wheel (DLL load
failures surface as ``OSError``), so this module centralises the guarded probe.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def torch_status() -> dict[str, Any]:
    """Probe torch once and cache the result (safe to call from anywhere)."""
    status: dict[str, Any] = {
        "available": False,
        "reason": None,
        "version": None,
        "cuda_available": False,
        "device_count": 0,
    }
    try:
        import torch  # noqa: PLC0415 - optional dependency, guarded on purpose
    except ImportError as exc:
        status["reason"] = f"ImportError: {exc}"
        return status
    except OSError as exc:  # broken wheel / missing runtime DLLs
        status["reason"] = f"OSError loading torch (broken wheel or missing runtime): {exc}"
        return status
    except Exception as exc:  # noqa: BLE001 - any load failure means "unavailable"
        status["reason"] = f"{type(exc).__name__}: {exc}"
        return status
    status["available"] = True
    status["version"] = getattr(torch, "__version__", None)
    try:
        status["cuda_available"] = bool(torch.cuda.is_available())
        status["device_count"] = int(torch.cuda.device_count()) if status["cuda_available"] else 0
        if status["cuda_available"]:
            status["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001 - driver probe failure is not fatal
        status["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return status


def torch_available() -> bool:
    """True only when ``import torch`` actually works."""
    return bool(torch_status()["available"])


def torch_unavailable_reason() -> str | None:
    reason = torch_status()["reason"]
    return None if reason is None else str(reason)


def require_torch() -> Any:
    """Import and return torch, raising a clear error when it cannot load."""
    if not torch_available():
        raise RuntimeError(
            "torch is not usable on this machine: " + str(torch_unavailable_reason())
        )
    import torch  # noqa: PLC0415

    return torch


@lru_cache(maxsize=1)
def pyg_status() -> dict[str, Any]:
    """Probe torch_geometric after torch itself is known to work."""
    status: dict[str, Any] = {"available": False, "reason": None, "version": None}
    if not torch_available():
        status["reason"] = f"torch unavailable: {torch_unavailable_reason()}"
        return status
    try:
        import torch_geometric  # noqa: PLC0415

        status["available"] = True
        status["version"] = getattr(torch_geometric, "__version__", None)
    except Exception as exc:  # noqa: BLE001 - extension wheels fail loudly
        status["reason"] = f"{type(exc).__name__}: {exc}"
    return status


def pyg_available() -> bool:
    return bool(pyg_status()["available"])


__all__ = [
    "pyg_available",
    "pyg_status",
    "require_torch",
    "torch_available",
    "torch_status",
    "torch_unavailable_reason",
]
