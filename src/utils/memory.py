"""Pre-flight memory guard for training on low-memory GPUs (D-049/T24).

The canonical path (D-052) is neighbor-sampled edge mini-batching, so only one
bounded per-step subgraph is allocated; the runner passes the per-step edge cap
here. The full-graph estimate is still used for the diagnostic full_batch mode
(which stores the whole-graph message-passing activations for one backward pass). That fits the 8 GB RTX 5060 but not the 3 GB
legacy card, so ``run_train`` estimates peak memory before training and warns with
an actionable fallback instead of dying inside ``scatter_add_``.

The estimate is deliberately conservative (it includes an autograd factor and a
fixed workspace overhead) and is calibrated against the measured full-size
behaviour: 1.13M edges, hidden 128, edge-dim 50 â†’ 2.4â€“3 GB peak on GPU.
"""

from __future__ import annotations

from typing import Any, Mapping

BYTES_PER_FLOAT32 = 4
#: Autograd stores forward activations plus gradients; conservative multiplier.
_BACKWARD_FACTOR = 1.5
#: Fixed allocator/workspace overhead (CUDA context, temporaries).
_WORKSPACE_OVERHEAD_BYTES = 384 * 1024 * 1024


def estimate_training_peak_bytes(
    *,
    n_edges: int,
    node_dim: int,
    edge_dim: int,
    hidden_dim: int,
    n_classes: int,
    n_layers: int = 2,
    chunk_edges: int | None = None,
) -> int:
    """Conservative peak-memory estimate for one full-batch forward+backward.

    One message-passing pass saves per-layer edge messages (``n_edges x hidden``)
    plus the logits for the chunked edge head. ``chunk_edges`` only bounds the
    loss head, not the message activations, so it is a small correction.
    """
    n_edges = max(int(n_edges), 1)
    hidden_dim = max(int(hidden_dim), 1)
    edge_dim = max(int(edge_dim), 1)
    n_layers = max(int(n_layers), 1)
    message_bytes = n_edges * (hidden_dim + edge_dim) * BYTES_PER_FLOAT32 * n_layers
    autograd_bytes = int(message_bytes * _BACKWARD_FACTOR)
    head_edges = n_edges if not chunk_edges else min(n_edges, max(int(chunk_edges), 1))
    logits_bytes = head_edges * max(int(n_classes), 1) * BYTES_PER_FLOAT32 * 2  # logits + grad
    parameter_bytes = 64 * 1024 * 1024  # small E-GraphSAGE/GCN head + optimiser state
    return int(autograd_bytes + logits_bytes + parameter_bytes + _WORKSPACE_OVERHEAD_BYTES)


def _gib(value: int | float) -> float:
    return round(float(value) / float(1024**3), 3)


def device_memory_info(device: Any) -> dict[str, Any]:
    """Total/free memory of the target device in bytes (CPU via psutil when present)."""
    device_text = str(device)
    if device_text.startswith("cuda"):
        try:
            import torch

            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return {"device": device_text, "total_bytes": int(total_bytes), "free_bytes": int(free_bytes)}
        except Exception:  # noqa: BLE001 - fall through to unknown
            return {"device": device_text, "total_bytes": None, "free_bytes": None}
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {"device": "cpu", "total_bytes": int(memory.total), "free_bytes": int(memory.available)}
    except Exception:  # noqa: BLE001
        return {"device": "cpu", "total_bytes": None, "free_bytes": None}


def check_memory_budget(
    *,
    n_edges: int,
    model_config: Mapping[str, Any],
    node_dim: int,
    edge_dim: int,
    device: Any,
    guard: str = "warn",
    chunk_edges: int | None = None,
    reserve_fraction: float = 0.15,
) -> dict[str, Any]:
    """Estimate peak memory and decide whether it fits the target device.

    ``guard='warn'`` never blocks; ``'error'`` marks ``fits=False`` so the caller can
    abort before training; ``'off'`` skips the check.
    """
    config = dict(model_config or {})
    estimate = estimate_training_peak_bytes(
        n_edges=n_edges,
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=int(config.get("hidden_dim", 128)),
        n_classes=int(config.get("num_classes", 10)),
        n_layers=int(config.get("num_layers", 2)),
        chunk_edges=chunk_edges,
    )
    info = device_memory_info(device)
    result: dict[str, Any] = {
        "guard": guard,
        "estimated_peak_bytes": estimate,
        "estimated_peak_gib": _gib(estimate),
        "device": info,
        "n_edges": int(n_edges),
        "chunk_edges": int(chunk_edges) if chunk_edges else None,
        "fits": None,
        "warning": None,
    }
    if guard == "off":
        return result
    free_bytes = info.get("free_bytes")
    total_bytes = info.get("total_bytes")
    if free_bytes is None:
        result["warning"] = "device memory unknown; skipping fit check"
        return result
    budget = int(max(float(free_bytes) * (1.0 - float(reserve_fraction)), 0))
    fits = estimate <= budget
    result["fits"] = bool(fits)
    result["budget_bytes"] = budget
    if not fits:
        result["warning"] = (
            "Estimated training peak "
            f"{_gib(estimate)} GiB exceeds the safe budget {_gib(budget)} GiB on {info.get('device')}; "
            "the full-graph pass can OOM. Fallbacks: (a) run full size on the 8 GB canonical lane "
            "(rtx5060), (b) use --max-edges for a smoke subgraph (legacy lane), or (c) reduce "
            "hidden_dim/chunk_edges. batching=mini_batch does NOT reduce peak memory."
        )
    return result


__all__ = [
    "BYTES_PER_FLOAT32",
    "check_memory_budget",
    "device_memory_info",
    "estimate_training_peak_bytes",
]
