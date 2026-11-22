"""Memory-bounded evaluation logits (T30).

The pre-T30 evaluation called ``model(x, edge_index, edge_attr)`` and then sliced
logits for the split mask.  That materialises the edge head for every one of the
period's edges (e.g. ~1.4 GiB of ``hidden[src]``/``hidden[dst]`` concatenation for
1.13M edges + 128 hidden, OOM on the 3 GB legacy card) even though only the split
edges are needed.

Fixed paths, in order of preference:
1. ``edge_batch`` - pass the split indices to the model's batch-only head; exactly
   equal to the full-logit slice (tested <=1e-6) and memory O(split edges).
2. ``chunked`` - for models without ``edge_batch``: compute node embeddings once
   and project the head in chunks of ``chunk_edges`` (still exactly equal to the
   full-logit slice; never materialises the full-edge head repr).
3. ``full`` - legacy fallback only for models supporting neither path.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch


def supports_edge_batch(model: Any) -> bool:
    return "edge_batch" in inspect.signature(model.forward).parameters


def supports_chunked(model: Any) -> bool:
    return hasattr(model, "node_embeddings") and hasattr(model, "edge_logits")


def predict_mode(model: Any, *, chunk_edges: int | None = None) -> str:
    if supports_edge_batch(model):
        return "edge_batch"
    if chunk_edges is not None and int(chunk_edges) > 0 and supports_chunked(model):
        return "chunked"
    return "full"


def predict_logits(
    model: Any,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    indices: torch.Tensor,
    *,
    device: Any | None = None,
    chunk_edges: int | None = None,
    force_mode: str | None = None,
) -> torch.Tensor:
    """Return logits for exactly ``indices`` (order preserved), memory-bounded."""
    target_device = device if device is not None else x.device
    indices = indices.to(target_device)
    mode = force_mode or predict_mode(model, chunk_edges=chunk_edges)
    with torch.inference_mode():
        if mode == "edge_batch":
            # T36: edge_batch is memory O(indices), but on very tight lanes the
            # split indices themselves can still be large; chunk them when a
            # positive chunk_edges budget is given (bit-identical concatenation).
            chunk_size = int(chunk_edges or 0)
            if chunk_size <= 0:
                return model(x, edge_index, edge_attr, edge_batch=indices)
            outputs: list[torch.Tensor] = []
            for offset in range(0, len(indices), chunk_size):
                outputs.append(
                    model(x, edge_index, edge_attr, edge_batch=indices[offset : offset + chunk_size])
                )
            if not outputs:
                n_classes = int(getattr(model, "num_classes", 0) or 0)
                return torch.empty((0, n_classes), dtype=torch.float32, device=target_device)
            return torch.cat(outputs, dim=0)
        if mode == "chunked":
            if not supports_chunked(model):
                raise ValueError("chunked mode requires model.node_embeddings/edge_logits")
            chunk_size = max(int(chunk_edges or 0), 1)
            hidden = model.node_embeddings(x, edge_index, edge_attr)
            outputs: list[torch.Tensor] = []
            for offset in range(0, len(indices), chunk_size):
                outputs.append(model.edge_logits(hidden, edge_index, edge_attr, indices[offset : offset + chunk_size]))
            if not outputs:
                return torch.empty((0,), dtype=torch.long, device=target_device)
            return torch.cat(outputs, dim=0)
        logits = model(x, edge_index, edge_attr)
        return logits[indices]


__all__ = ["predict_logits", "predict_mode", "supports_chunked", "supports_edge_batch"]
