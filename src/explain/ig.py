"""S7: Integrated Gradients adapter (deterministic, zero baseline, predicted class).

Attribution formulation (Doc 2 S3.3 + S7): for a target edge and its predicted
class logit, integrate gradients along the straight path from a zero feature
baseline to the real edge-feature vector using ``steps`` Riemann samples. The
completeness identity ``sum(attributions) ~= f(x) - f(baseline)`` is recorded so
the explanation can be audited.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


def _as_device(device: str) -> torch.device:
    return torch.device(device)


def integrated_gradients(
    model: Any,
    data: Any,
    target_edge: int,
    *,
    target_class: int | None = None,
    steps: int = 32,
    baseline: float = 0.0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Return IG feature attributions for one edge in a (sub)graph."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    device_obj = _as_device(device)
    model.to(device_obj).eval()
    x = data.x.to(device_obj)
    edge_index = data.edge_index.to(device_obj)
    edge_attr = data.edge_attr.to(device_obj)
    target_edge = int(target_edge)
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
        if target_class is None:
            target_class = int(logits[target_edge].argmax())
        target_class = int(target_class)
        base_value = logits[target_edge, target_class]
    base_features = edge_attr[target_edge].detach().clone()
    baseline_tensor = torch.full_like(base_features, float(baseline))
    attributions = torch.zeros_like(base_features)
    for step in range(int(steps)):
        alpha = (step + 0.5) / float(steps)
        interpolated = baseline_tensor + alpha * (base_features - baseline_tensor)
        current = edge_attr.clone()
        current[target_edge] = interpolated
        current.requires_grad_(True)
        logits = model(x, edge_index, current)
        score = logits[target_edge, target_class]
        grads = torch.autograd.grad(score, current, retain_graph=False)[0][target_edge]
        attributions += grads.detach()
    attributions = (base_features - baseline_tensor) * (attributions / float(steps))
    with torch.no_grad():
        current = edge_attr.clone()
        current[target_edge] = baseline_tensor
        baseline_logits = model(x, edge_index, current)
        baseline_value = baseline_logits[target_edge, target_class]
    completeness = float(attributions.sum()) - float((base_value - baseline_value))
    return {
        "explainer": "ig",
        "target_edge": target_edge,
        "target_class": target_class,
        "steps": int(steps),
        "baseline": float(baseline),
        "attributions": attributions.detach().cpu().numpy().astype("float64"),
        "attribution_sum": float(attributions.sum()),
        "model_output": float(base_value),
        "baseline_output": float(baseline_value),
        "completeness_delta": completeness,
    }


def top_k_features(attributions: np.ndarray, k: int, *, by_absolute: bool = True) -> list[dict[str, Any]]:
    """Top-k feature indices with signed attribution values."""
    values = np.asarray(attributions, dtype="float64")
    order = np.argsort(-np.abs(values) if by_absolute else -values)[: int(k)]
    return [{"feature_index": int(index), "attribution": float(values[index])} for index in order]


__all__ = ["integrated_gradients", "top_k_features"]
