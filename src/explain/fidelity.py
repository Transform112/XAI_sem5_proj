"""S7 fidelity checks: sufficiency / comprehensiveness versus a random mask.

For an anchor edge with feature attributions:

* **sufficiency**: keep only the top-k attributed features (zero the rest) and
  measure how much of the predicted-class probability is preserved;
* **comprehensiveness**: remove the top-k attributed features (zero them) and
  measure the drop in predicted-class probability;
* the comprehensiveness of a random k-feature mask is the control: a faithful
  explanation should be more comprehensive than random removal.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch


def _predict(
    model: Any,
    data: Any,
    target_edge: int,
    edge_attr: torch.Tensor,
    device: torch.device,
) -> tuple[int, float]:
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device), edge_attr.to(device))
        probabilities = torch.softmax(logits[int(target_edge)], dim=-1)
        predicted = int(probabilities.argmax())
        return predicted, float(probabilities[predicted])


def feature_fidelity(
    model: Any,
    data: Any,
    target_edge: int,
    attributions: Sequence[float] | np.ndarray,
    *,
    k_values: Sequence[int] = (5, 10),
    random_trials: int = 5,
    seed: int = 42,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Per-k sufficiency/comprehensiveness with a seeded random-mask control."""
    device_obj = torch.device(device)
    model.to(device_obj).eval()
    x = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr.detach().clone()
    target_edge = int(target_edge)
    predicted, base_probability = _predict(model, data, target_edge, edge_attr, device_obj)
    values = np.asarray(attributions, dtype="float64").reshape(-1)
    n_features = len(values)
    order = np.argsort(-np.abs(values))
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    for k in k_values:
        k = int(min(int(k), n_features))
        if k <= 0:
            continue
        keep = np.zeros(n_features, dtype=bool)
        keep[order[:k]] = True
        # Sufficiency: keep only top-k; zero the rest.
        sufficient_attr = edge_attr.clone()
        zero_rows = ~torch.as_tensor(keep, device=edge_attr.device)
        sufficient_attr[target_edge, zero_rows] = 0.0
        _, sufficient_probability = _predict(model, data, target_edge, sufficient_attr, device_obj)
        # Comprehensiveness: remove top-k.
        removed_attr = edge_attr.clone()
        removed_attr[target_edge, torch.as_tensor(order[:k], device=edge_attr.device)] = 0.0
        _, removed_probability = _predict(model, data, target_edge, removed_attr, device_obj)
        # Random-mask control (same k, random features).
        random_probabilities = []
        for _trial in range(int(random_trials)):
            random_indices = rng.choice(n_features, size=k, replace=False)
            random_attr = edge_attr.clone()
            random_attr[target_edge, torch.as_tensor(random_indices, device=edge_attr.device)] = 0.0
            _, random_probability = _predict(model, data, target_edge, random_attr, device_obj)
            random_probabilities.append(random_probability)
        random_probability = float(np.mean(random_probabilities)) if random_probabilities else base_probability
        results.append(
            {
                "k": k,
                "predicted_class": predicted,
                "base_probability": base_probability,
                "sufficiency_probability": sufficient_probability,
                "sufficiency_ratio": sufficient_probability / base_probability if base_probability > 0 else 0.0,
                "comprehensiveness": base_probability - removed_probability,
                "comprehensiveness_random": base_probability - random_probability,
                "comprehensiveness_over_random": (
                    (base_probability - removed_probability) / (base_probability - random_probability)
                    if base_probability - random_probability > 1e-12
                    else None
                ),
                "random_trials": int(random_trials),
            }
        )
    return results


def edge_mask_fidelity(
    model: Any,
    data: Any,
    target_edge: int,
    edge_mask: Sequence[float] | np.ndarray,
    *,
    m: int = 5,
    random_trials: int = 5,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Edge-mask fidelity at ``m``: sufficiency/comprehensiveness vs random control.

    Sufficiency keeps only the top-m masked edges' features (zeroes the others);
    comprehensiveness removes the top-m masked edges' features. The target edge's
    own feature row is never modified (it is the explained unit).
    """
    device_obj = torch.device(device)
    model.to(device_obj).eval()
    target_edge = int(target_edge)
    edge_attr = data.edge_attr.detach().clone()
    predicted, base_probability = _predict(model, data, target_edge, edge_attr, device_obj)
    values = np.asarray(edge_mask, dtype="float64").reshape(-1)
    n_edges = len(values)
    m = int(min(max(int(m), 1), n_edges))
    order = np.argsort(-np.abs(values))
    rng = np.random.default_rng(seed)
    candidates = [index for index in order if int(index) != target_edge]
    keep = np.zeros(n_edges, dtype=bool)
    keep[candidates[:m]] = True
    keep[target_edge] = True  # never perturb the explained edge itself
    sufficient_attr = edge_attr.clone()
    zero_rows = ~torch.as_tensor(keep, device=edge_attr.device)
    sufficient_attr[zero_rows] = 0.0
    _, sufficient_probability = _predict(model, data, target_edge, sufficient_attr, device_obj)
    removed_attr = edge_attr.clone()
    removed_attr[torch.as_tensor(candidates[:m], device=edge_attr.device)] = 0.0
    _, removed_probability = _predict(model, data, target_edge, removed_attr, device_obj)
    random_probabilities = []
    for _trial in range(int(random_trials)):
        random_indices = rng.choice(n_edges, size=m, replace=False)
        random_attr = edge_attr.clone()
        random_attr[torch.as_tensor(random_indices, device=edge_attr.device)] = 0.0
        _, random_probability = _predict(model, data, target_edge, random_attr, device_obj)
        random_probabilities.append(random_probability)
    random_probability = float(np.mean(random_probabilities)) if random_probabilities else base_probability
    return {
        "k": m,
        "predicted_class": predicted,
        "base_probability": base_probability,
        "sufficiency_probability": sufficient_probability,
        "sufficiency_ratio": sufficient_probability / base_probability if base_probability > 0 else 0.0,
        "comprehensiveness": base_probability - removed_probability,
        "comprehensiveness_random": base_probability - random_probability,
        "comprehensiveness_over_random": (
            (base_probability - removed_probability) / (base_probability - random_probability)
            if base_probability - random_probability > 1e-12
            else None
        ),
        "random_trials": int(random_trials),
    }


__all__ = ["edge_mask_fidelity", "feature_fidelity"]
