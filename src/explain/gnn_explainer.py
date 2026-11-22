"""S7: GNNExplainer adapter via the PyG Explainer API (edge mask object + feature mask).

The wrapper exposes the E-GraphSAGE forward as ``forward(x, edge_index)`` with
the edge attributes captured in the module, which is the signature PyG's
``Explainer`` expects; because our layers subclass ``MessagePassing`` the
explainer edge mask is applied inside ``propagate`` (checked by a unit test on
the model side).

Determinism: the global torch seed is set from the configured seed before the
explainer runs, and the seed is recorded in the returned payload so cached
explanations are reproducible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


class _EdgeAttrWrapper(nn.Module):
    """Wrap an edge-feature model as ``forward(x, edge_index)`` for PyG Explainer."""

    def __init__(self, model: nn.Module, edge_attr: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("edge_attr", edge_attr.detach().clone())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.model(x, edge_index, self.edge_attr)


def gnn_explainer(
    model: Any,
    data: Any,
    target_edge: int,
    *,
    epochs: int = 100,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Return edge mask + node feature mask for one anchor edge."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    device_obj = torch.device(device)
    model.to(device_obj).eval()
    wrapper = _EdgeAttrWrapper(model, data.edge_attr.to(device_obj)).to(device_obj)
    x = data.x.to(device_obj)
    edge_index = data.edge_index.to(device_obj)
    from torch_geometric.explain import Explainer, GNNExplainer

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=int(epochs)),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="edge",
            return_type="log_probs",
        ),
    )
    target = torch.tensor([int(target_edge)], dtype=torch.long, device=device_obj)
    errors: list[str] = []
    explanation = None
    for target_argument in (target, None):
        try:
            if target_argument is None:
                explanation = explainer(x, edge_index)
            else:
                explanation = explainer(x, edge_index, target=target_argument)
            break
        except Exception as exc:  # noqa: BLE001 - PyG API differs across versions
            errors.append(f"{type(exc).__name__}: {exc}")
    if explanation is None:
        raise RuntimeError("PyG Explainer/GNNExplainer failed: " + " | ".join(errors))
    edge_mask = getattr(explanation, "edge_mask", None)
    if edge_mask is None:
        raise RuntimeError("GNNExplainer returned no edge_mask")
    edge_mask_np = edge_mask.detach().cpu().numpy().astype("float64").reshape(-1)
    node_feature_mask = getattr(explanation, "node_mask", None)
    node_mask_np = (
        node_feature_mask.detach().cpu().numpy().astype("float64")
        if node_feature_mask is not None
        else None
    )
    top_edges = np.argsort(-edge_mask_np)[:5].tolist()
    return {
        "explainer": "gnn_explainer",
        "target_edge": int(target_edge),
        "epochs": int(epochs),
        "seed": int(seed),
        "edge_mask": edge_mask_np,
        "node_feature_mask": node_mask_np,
        "target_edge_mask_value": float(edge_mask_np[int(target_edge)]),
        "top_edges": [{"edge_index": int(value), "mask": float(edge_mask_np[int(value)])} for value in top_edges],
        "target_used": "with_target" if errors == [] else "without_target",
        "api_errors": errors,
    }


__all__ = ["gnn_explainer"]
