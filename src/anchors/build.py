"""S6 anchor builder: stably-correct test-mask edges with sampled 2-hop subgraphs.

Anchor contract (Doc 2 S6 + D-033):

* candidates are period-j **test-mask** edges that checkpoint ``M_j`` classifies
  correctly with softmax confidence >= ``confidence_min`` (0.9),
* stratified up to ``max_per_class`` (100) per class and ``max_per_period``
  (1000) per period,
* classes with fewer than ``rare_min_eligible`` (10) eligible anchors are merged
  into a single documented ``rare`` stratum instead of being dropped (D-033),
* each anchor stores a fixed-seed sampled 2-hop computation subgraph (fanout
  <= 10 per hop), its edge features, label, global host ids and the anchor id
  (raw flow/row id), so later explainers and neighbourhood-change analyses use
  the exact same computation graph,
* a retained-fraction table records, for every later checkpoint ``M_t``, whether
  the anchor is still classified correctly with its original class.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from src.utils.logging import get_logger

LOGGER = get_logger("xdrift.anchors")

DEFAULT_ANCHOR_CONFIG: dict[str, Any] = {
    "confidence_min": 0.9,
    "max_per_class": 100,
    "max_per_period": 1000,
    "rare_min_eligible": 10,
    "rare_stratum": "rare",
    "sampling_seed": 42,
    "num_hops": 2,
    "fanout": 10,
    # Dense host graphs (40-44 hosts, >1M edges) make the induced 2-hop subgraph
    # contain most of the period's edges; cap the stored/explained context
    # deterministically (target edge always kept) so anchor artifacts and S7
    # explanation cost stay bounded. Toy/sparse graphs are unaffected.
    "max_subgraph_edges": 2000,
}


def _build_adjacency(data: Any) -> tuple[dict[int, list[int]], np.ndarray, np.ndarray]:
    """Undirected adjacency lists (once per period) for bounded BFS sampling.

    Vectorised: the host graph has <=~44 nodes but >1M edges, so collapse the edge
    endpoint pairs with numpy (unique) instead of looping over every edge in Python.
    """
    src = np.asarray(data.edge_index[0].detach().cpu().numpy(), dtype=np.int64)
    dst = np.asarray(data.edge_index[1].detach().cpu().numpy(), dtype=np.int64)
    nodes = np.concatenate([src, dst])
    neighbours = np.concatenate([dst, src])
    pairs = np.unique(np.stack([nodes, neighbours], axis=1), axis=0)
    adjacency: dict[int, list[int]] = {}
    for source, target in pairs:
        adjacency.setdefault(int(source), []).append(int(target))
    return {node: adjacency[node] for node in sorted(adjacency)}, src, dst


def _sample_subgraph(
    data: Any,
    target_edge: int,
    *,
    adjacency: Mapping[int, Sequence[int]],
    src_list: Sequence[int],
    dst_list: Sequence[int],
    num_hops: int,
    fanout: int,
    rng: np.random.Generator,
    full_edge_ids: torch.Tensor | None = None,
    max_subgraph_edges: int | None = None,
) -> dict[str, Any]:
    """Sample a bounded BFS neighbourhood around an edge's two hosts."""
    import torch

    src = int(data.edge_index[0, target_edge])
    dst = int(data.edge_index[1, target_edge])
    node_set = {src, dst}
    frontier = [src, dst]
    for _hop in range(num_hops):
        next_frontier: list[int] = []
        for node in frontier:
            candidates = [value for value in adjacency.get(node, []) if value not in node_set]
            if len(candidates) > fanout:
                picked = rng.choice(len(candidates), size=fanout, replace=False)
                candidates = [candidates[int(position)] for position in picked]
            for neighbour in candidates:
                if neighbour not in node_set:
                    node_set.add(neighbour)
                    next_frontier.append(neighbour)
        frontier = next_frontier
        if not frontier:
            break
    node_list = np.asarray(sorted(node_set), dtype=np.int64)
    node_mask = np.zeros(int(data.x.shape[0]), dtype=bool)
    node_mask[node_list] = True
    src_np = np.asarray(src_list, dtype=np.int64)
    dst_np = np.asarray(dst_list, dtype=np.int64)
    # Vectorised sub-edge membership + local remap (no Python loop over edges).
    n_edges_total = int(src_np.shape[0])
    n_nodes_total = int(data.x.shape[0])
    edge_mask = node_mask[src_np] & node_mask[dst_np]
    sub_edges = np.flatnonzero(edge_mask)
    if max_subgraph_edges is not None and len(sub_edges) > int(max_subgraph_edges):
        # Deterministic bounded context: always keep the anchor edge, sample the
        # remaining context edges from the induced 2-hop subgraph (same rng stream).
        target_position = int(np.searchsorted(sub_edges, target_edge))
        others = np.delete(sub_edges, target_position)
        n_take = min(max(int(max_subgraph_edges) - 1, 1), len(others))
        if n_take < len(others):
            picked = np.sort(rng.choice(len(others), size=n_take, replace=False))
            others = others[picked]
        sub_edges = np.sort(np.concatenate(([target_edge], others)))
    all_edges = len(sub_edges) == n_edges_total
    all_nodes_sorted = len(node_list) == n_nodes_total and np.array_equal(
        node_list, np.arange(n_nodes_total, dtype=np.int64)
    )
    if all_edges:
        # Dense host graph: the sampled 2-hop neighbourhood contains every host, so
        # every anchor's computation graph is the whole frozen graph. Share the period
        # tensors across anchors instead of copying edge_attr (E x 50 floats) per
        # anchor; torch.save dedups shared storages, so files stay one graph copy.
        sub_edges_tensor = (
            full_edge_ids
            if full_edge_ids is not None
            else torch.arange(n_edges_total, dtype=torch.long)
        )
        sub_edge_attr = data.edge_attr
    else:
        sub_edges_tensor = torch.as_tensor(sub_edges, dtype=torch.long)
        sub_edge_attr = data.edge_attr[sub_edges_tensor]
    node_tensor = torch.as_tensor(node_list, dtype=torch.long)
    if all_edges and all_nodes_sorted:
        local_edge_index = data.edge_index.long()
        sub_node_global_ids = data.node_global_ids
    else:
        local_src = np.searchsorted(node_list, src_np[sub_edges])
        local_dst = np.searchsorted(node_list, dst_np[sub_edges])
        local_edge_index = torch.as_tensor(np.stack([local_src, local_dst]), dtype=torch.long)
        sub_node_global_ids = data.node_global_ids[node_tensor]
    target_local = int(np.searchsorted(sub_edges, target_edge))
    return {
        "sub_edge_index": local_edge_index,
        "sub_edge_attr": sub_edge_attr,
        "sub_edge_ids": sub_edges_tensor,
        "sub_node_global_ids": sub_node_global_ids,
        "target_local_index": torch.tensor([target_local], dtype=torch.long),
        "target_edge": torch.tensor([target_edge], dtype=torch.long),
        "n_sub_nodes": int(node_tensor.numel()),
        "n_sub_edges": int(sub_edges_tensor.numel()),
    }


def select_anchors(
    model: Any,
    data: Any,
    *,
    config: Mapping[str, Any] | None = None,
    device: str = "cpu",
    seed: int | None = None,
) -> dict[str, Any]:
    """Select anchors for one period graph from checkpoint predictions."""
    import torch

    cfg = {**DEFAULT_ANCHOR_CONFIG, **dict(config or {})}
    seed = int(cfg["sampling_seed"] if seed is None else seed)
    model.to(device).eval()
    # T31: head-only logits for test positions (no full-edge head; same values as
    # the pre-T31 full-logit slice, tested <=1e-6).
    from src.eval.predict import predict_logits

    test_mask = getattr(data, "test_mask")
    test_positions = torch.nonzero(test_mask, as_tuple=False).flatten()
    test_logits = predict_logits(
        model,
        data.x.to(device),
        data.edge_index.to(device),
        data.edge_attr.to(device),
        test_positions,
        device=device,
    ).cpu()
    probabilities = torch.softmax(test_logits, dim=-1)
    predictions = test_logits.argmax(dim=-1)
    confidences = probabilities.gather(1, predictions.unsqueeze(1)).squeeze(1)
    y_true = data.y[test_positions]
    eligible_mask = (predictions == y_true) & (confidences >= float(cfg["confidence_min"]))
    eligible = test_positions[eligible_mask].cpu().numpy()
    if len(eligible) == 0:
        return {
            "period": int(getattr(data, "period", -1)),
            "anchors": [],
            "n_eligible": 0,
            "n_selected": 0,
            "strata": {},
            "rare_classes": [],
            "class_counts": {},
        }
    labels = y_true[eligible_mask].cpu().numpy()
    rng = np.random.default_rng(seed)
    by_class: dict[int, np.ndarray] = {
        int(class_id): eligible[labels == class_id] for class_id in np.unique(labels)
    }
    rare_classes = [class_id for class_id, values in by_class.items() if len(values) < int(cfg["rare_min_eligible"])]
    strata: dict[str, list[int]] = {}
    for class_id, values in by_class.items():
        if class_id in rare_classes:
            continue
        take = min(len(values), int(cfg["max_per_class"]))
        strata[str(class_id)] = [int(value) for value in rng.choice(values, size=take, replace=False)]
    rare_values = np.concatenate([by_class[class_id] for class_id in rare_classes]) if rare_classes else np.array([], dtype="int64")
    if len(rare_values):
        take = min(len(rare_values), int(cfg["max_per_class"]))
        strata[str(cfg["rare_stratum"])] = [int(value) for value in rng.choice(rare_values, size=take, replace=False)]
    selected = [edge for values in strata.values() for edge in values]
    if len(selected) > int(cfg["max_per_period"]):
        capped: list[int] = []
        per_stratum = max(1, int(cfg["max_per_period"]) // max(len(strata), 1))
        for name, values in strata.items():
            capped.extend(values[:per_stratum])
        selected = capped[: int(cfg["max_per_period"])]
    # T40/F1: D-033 provenance must describe the POST-CAP selected set. Rebuild the
    # stratum membership from ``selected`` so sum(strata) == n_selected exactly.
    selected_set = {int(edge) for edge in selected}
    stratum_of: dict[int, str] = {}
    for name, values in strata.items():
        for edge in values:
            if int(edge) in selected_set:
                stratum_of[int(edge)] = name
    kept_strata: dict[str, int] = {name: 0 for name in strata}
    for name in stratum_of.values():
        kept_strata[name] += 1
    if sum(kept_strata.values()) != len(selected):
        raise RuntimeError(
            f"anchor strata accounting mismatch: sum(strata)={sum(kept_strata.values())} "
            f"!= n_selected={len(selected)}"
        )
    anchors: list[dict[str, Any]] = []
    adjacency, src_list, dst_list = _build_adjacency(data)
    test_positions_np = test_positions.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    confidences_np = confidences.cpu().numpy()
    full_edge_ids = torch.arange(int(data.edge_index.shape[1]), dtype=torch.long)
    for edge_index_value in sorted(selected):
        subgraph = _sample_subgraph(
            data,
            edge_index_value,
            adjacency=adjacency,
            src_list=src_list,
            dst_list=dst_list,
            num_hops=int(cfg["num_hops"]),
            fanout=int(cfg["fanout"]),
            rng=np.random.default_rng(seed + edge_index_value),
            full_edge_ids=full_edge_ids,
            max_subgraph_edges=int(cfg.get("max_subgraph_edges", 2000)),
        )
        label = int(data.y[edge_index_value])
        test_position = int(np.searchsorted(test_positions_np, int(edge_index_value)))
        predicted = int(predictions_np[test_position])
        confidence = float(confidences_np[test_position])
        anchors.append(
            {
                "anchor_id": int(data.edge_row_ids[edge_index_value]),
                "period_local_edge_index": int(edge_index_value),
                "period": int(getattr(data, "period", -1)),
                "label": label,
                "predicted_class": predicted,
                "confidence": confidence,
                "stratum": stratum_of.get(int(edge_index_value), "unknown"),
                "host_src_global": int(data.edge_host_src_global[edge_index_value]),
                "host_dst_global": int(data.edge_host_dst_global[edge_index_value]),
                "edge_ts": int(data.edge_ts[edge_index_value]),
                **subgraph,
            }
        )
    return {
        "period": int(getattr(data, "period", -1)),
        "anchors": anchors,
        "n_eligible": int(len(eligible)),
        "n_selected": len(anchors),
        "strata": kept_strata,
        "rare_classes": sorted(int(value) for value in rare_classes),
        "rare_stratum": str(cfg["rare_stratum"]),
        "class_counts": {
            str(int(key)): int(value) for key, value in zip(*np.unique(labels, return_counts=True))
        },
        "config": dict(cfg),
    }


def saved_anchor_summary(anchor_set: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-ready summary of an anchor set (no tensors)."""
    return {
        "period": anchor_set.get("period"),
        "n_eligible": anchor_set.get("n_eligible"),
        "n_selected": anchor_set.get("n_selected"),
        "strata": anchor_set.get("strata", {}),
        "rare_classes": anchor_set.get("rare_classes", []),
        "class_counts": anchor_set.get("class_counts", {}),
        "config": anchor_set.get("config", {}),
    }


def retained_fraction_table(
    anchor_sets: Mapping[int, Mapping[str, Any]],
    checkpoints: Mapping[int, Any],
    *,
    graphs: Mapping[int, Any] | None = None,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Retained-fraction table: is each anchor still correctly classified later?

    T31: memory-bounded (head-only logits at the anchor positions, no full-edge head)
    and, when ``graphs`` is supplied, each anchor is evaluated with later checkpoint
    models on the **anchor's own period graph** (the fixed input edge), not on a
    later period's different edge set.
    """
    import torch

    from src.eval.predict import predict_logits

    rows: list[dict[str, Any]] = []
    for period, anchor_set in sorted(anchor_sets.items()):
        anchors = anchor_set.get("anchors", [])
        if not anchors:
            continue
        positions = torch.as_tensor(
            [int(anchor["period_local_edge_index"]) for anchor in anchors], dtype=torch.long
        )
        for later_period, (model, pair_data) in sorted(checkpoints.items()):
            if later_period <= period:
                continue
            model.to(device).eval()
            data = graphs.get(period, pair_data) if graphs else pair_data
            anchor_logits = predict_logits(
                model,
                data.x.to(device),
                data.edge_index.to(device),
                data.edge_attr.to(device),
                positions,
                device=device,
            )
            total = len(anchors)
            retained = 0
            per_class: dict[str, dict[str, int]] = {}
            per_anchor_ok: list[bool] = []
            for index, anchor in enumerate(anchors):
                predicted = int(anchor_logits[index].argmax())
                ok = predicted == int(anchor["label"])
                per_anchor_ok.append(bool(ok))
                retained += int(ok)
                label = str(int(anchor["label"]))
                bucket = per_class.setdefault(label, {"retained": 0, "total": 0})
                bucket["retained"] += int(ok)
                bucket["total"] += 1
            rows.append(
                {
                    "anchor_period": int(period) + 1,
                    "later_checkpoint": int(later_period) + 1,
                    "retained": retained,
                    "total": total,
                    "retained_fraction": (retained / total) if total else 0.0,
                    "per_class": per_class,
                    # Per-anchor statuses let the canonical retention table report
                    # cumulative (non-increasing) retention per class/checkpoint.
                    "per_anchor": per_anchor_ok,
                }
            )
    return rows


__all__ = [
    "DEFAULT_ANCHOR_CONFIG",
    "retained_fraction_table",
    "saved_anchor_summary",
    "select_anchors",
]

