"""S1: build per-period PyG graph artefacts from the frozen C1 state.

The graph contract (Doc 2 S1):
* nodes = global host ids (deduplicated per period, remapped to 0..N-1 locally;
  the global ids are stored on the object for neighbourhood analysis),
* ``x`` = constant ones vector (dimension ``graph.node_feature_dim``),
* ``edge_index`` = endpoint node ids (PyG convention ``[src, dst]``),
* ``edge_attr`` = processed edge features from the frozen, period-1-train-only
  preprocessing state (no IP / raw timestamp columns),
* ``y`` = multiclass class id in the frozen 10-class label space,
* train/val/test boolean masks from the frozen C1 split masks,
* sidecars for later stages: raw flow/row id, start timestamp, global host ids.

Nothing in this module decides splits, periods or scaling: it consumes the
frozen state and fails loudly if the freshly derived split hash does not match
the frozen ``split_manifest.json`` hash (D-042: no frozen-state changes).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.preprocess import Preprocessor
from src.data.registry import DatasetSpec
from src.utils.logging import get_logger
from src.utils.manifest import sha256_file, utc_now, write_json

LOGGER = get_logger("xdrift.graphs")

SPLITS: tuple[str, ...] = ("train", "val", "test")


def require_torch_geometric():
    """Return (torch, Data) or raise a clear error when the wheel is broken."""
    from src.utils.torch_support import torch_available, torch_unavailable_reason

    if not torch_available():
        raise RuntimeError("torch unavailable on this lane: " + str(torch_unavailable_reason()))
    import torch  # noqa: PLC0415
    from torch_geometric.data import Data  # noqa: PLC0415

    return torch, Data


# ---------------------------------------------------------------------------
# One period
# ---------------------------------------------------------------------------
def build_period_graph(
    frame: pd.DataFrame,
    splits: Any,
    preprocessor: Preprocessor,
    spec: DatasetSpec,
    period: int,
    *,
    node_feature_dim: int = 8,
    node_features: np.ndarray | None = None,
    node_feature_mode: str = "constant",
    period_col: str = "_period",
) -> tuple[Any, dict[str, Any]]:
    """Build the PyG ``Data`` object for one period and return (data, stats).

    ``node_features`` optionally supplies a causal node-feature matrix aligned
    with the sorted global host ids.  The default keeps the canonical
    E-GraphSAGE constant-ones nodes; step 05 can pass a reputation matrix
    instead.  Edge features always come from the frozen preprocessor.
    """
    torch, Data = require_torch_geometric()
    if period_col not in frame.columns:
        raise KeyError(f"{period_col!r} missing from frame; run assign_periods first")
    positions = np.flatnonzero(frame[period_col].to_numpy(dtype="int64") == int(period))
    if positions.size == 0:
        raise ValueError(f"period {period} has no rows")
    subset = frame.iloc[positions]
    src_global = subset["_host_src_id"].to_numpy(dtype="int64")
    dst_global = subset["_host_dst_id"].to_numpy(dtype="int64")
    node_global_ids = np.unique(np.concatenate([src_global, dst_global]))
    src_local = np.searchsorted(node_global_ids, src_global).astype("int64")
    dst_local = np.searchsorted(node_global_ids, dst_global).astype("int64")
    features = preprocessor.transform(subset)
    expected_features = list(preprocessor.feature_names)
    if set(expected_features) & {
        spec.column("host_src"),
        spec.column("host_dst"),
        spec.column("timestamp_start"),
        spec.column("timestamp_end"),
    }:
        raise RuntimeError("feature matrix contains leakage columns (IP/timestamp); refusing to build graphs")
    n_edges = int(len(subset))
    if node_features is None:
        x = np.ones((len(node_global_ids), int(node_feature_dim)), dtype="float32")
    else:
        x = np.asarray(node_features, dtype="float32")
        if x.ndim != 2 or x.shape[0] != len(node_global_ids):
            raise ValueError(
                f"node_features must have shape ({len(node_global_ids)}, d); got {x.shape}"
            )
        if not np.isfinite(x).all():
            raise ValueError("node_features contains non-finite values; refusing to build graph")
        node_feature_dim = int(x.shape[1])
    y = subset["_class_id"].to_numpy(dtype="int64")
    if y.min() < 0:
        raise RuntimeError("negative class id in graph")
    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(np.stack([src_local, dst_local]).astype("int64")),
        edge_attr=torch.from_numpy(features.astype("float32")),
        y=torch.from_numpy(y),
    )
    data.node_global_ids = torch.from_numpy(node_global_ids)
    data.edge_row_ids = torch.from_numpy(subset["_row_id"].to_numpy(dtype="int64"))
    data.edge_ts = torch.from_numpy(subset["_ts"].to_numpy(dtype="int64"))
    data.edge_host_src_global = torch.from_numpy(src_global)
    data.edge_host_dst_global = torch.from_numpy(dst_global)
    data.period = int(period)
    data.dataset = spec.name

    split_counts: dict[str, int] = {}
    for name in SPLITS:
        if period not in splits.masks:
            raise KeyError(f"frozen split masks have no period {period}")
        global_mask = np.asarray(splits.masks[period][name], dtype="int64")
        local = np.searchsorted(positions, global_mask)
        if not np.array_equal(positions[local], global_mask):
            raise RuntimeError("split mask is inconsistent with the computed period rows")
        boolean = np.zeros(n_edges, dtype=bool)
        boolean[local] = True
        setattr(data, f"{name}_mask", torch.from_numpy(boolean))
        split_counts[name] = int(len(local))
    if sum(split_counts.values()) != n_edges:
        raise RuntimeError(
            f"split masks cover {sum(split_counts.values())} edges but the period has {n_edges}"
        )
    n_classes = int(y.max()) + 1
    if spec.class_order and n_classes > len(spec.class_order):
        raise RuntimeError(
            f"label space has {n_classes} classes but the frozen class_order only defines "
            f"{len(spec.class_order)}"
        )
    stats = {
        "period": int(period) + 1,
        "n_edges": n_edges,
        "n_nodes": int(len(node_global_ids)),
        "n_features": int(features.shape[1]),
        "n_classes": n_classes,
        "class_counts": {str(int(key)): int(value) for key, value in zip(*np.unique(y, return_counts=True))},
        "split_counts": split_counts,
        "node_feature_dim": int(node_feature_dim),
        "node_feature_mode": str(node_feature_mode),
    }
    return data, stats


# ---------------------------------------------------------------------------
# All periods + manifest
# ---------------------------------------------------------------------------
def build_graphs(
    *,
    frame: pd.DataFrame,
    splits: Any,
    preprocessor: Preprocessor,
    spec: DatasetSpec,
    out_dir: str | Path,
    node_feature_dim: int = 8,
    overwrite: bool = False,
    expected_frozen_hash: str | None = None,
    frozen_hash: str | None = None,
    raw_csv: str | Path | None = None,
    preprocessor_state_path: str | Path | None = None,
    config_pins: Mapping[str, Any] | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Build and persist ``period_{1..T}.pt`` plus a hashed manifest."""
    torch, _ = require_torch_geometric()
    dataset_name = dataset or spec.name
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if expected_frozen_hash and frozen_hash and expected_frozen_hash != frozen_hash:
        raise RuntimeError(
            "derived split hash does not match the frozen split_manifest hash; "
            "refusing to build graph artefacts (D-042 frozen state)"
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "s1_graph_artifacts",
        "dataset": dataset_name,
        "generated_at_utc": utc_now(),
        "frozen_split_hash": frozen_hash,
        "split_hash_matches_frozen": bool(
            expected_frozen_hash is None or expected_frozen_hash == frozen_hash
        ),
        "node_feature_dim": int(node_feature_dim),
        "feature_names": list(preprocessor.feature_names),
        "feature_manifest_note": "feature names come from the frozen preprocessor state; no IP/timestamp columns",
        "periods": {},
        "pins": dict(config_pins or {}),
    }
    if raw_csv is not None and Path(raw_csv).is_file():
        manifest["raw_csv"] = {"path": str(raw_csv), "bytes": Path(raw_csv).stat().st_size, "sha256": sha256_file(raw_csv)}
    if preprocessor_state_path is not None and Path(preprocessor_state_path).is_file():
        manifest["preprocessor_state"] = {
            "path": str(preprocessor_state_path),
            "sha256": sha256_file(preprocessor_state_path),
        }
    period_records: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    start = time.monotonic()
    for period in sorted(splits.masks):
        target = directory / f"period_{period + 1}.pt"
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} exists; pass --overwrite to rebuild (frozen state should not change)")
        data, stats = build_period_graph(
            frame,
            splits,
            preprocessor,
            spec,
            period,
            node_feature_dim=node_feature_dim,
        )
        temp = directory / f".period_{period + 1}.pt.tmp"
        torch.save(data, temp)
        sha = sha256_file(temp)
        temp.replace(target)
        record = {**stats, "file": str(target), "bytes": target.stat().st_size, "sha256": sha}
        manifest["periods"][str(period + 1)] = record
        period_records.append(record)
        combined.update(sha.encode("ascii"))
        LOGGER.info(
            "graph period %s: %s edges, %s nodes, %s features -> %s",
            period + 1,
            f"{stats['n_edges']:,}",
            stats["n_nodes"],
            stats["n_features"],
            target.name,
        )
    manifest["total_bytes"] = sum(int(record["bytes"]) for record in period_records)
    manifest["artifact_fingerprint_sha256"] = combined.hexdigest()
    manifest["build_seconds"] = round(time.monotonic() - start, 3)
    tokens = (int(sum(record["n_edges"] for record in period_records)))
    manifest["total_edges"] = tokens
    write_json(directory / "graphs_manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Smoke subgraph helper
# ---------------------------------------------------------------------------
def subsample_data(data: Any, max_edges: int, seed: int = 42) -> Any:
    """Return a random edge-subgraph copy for smoke runs (masks preserved)."""
    torch, Data = require_torch_geometric()
    n_edges = int(data.num_edges)
    if n_edges <= max_edges:
        return data
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(n_edges, size=int(max_edges), replace=False))
    keep_t = torch.as_tensor(keep, dtype=torch.long)
    edge_index = data.edge_index[:, keep_t]
    nodes = torch.unique(edge_index)  # sorted
    remap = torch.full((int(data.num_nodes),), -1, dtype=torch.long)
    remap[nodes] = torch.arange(nodes.numel(), dtype=torch.long)
    new_edge_index = remap[edge_index]
    new = Data(
        x=data.x[nodes],
        edge_index=new_edge_index,
        edge_attr=data.edge_attr[keep_t],
        y=data.y[keep_t],
    )
    new.node_global_ids = data.node_global_ids[nodes]
    for name in SPLITS:
        mask = getattr(data, f"{name}_mask")[keep_t]
        setattr(new, f"{name}_mask", mask)
    new.edge_row_ids = data.edge_row_ids[keep_t]
    new.edge_ts = data.edge_ts[keep_t]
    new.edge_host_src_global = data.edge_host_src_global[keep_t]
    new.edge_host_dst_global = data.edge_host_dst_global[keep_t]
    new.period = getattr(data, "period", None)
    new.dataset = getattr(data, "dataset", None)
    new.subsampled_from_edges = n_edges
    return new


__all__ = ["SPLITS", "build_graphs", "build_period_graph", "require_torch_geometric", "subsample_data"]
