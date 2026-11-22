#!/usr/bin/env python
"""Step 05 - build frozen per-period graph snapshots.

This is the bridge between the tabular pipeline (steps 02-04) and GNN training.
It reads only frozen inputs and never re-cleans, re-splits or re-fits anything:

    cleaned table          datasets/processed/<dataset>/clean/
    preprocessor state      datasets/processed/<dataset>/clean/preprocessor_state.json
    split masks + manifest  datasets/splits/<dataset>/<strategy>/
    balance indices         datasets/processed/<dataset>/balance/

For every period it writes one PyG ``Data`` object with:

    x                 node features (constant ones, reputation, or both)
    edge_index        [source host, destination host]
    edge_attr         50 frozen preprocessed edge features
    y                 class id
    train/val/test    frozen boolean masks
    edge_row_ids      row ids into the cleaned table
    edge_ts           start timestamps (for drift analysis only)
    node_global_ids   stable host ids

The optional node-feature modes are:

    constant            eight constant ones (original E-GraphSAGE convention)
    reputation          eight causal, label-free host-history features
    constant+reputation both blocks concatenated (16 columns)

Reputation is computed from *prior periods only*.  Period 1 is a cold start;
a host that has no history gets the neutral vector.  No labels, current-period
statistics, test statistics or future periods are used.

The script also saves a manifest and five diagnostic figures.  It does not
train a model.  Bounded 2/3/4-hop contexts are a later explanation/training
variant; this step stores the full-period graph that those experiments can
sample from.

Usage
-----
    python src/main/05_build_graphs.py --dry-run
    python src/main/05_build_graphs.py --dataset NF-UNSW-NB15-v3
    python src/main/05_build_graphs.py --node-features constant,reputation
    python src/main/05_build_graphs.py --node-features constant+reputation

On a machine without torch/PyG, use ``--dry-run`` to validate the node-feature
and split contracts without writing graphs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

try:
    import numpy as np
    import pandas as pd

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless Kaggle/SSH
        import matplotlib.pyplot as plt
    except ImportError:  # --dry-run does not need figures
        plt = None

    from src.data.graphs import build_period_graph, require_torch_geometric
    from src.data.preprocess import Preprocessor
    from src.data.registry import load_dataset_spec, resolve_dataset_names
    from src.data.splits import SplitResult, load_split_masks
    from src.utils.config import load_config
    from src.utils.manifest import sha256_file, sha256_text, write_json
    from src.utils.paths import load_paths
except ImportError as exc:  # pragma: no cover - only on a bare interpreter
    sys.exit(
        f"Missing dependency: {exc}\n"
        "Install the project requirements with:\n"
        "  python -m pip install -r platforms/requirements/common.txt"
    )


ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG = "configs/train.yaml"
AUDIT_CONFIG = "configs/audit.yaml"
C1_SPLIT_HASH = "010a80d955802a2f62dedf472a3274e1afee17fb4b21c57541fe428b8feb47ff"
C1_EDGE_COUNTS = (1_131_909, 304_675, 304_675, 304_675, 304_675)
C1_HOST_COUNTS = (40, 42, 42, 42, 42)
FOUR_NF_V3_DATASETS = (
    "NF-UNSW-NB15-v3",
    "NF-ToN-IoT-v3",
    "NF-BoT-IoT-v3",
    "NF-CSE-CIC-IDS2018-v3",
)
# Deterministic 20% sample of the optional CIC dataset.  It is a diagnostic
# replication, never a replacement for the full paper protocol.
SAMPLE_NF_V3_DATASETS = ("NF-CSE-CIC-IDS2018-v3-sample",)
KNOWN_NF_V3_DATASETS = FOUR_NF_V3_DATASETS + SAMPLE_NF_V3_DATASETS

REPUTATION_FEATURES = (
    "is_new_host",
    "log1p_prior_flows",
    "log1p_prior_bytes",
    "log1p_prior_degree",
    "log1p_prior_ports",
    "log1p_prior_protocols",
    "log1p_bytes_per_flow",
    "activity_z",
)
MODE_CHOICES = ("constant", "reputation", "constant+reputation")
BLUE = "#2f6f9f"
ORANGE = "#b45f4d"
GREEN = "#3a7d5d"
GREY = "#6b7c93"


# ---------------------------------------------------------------------------
# Small notebook-style output helpers
# ---------------------------------------------------------------------------
def cell(number: int, total: int, title: str) -> None:
    print()
    print("=" * 78)
    print(f"CELL {number:02d}/{total:02d} | {title}")
    print("=" * 78)


def info(text: str) -> None:
    print(f"[INFO] {text}")


def ok(text: str, detail: str = "") -> None:
    print(f"[OK  ] {text}" + (f" {detail}" if detail else ""))


def warn(text: str, detail: str = "") -> None:
    print(f"[WARN] {text}" + (f" {detail}" if detail else ""))


def fail(text: str, detail: str = "") -> None:
    print(f"[FAIL] {text}" + (f" {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen per-period PyG graphs with optional host reputation."
    )
    parser.add_argument("--dataset", default="all", help="comma-separated names or 'all'")
    parser.add_argument("--config", default=TRAIN_CONFIG, help="training/graph YAML")
    parser.add_argument("--audit-config", default=AUDIT_CONFIG, help="audit/split YAML")
    parser.add_argument("--root", default=None, help="project root (default: auto-detect)")
    parser.add_argument("--lane", default=None, help="path lane (cpu, kaggle, ...)")
    parser.add_argument(
        "--node-features",
        default="constant",
        help="comma-separated modes: constant, reputation, constant+reputation",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing graphs")
    parser.add_argument("--strict", action="store_true", help="missing input is an error")
    parser.add_argument("--no-hash", action="store_true", help="skip per-file SHA-256")
    parser.add_argument("--no-figures", action="store_true", help="skip diagnostic figures")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and node features without writing .pt files",
    )
    parser.add_argument("--debug", action="store_true", help="print tracebacks")
    return parser.parse_args(argv)


def choose_datasets(value: str, root: Path) -> list[str]:
    names = resolve_dataset_names(value, project_root=root)
    unknown = [name for name in names if name not in KNOWN_NF_V3_DATASETS]
    if unknown:
        raise ValueError(f"not an NF-v3 dataset: {unknown}")
    return names


def parse_modes(value: str) -> list[str]:
    modes = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [mode for mode in modes if mode not in MODE_CHOICES]
    if unknown:
        raise ValueError(f"unknown node-feature mode(s): {unknown}; choose {MODE_CHOICES}")
    return modes


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def cleaned_paths(processed_dir: Path) -> tuple[Path, Path]:
    parquet = processed_dir / "clean" / "cleaned.parquet"
    pickle = processed_dir / "clean" / "cleaned.pkl"
    state = processed_dir / "clean" / "preprocessor_state.json"
    table = parquet if parquet.exists() else pickle
    if not table.exists():
        raise FileNotFoundError(f"no cleaned table under {processed_dir / 'clean'}")
    if not state.exists():
        raise FileNotFoundError(f"no preprocessor state at {state}")
    return table, state


def load_cleaned(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_pickle(path)


def load_split_result(dataset_dir: Path, strategy: str) -> tuple[SplitResult, dict[str, Any], str]:
    manifest_path = dataset_dir / strategy / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no split manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    masks = {
        period: load_split_masks(dataset_dir / strategy, period + 1)
        for period in range(int(manifest.get("n_periods", 5)))
    }
    result = SplitResult(
        masks=masks,
        stats=manifest.get("periods", {}),
        fractions=manifest.get("fractions", {}),
        seed=int(manifest.get("seed", 42)),
        mode=str(manifest.get("mode", "grouped")),
    )
    return result, manifest, str(manifest.get("frozen_hash", ""))


def load_train_indices(balance_dir: Path, period: int) -> tuple[np.ndarray, dict[str, Any]]:
    summary_path = balance_dir / "balance_summary.json"
    indices_path = balance_dir / f"period_{period + 1}_train_indices.npz"
    if not summary_path.exists() or not indices_path.exists():
        raise FileNotFoundError(f"missing balance outputs under {balance_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(indices_path) as payload:
        indices = np.asarray(payload["indices"], dtype="int64")
    return indices, summary


# ---------------------------------------------------------------------------
# Causal reputation features
# ---------------------------------------------------------------------------
def reputation_matrix(
    frame: pd.DataFrame,
    *,
    period: int,
    node_global_ids: np.ndarray,
    spec: Any,
) -> np.ndarray:
    """Return (n_nodes, 8) causal host-history features.

    All statistics come from periods strictly before ``period``.  Period 1
    has no history and therefore only sets ``is_new_host``.
    """
    ids = np.asarray(node_global_ids, dtype="int64")
    features = np.zeros((len(ids), len(REPUTATION_FEATURES)), dtype="float32")

    # No history: every host is new and the rest of the vector is neutral.
    if period <= 0:
        features[:, 0] = 1.0
        return features

    history = frame.loc[frame["_period"].to_numpy(dtype="int64") < int(period)]
    if history.empty:
        features[:, 0] = 1.0
        return features

    src = history["_host_src_id"].to_numpy(dtype="int64")
    dst = history["_host_dst_id"].to_numpy(dtype="int64")
    in_bytes = history["IN_BYTES"].to_numpy(dtype="float64")
    out_bytes = history["OUT_BYTES"].to_numpy(dtype="float64")
    total_bytes = np.nan_to_num(in_bytes, nan=0.0) + np.nan_to_num(out_bytes, nan=0.0)
    dst_port = history[spec.column("port_dst")].to_numpy(dtype="float64")
    src_port = history[spec.column("port_src")].to_numpy(dtype="float64")
    protocol = history[spec.column("protocol")].to_numpy(dtype="float64")

    max_id = int(max(int(ids.max()) if len(ids) else -1, int(src.max()) if len(src) else -1, int(dst.max()) if len(dst) else -1))
    flows = np.bincount(src, minlength=max_id + 1) + np.bincount(dst, minlength=max_id + 1)
    bytes_sum = np.bincount(src, weights=total_bytes, minlength=max_id + 1) + np.bincount(
        dst, weights=total_bytes, minlength=max_id + 1
    )

    for row, host in enumerate(ids):
        mask = (src == host) | (dst == host)
        if not mask.any():
            features[row, 0] = 1.0
            continue
        peers = np.union1d(src[mask], dst[mask])
        peers = peers[peers != host]
        ports = np.union1d(
            dst_port[mask & (src == host)],
            src_port[mask & (dst == host)],
        )
        protocols = np.unique(protocol[mask])
        n_flows = int(flows[host])
        n_bytes = float(bytes_sum[host])
        features[row, 1] = np.log1p(max(n_flows, 0))
        features[row, 2] = np.log1p(max(n_bytes, 0.0))
        features[row, 3] = np.log1p(len(peers))
        features[row, 4] = np.log1p(len(ports))
        features[row, 5] = np.log1p(len(protocols))
        features[row, 6] = np.log1p(n_bytes / max(n_flows, 1))

    seen = features[:, 1] > 0
    if seen.any():
        activity = features[seen, 1]
        spread = float(activity.std())
        if spread > 0:
            features[seen, 7] = (activity - float(activity.mean())) / spread
    return features


def node_feature_block(mode: str, *, node_global_ids: np.ndarray, frame: pd.DataFrame,
                       period: int, spec: Any, constant_dim: int = 8) -> tuple[np.ndarray, list[str]]:
    if mode == "constant":
        return np.ones((len(node_global_ids), int(constant_dim)), dtype="float32"), [
            f"constant_{index}" for index in range(int(constant_dim))
        ]
    reputation = reputation_matrix(
        frame, period=period, node_global_ids=node_global_ids, spec=spec
    )
    if mode == "reputation":
        return reputation, list(REPUTATION_FEATURES)
    constants = np.ones((len(node_global_ids), int(constant_dim)), dtype="float32")
    features = np.concatenate([constants, reputation], axis=1)
    names = [f"constant_{index}" for index in range(int(constant_dim))] + list(REPUTATION_FEATURES)
    return features, names


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_period(
    *,
    dataset: str,
    period: int,
    frame: pd.DataFrame,
    data: Any,
    node_features: np.ndarray,
    expected_edge_features: int,
    expected_hash: str | None,
    frozen_hash: str,
) -> dict[str, Any]:
    """Hard structural checks before a graph is accepted."""
    torch, _ = require_torch_geometric()
    positions = np.flatnonzero(frame["_period"].to_numpy(dtype="int64") == period)
    n_edges = int(len(positions))
    n_nodes = int(data.x.shape[0])

    if device_is_cuda(data.x):
        raise RuntimeError("graph unexpectedly on CUDA; CPU graphs only")
    if not torch.isfinite(data.edge_attr).all():
        raise RuntimeError(f"period {period + 1}: non-finite edge features")
    if not np.isfinite(node_features).all():
        raise RuntimeError(f"period {period + 1}: non-finite node features")
    if node_features.shape[1] != int(data.x.shape[1]):
        raise RuntimeError(f"period {period + 1}: node features and x have different widths")
    names = list(getattr(data, "node_feature_names", []))
    if names and len(names) != node_features.shape[1]:
        raise RuntimeError(f"period {period + 1}: {len(names)} node-feature names for {node_features.shape[1]} columns")
    if "is_new_host" in names:
        column = names.index("is_new_host")
        values = np.unique(node_features[:, column])
        if not set(float(v) for v in values).issubset({0.0, 1.0}):
            raise RuntimeError(f"period {period + 1}: is_new_host is not binary")
        if period == 0 and not np.all(node_features[:, column] == 1.0):
            raise RuntimeError(f"period {period + 1}: cold-start is_new_host must be 1 for every host")
    if tuple(data.edge_row_ids.tolist()) != tuple(frame.iloc[positions]["_row_id"].tolist()):
        raise RuntimeError(f"period {period + 1}: edge row ids do not match cleaned rows")
    if len(set(map(int, data.edge_row_ids.tolist()))) != n_edges:
        raise RuntimeError(f"period {period + 1}: duplicate edge row ids")
    if int(data.y.min()) < 0:
        raise RuntimeError(f"period {period + 1}: negative class id")
    if data.edge_attr.shape[1] != expected_edge_features:
        raise RuntimeError(
            f"period {period + 1}: expected {expected_edge_features} edge features, "
            f"got {data.edge_attr.shape[1]} from the fitted preprocessor"
        )

    split_sum = 0
    for name in ("train", "val", "test"):
        mask = getattr(data, f"{name}_mask").to(torch.bool)
        split_sum += int(mask.sum())
    if split_sum != n_edges:
        raise RuntimeError(f"period {period + 1}: split masks cover {split_sum}/{n_edges} edges")

    if expected_hash and frozen_hash and expected_hash != frozen_hash:
        raise RuntimeError(
            f"{dataset}: split hash {frozen_hash} does not match expected {expected_hash}"
        )
    if dataset == "NF-UNSW-NB15-v3":
        expected_edges = C1_EDGE_COUNTS[period]
        expected_nodes = C1_HOST_COUNTS[period]
        if n_edges != expected_edges:
            raise RuntimeError(
                f"period {period + 1}: {n_edges} edges, expected {expected_edges} (C1)"
            )
        if n_nodes != expected_nodes:
            raise RuntimeError(
                f"period {period + 1}: {n_nodes} hosts, expected {expected_nodes} (C1)"
            )

    return {
        "n_edges": n_edges,
        "n_nodes": n_nodes,
        "n_features": int(data.edge_attr.shape[1]),
        "node_feature_dim": int(data.x.shape[1]),
    }


def device_is_cuda(tensor: Any) -> bool:
    try:
        return bool(tensor.is_cuda)
    except AttributeError:
        return False


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(
    *,
    frame: pd.DataFrame,
    dataset: str,
    split_result: SplitResult,
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    periods = list(range(5))
    edges = []
    nodes = []
    split_counts = {"train": [], "val": [], "test": []}
    host_sets: list[set[int]] = []
    degrees: list[np.ndarray] = []

    for period in periods:
        positions = np.flatnonzero(frame["_period"].to_numpy(dtype="int64") == period)
        subset = frame.iloc[positions]
        src = subset["_host_src_id"].to_numpy(dtype="int64")
        dst = subset["_host_dst_id"].to_numpy(dtype="int64")
        ids = np.unique(np.concatenate([src, dst]))
        host_sets.append(set(map(int, ids.tolist())))
        edges.append(int(len(subset)))
        nodes.append(int(len(ids)))
        degrees.append(np.bincount(np.concatenate([src, dst])))
        for name in split_counts:
            split_counts[name].append(int(len(split_result.masks[period][name])))

    figures: list[Path] = []

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels = [f"P{period + 1}" for period in periods]
    bars = ax.bar(labels, edges, color=BLUE)
    for bar, value, n_nodes in zip(bars, edges, nodes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,}\n({n_nodes} hosts)",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("edges (flows)")
    ax.set_title(f"{dataset}: graph size per period")
    ax.margins(y=0.18)
    _style(ax)
    path = out_dir / "fig01_graph_size.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bottom = np.zeros(len(periods))
    for name, colour in (("train", BLUE), ("val", ORANGE), ("test", GREEN)):
        values = np.asarray(split_counts[name])
        ax.bar(labels, values, bottom=bottom, label=name, color=colour)
        bottom += values
    ax.set_ylabel("edges")
    ax.set_title(f"{dataset}: frozen split support")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _style(ax)
    path = out_dir / "fig02_split_support.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax = axes[0]
    degree = degrees[0]
    positive = degree[degree > 0]
    bins = np.logspace(0, np.log10(max(2, positive.max() + 1)), 25)
    ax.hist(positive, bins=bins, color=BLUE)
    ax.set_xscale("log")
    ax.set_xlabel("flows per host (P1)")
    ax.set_ylabel("hosts")
    ax.set_title("Host activity")
    _style(ax)

    ax = axes[1]
    shares = []
    for left, right in zip(host_sets[:-1], host_sets[1:]):
        union = left | right
        shares.append(len(left & right) / max(len(union), 1))
    ax.plot([f"P{index + 1}->P{index + 2}" for index in range(len(shares))], shares,
            marker="o", color=ORANGE)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Jaccard(host sets)")
    ax.set_title("Host overlap across periods")
    _style(ax)

    path = out_dir / "fig03_hosts.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    new_fraction = []
    for period in periods:
        ids = host_sets[period]
        if period == 0:
            new_fraction.append(1.0)
        else:
            old = set.union(*host_sets[:period])
            new_fraction.append(len(ids - old) / max(len(ids), 1))
    ax.bar(labels, new_fraction, color=GREEN)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of hosts")
    ax.set_title(f"{dataset}: new hosts (reputation cold start)")
    _style(ax)
    path = out_dir / "fig04_new_hosts.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axis("off")
    ax.text(0.02, 0.85, f"{dataset}: graph contract", fontsize=13, weight="bold")
    lines = [
        f"periods: {len(periods)}",
        f"edges: {', '.join(f'{value:,}' for value in edges)}",
        f"hosts: {', '.join(str(value) for value in nodes)}",
        "edge features: 50 (frozen preprocessor)",
        "node modes: constant / reputation / constant+reputation",
        "reputation: prior periods only, label-free",
        "target edge kept in every graph",
    ]
    for index, line in enumerate(lines):
        ax.text(0.02, 0.68 - index * 0.09, line, fontsize=10)
    path = out_dir / "fig05_contract.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)
    return figures


def _style(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------
def build_dataset(
    *,
    dataset: str,
    root: Path,
    paths: Any,
    graph_config: Mapping[str, Any],
    audit_config: Any,
    modes: list[str],
    overwrite: bool,
    no_hash: bool,
    no_figures: bool,
    dry_run: bool,
    strict: bool,
) -> dict[str, Any]:
    processed_dir = paths.processed_dataset(dataset)
    split_root = paths.dataset_splits(dataset)
    strategy = str(audit_config.get("periods", {}).get("strategy", "session_aware_1_4"))
    dataset_split_dir = split_root / strategy

    try:
        table_path, state_path = cleaned_paths(processed_dir)
        split_result, split_manifest, frozen_hash = load_split_result(split_root, strategy)
        balance_dir = processed_dir / "balance"
    except FileNotFoundError as exc:
        if strict:
            raise
        warn(f"{dataset}: skipped", str(exc))
        return {"dataset": dataset, "status": "skipped", "reason": str(exc)}

    frame = load_cleaned(table_path)
    if "_period" not in frame.columns:
        raise RuntimeError("cleaned table has no _period column; run step 02 first")
    spec = load_dataset_spec(dataset, project_root=root)
    preprocessor = None
    if not dry_run:
        preprocessor = Preprocessor.load(spec, state_path)

    expected_hash = C1_SPLIT_HASH if dataset == "NF-UNSW-NB15-v3" else None
    if expected_hash and frozen_hash != expected_hash:
        raise RuntimeError(
            f"{dataset}: split manifest hash {frozen_hash!r} != signed C1 {expected_hash!r}"
        )
    info(f"{dataset}: {len(frame):,} rows, split hash {frozen_hash[:12]}...")

    per_mode: dict[str, Any] = {}
    for mode in modes:
        graph_dir = paths.dataset_graphs(dataset) / mode
        summary_dir = paths.results_root / "text" / "data_audit" / "graphs" / dataset
        figure_dir = paths.figures_root / "graphs" / dataset
        if not dry_run:
            graph_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)
        periods_payload: dict[str, Any] = {}

        for period in range(5):
            positions = np.flatnonzero(frame["_period"].to_numpy(dtype="int64") == period)
            subset = frame.iloc[positions]
            src = subset["_host_src_id"].to_numpy(dtype="int64")
            dst = subset["_host_dst_id"].to_numpy(dtype="int64")
            node_ids = np.unique(np.concatenate([src, dst]))
            node_features, feature_names = node_feature_block(
                mode,
                node_global_ids=node_ids,
                frame=frame,
                period=period,
                spec=spec,
                constant_dim=int(graph_config.get("node_feature_dim", 8)),
            )
            train_indices, balance_summary = load_train_indices(balance_dir, period)
            local = np.searchsorted(positions, train_indices)
            if len(local) and not np.array_equal(positions[local], train_indices):
                raise RuntimeError(f"period {period + 1}: balance indices do not align with cleaned rows")

            if dry_run:
                ok(
                    f"{dataset}/{mode}/P{period + 1}",
                    f"{len(positions):,} edges, {len(node_ids)} nodes, "
                    f"x={node_features.shape[1]}, train={len(train_indices):,}",
                )
                periods_payload[str(period + 1)] = {
                    "edges": int(len(positions)),
                    "nodes": int(len(node_ids)),
                    "node_feature_dim": int(node_features.shape[1]),
                    "node_feature_names": feature_names,
                    "selected_train_rows": int(len(train_indices)),
                }
                continue

            graph_path = graph_dir / f"period_{period + 1}.pt"
            if graph_path.exists() and not overwrite:
                raise FileExistsError(f"{graph_path} exists; pass --overwrite to replace it")

            data, stats = build_period_graph(
                frame,
                split_result,
                preprocessor,
                spec,
                period,
                node_feature_dim=node_features.shape[1],
                node_features=node_features,
                node_feature_mode=mode,
            )
            validate_period(
                dataset=dataset,
                period=period,
                frame=frame,
                data=data,
                node_features=node_features,
                expected_edge_features=len(preprocessor.feature_names),
                expected_hash=expected_hash,
                frozen_hash=frozen_hash,
            )

            torch, _ = require_torch_geometric()
            train_local = np.zeros(len(positions), dtype=bool)
            train_local[local] = True
            data.train_indices = torch.from_numpy(local.astype("int64"))
            data.train_mask_balanced = torch.from_numpy(train_local)
            data.selected_train_count = int(len(local))
            data.node_feature_mode = mode
            data.node_feature_names = feature_names
            data.split_manifest_hash = frozen_hash
            torch.save(data, graph_path)

            digest = None if no_hash else sha256_file(graph_path)
            periods_payload[str(period + 1)] = {
                **stats,
                "node_feature_mode": mode,
                "node_feature_names": feature_names,
                "selected_train_rows": int(len(local)),
                "graph_path": graph_path.relative_to(root).as_posix(),
                "graph_sha256": digest,
            }
            ok(
                f"{dataset}/{mode}/P{period + 1}",
                f"{len(positions):,} edges, {len(node_ids)} nodes, x={node_features.shape[1]}",
            )

        if dry_run:
            per_mode[mode] = {"status": "dry-run", "periods": periods_payload}
            continue

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "kind": "s1_graph_artifacts",
            "dataset": dataset,
            "node_feature_mode": mode,
            "node_feature_dim": int(next(iter(periods_payload.values()))["node_feature_dim"]),
            "node_feature_names": next(iter(periods_payload.values()))["node_feature_names"],
            "split_strategy": strategy,
            "frozen_split_hash": frozen_hash,
            "split_manifest": str((dataset_split_dir / "split_manifest.json").relative_to(root).as_posix()),
            "preprocessor_state": str(state_path.relative_to(root).as_posix()),
            "periods": periods_payload,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        manifest["manifest_sha256"] = sha256_text(
            json.dumps({key: manifest[key] for key in manifest if key != "manifest_sha256"}, sort_keys=True)
        )
        manifest_path = graph_dir / "graphs_manifest.json"
        write_json(manifest_path, manifest)

        figures: list[Path] = []
        if not no_figures and plt is not None:
            try:
                figures = make_figures(
                    frame=frame,
                    dataset=dataset,
                    split_result=split_result,
                    out_dir=figure_dir,
                )
            except Exception as exc:  # pragma: no cover - plotting is optional
                warn("figures skipped", str(exc))

        summary = {
            "schema_version": 1,
            "kind": "graph_build_summary",
            "dataset": dataset,
            "node_feature_mode": mode,
            "node_feature_dim": manifest["node_feature_dim"],
            "node_feature_names": manifest["node_feature_names"],
            "rows_cleaned": int(len(frame)),
            "frozen_split_hash": frozen_hash,
            "split_matches_c1": bool(expected_hash is None or frozen_hash == expected_hash),
            "periods": periods_payload,
            "manifest": str(manifest_path.relative_to(root).as_posix()),
            "figures": [str(path.relative_to(root).as_posix()) for path in figures],
        }
        write_json(summary_dir / f"{mode}.json", summary)
        per_mode[mode] = summary
        ok(f"{dataset}: {mode}", f"{len(periods_payload)} periods; manifest {manifest_path.name}")

    return {"dataset": dataset, "status": "ok", "modes": per_mode}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve() if args.root else ROOT
    started = time.time()

    cell(1, 3, "CONFIGURATION")
    config = load_config(args.config, project_root=root, lane=args.lane)
    audit_config = load_config(args.audit_config, project_root=root, lane=args.lane)
    paths = load_paths(lane=args.lane, project_root=root)
    names = choose_datasets(args.dataset, root)
    modes = parse_modes(args.node_features)
    graph_config = config.section("graph")
    info(f"project root {root}")
    ok("datasets", ", ".join(names))
    ok("node-feature modes", ", ".join(modes))
    ok("preprocessor edge dim", str(50))
    if args.dry_run:
        warn("dry-run", "no .pt files or manifests will be written")

    cell(2, 3, "GRAPH BUILD")
    results = []
    for dataset in names:
        try:
            result = build_dataset(
                dataset=dataset,
                root=root,
                paths=paths,
                graph_config=graph_config,
                audit_config=audit_config,
                modes=modes,
                overwrite=args.overwrite,
                no_hash=args.no_hash,
                no_figures=args.no_figures,
                dry_run=args.dry_run,
                strict=args.strict,
            )
            results.append(result)
        except Exception as exc:
            fail(f"{dataset}", str(exc))
            if args.debug:
                import traceback

                traceback.print_exc()
            if args.strict:
                return 1
            results.append({"dataset": dataset, "status": "failed", "error": str(exc)})

    cell(3, 3, "SUMMARY")
    built = [item for item in results if item.get("status") == "ok"]
    for item in built:
        modes_done = ", ".join(item.get("modes", {}))
        ok(item["dataset"], modes_done)
    failures = [item for item in results if item.get("status") == "failed"]
    if failures:
        for item in failures:
            fail(item["dataset"], item.get("error", "unknown error"))
        return 1
    if not built:
        warn("no graphs built", "check that cleaned datasets exist")
        return 2 if args.strict else 0
    print()
    print(f"Done in {time.time() - started:.1f}s")
    print("Next: train a model on one graph variant; 2/3/4-hop depth is a model config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
