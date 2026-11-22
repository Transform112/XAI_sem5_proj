"""Graph structure-existence checks (T4 / S-D1).

Why this module exists
----------------------
The whole study measures explanations of edges in a host-flow graph. If the
graph has no exploitable structure (isolated hosts, no repeated host pairs, no
neighbourhoods), graph lift and structural drift analyses are meaningless. This
module records the structural facts that decide whether the graph formulation is
defensible, per D-020/D-023:

* global and per-period host counts and host sets (with a stable set hash),
* host-set turnover and recurrence between consecutive periods,
* degree distribution (incident-flow degree per host),
* connected components (sizes, giant-component share),
* repeated directed host pairs (share of pairs / share of edges),
* self-loops,
* density (directed unique pairs / possible pairs),
* 2-hop neighbourhood size per edge (exact bitmask computation when the host
  count is <= 64, which covers the NF-v3 datasets) and the ratio to the
  connected-component size,
* benign-only periods (D-020: UNSW period 2 has no attack flows, so no attack
  anchors can be drawn there - the scorecard states this explicitly).

D-023 interpretation: the giant-component check is evaluated on the **per-period
graph** (the actual model input); the union-across-time components are reported
as a diagnostic. The 0.90 hard threshold is applied when a period has >= 100
hosts; below that the scorecard emits ``flag`` instead of ``fail`` because small
capture testbeds (UNSW: 44 hosts, two /24 capture subnets) genuinely produce
several components.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from src.utils.logging import get_logger

LOGGER = get_logger("xdrift.structure")

#: Per-period 2-hop computation is exact only while the host set fits one uint64.
MAX_HOSTS_FOR_EXACT_2HOP = 64


# ---------------------------------------------------------------------------
# Low-level graph statistics
# ---------------------------------------------------------------------------
def _component_info(n_hosts: int, src: np.ndarray, dst: np.ndarray) -> dict[str, Any]:
    if n_hosts == 0:
        return {"n_components": 0, "sizes": [], "giant_size": 0, "giant_fraction": 0.0, "labels": np.array([])}
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    adjacency = sparse.coo_matrix((np.ones(len(rows), dtype="int8"), (rows, cols)), shape=(n_hosts, n_hosts)).tocsr()
    n_components, labels = sparse.csgraph.connected_components(adjacency, directed=False)
    sizes = np.bincount(labels, minlength=n_components)
    giant = int(sizes.max()) if sizes.size else 0
    return {
        "n_components": int(n_components),
        "sizes": sorted((int(size) for size in sizes), reverse=True),
        "giant_size": giant,
        "giant_fraction": (giant / n_hosts) if n_hosts else 0.0,
        "labels": labels,
    }


def _degree_stats(n_hosts: int, src: np.ndarray, dst: np.ndarray) -> dict[str, Any]:
    """Incident-flow degree per host (each flow contributes 1 to its two hosts)."""
    if n_hosts == 0:
        return {"mode": "undirected_pair", "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0}
    degrees = np.bincount(np.concatenate([src, dst]), minlength=n_hosts)
    return {
        "mode": "undirected_pair",
        "median": float(np.median(degrees)),
        "mean": float(np.mean(degrees)),
        "p90": float(np.percentile(degrees, 90)),
        "max": int(degrees.max()),
    }


def _repeated_host_pairs(n_hosts: int, src: np.ndarray, dst: np.ndarray, n_edges: int) -> dict[str, Any]:
    if n_edges == 0 or n_hosts == 0:
        return {
            "unique_directed_pairs": 0,
            "unique_undirected_pairs": 0,
            "fraction_pairs": 0.0,
            "fraction_edges": 0.0,
            "max_flows_per_pair": 0,
        }
    directed_codes = src.astype("int64") * n_hosts + dst.astype("int64")
    _, directed_counts = np.unique(directed_codes, return_counts=True)
    low = np.minimum(src, dst)
    high = np.maximum(src, dst)
    undirected_codes = low.astype("int64") * n_hosts + high.astype("int64")
    unique_undirected = int(np.unique(undirected_codes).size)
    repeated = directed_counts > 1
    return {
        "unique_directed_pairs": int(len(directed_counts)),
        "unique_undirected_pairs": unique_undirected,
        "fraction_pairs": float(repeated.mean()),
        "fraction_edges": float(directed_counts[repeated].sum() / n_edges),
        "max_flows_per_pair": int(directed_counts.max()),
        "median_flows_per_repeated_pair": float(np.median(directed_counts[repeated])) if repeated.any() else 0.0,
    }


def _self_loops(src: np.ndarray, dst: np.ndarray, n_edges: int) -> dict[str, Any]:
    count = int(np.sum(src == dst))
    return {"count": count, "fraction": (count / n_edges) if n_edges else 0.0}


def _density(n_hosts: int, repeated: Mapping[str, Any], n_edges: int) -> dict[str, Any]:
    possible_directed = n_hosts * (n_hosts - 1) if n_hosts > 1 else 0
    possible_undirected = n_hosts * (n_hosts - 1) / 2 if n_hosts > 1 else 0
    return {
        "n_edges": int(n_edges),
        "possible_directed_pairs": int(possible_directed),
        "density_directed_unique": (
            float(repeated["unique_directed_pairs"] / possible_directed) if possible_directed else 0.0
        ),
        "density_undirected_unique": (
            float(repeated["unique_undirected_pairs"] / possible_undirected) if possible_undirected else 0.0
        ),
    }


def _two_hop_stats(
    n_hosts: int,
    src: np.ndarray,
    dst: np.ndarray,
    component_labels: np.ndarray,
    component_sizes: Sequence[int],
    n_edges: int,
) -> dict[str, Any]:
    """Exact 2-hop host-set size per edge using uint64 bitmasks (hosts <= 64)."""
    base: dict[str, Any] = {"available": n_hosts <= MAX_HOSTS_FOR_EXACT_2HOP and n_hosts > 0}
    if not base["available"]:
        base["reason"] = f"exact 2-hop bitmask computation requires <= {MAX_HOSTS_FOR_EXACT_2HOP} hosts"
        return base
    adjacency: list[set[int]] = [set() for _ in range(n_hosts)]
    for source, target in zip(src.tolist(), dst.tolist()):
        adjacency[source].add(target)
        adjacency[target].add(source)
    masks = np.zeros(n_hosts, dtype="uint64")
    for host, neighbours in enumerate(adjacency):
        mask = 1 << host
        for neighbour in neighbours:
            mask |= 1 << neighbour
        masks[host] = np.uint64(mask)
    radius2: list[int] = []
    for host, neighbours in enumerate(adjacency):
        mask = int(masks[host])
        for neighbour in neighbours:
            mask |= int(masks[neighbour])
        radius2.append(mask)
    radius2_array = np.array(radius2, dtype="uint64")
    if not hasattr(np, "bitwise_count"):
        base["reason"] = "numpy.bitwise_count unavailable (needs numpy >= 2.0)"
        base["available"] = False
        return base
    union = radius2_array[src] | radius2_array[dst]
    counts = np.bitwise_count(union)
    component_size_per_edge = np.asarray(component_sizes, dtype="float64")[component_labels[src]]
    ratio = counts / np.maximum(component_size_per_edge, 1.0)
    base.update(
        {
            "non_empty_fraction": float(np.mean(counts > 0)),
            "median": float(np.median(counts)),
            "mean": float(np.mean(counts)),
            "p90": float(np.percentile(counts, 90)),
            "max": int(counts.max()) if counts.size else 0,
            "median_component_size": float(np.median(component_size_per_edge)),
            "median_ratio_to_component": float(np.median(ratio)),
        }
    )
    return base


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def _host_set_hash(hosts: Sequence[int]) -> str:
    array = np.sort(np.asarray(list(hosts), dtype="<i8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _benign_class_id(class_map: Mapping[str, int] | None) -> int | None:
    if not class_map:
        return None
    if "Benign" in class_map:
        return int(class_map["Benign"])
    return int(min(class_map.values()))


def _block(
    frame: pd.DataFrame,
    *,
    period_col: str,
    n_hosts_global: int,
    class_map: Mapping[str, int] | None,
    two_hop: bool = True,
) -> dict[str, Any]:
    src = frame["_host_src_id"].to_numpy(dtype="int64")
    dst = frame["_host_dst_id"].to_numpy(dtype="int64")
    n_edges = int(len(frame))
    hosts = np.unique(np.concatenate([src, dst])) if n_edges else np.array([], dtype="int64")
    n_hosts = int(max(hosts.max() + 1, n_hosts_global) if hosts.size else n_hosts_global)
    components = _component_info(n_hosts, src, dst)
    repeated = _repeated_host_pairs(n_hosts, src, dst, n_edges)
    block: dict[str, Any] = {
        "n_edges": n_edges,
        "n_hosts": int(hosts.size),
        "hosts": int(hosts.size),
        "host_set_sha256": _host_set_hash(hosts.tolist()),
        "host_ids": [int(value) for value in hosts.tolist()],
        "degree": _degree_stats(n_hosts, src, dst),
        "components": {
            "n_components": components["n_components"],
            "sizes": components["sizes"],
            "giant_size": components["giant_size"],
            "giant_fraction": components["giant_fraction"],
        },
        "repeated_host_pairs": repeated,
        "self_loops": _self_loops(src, dst, n_edges),
        "density": _density(n_hosts, repeated, n_edges),
    }
    if two_hop:
        block["two_hop"] = _two_hop_stats(
            n_hosts, src, dst, components["labels"], components["sizes"], n_edges
        )
    benign_id = _benign_class_id(class_map)
    if benign_id is not None and "_class_id" in frame.columns:
        class_ids = np.unique(frame["_class_id"].to_numpy(dtype="int64"))
        block["class_ids_present"] = [int(value) for value in class_ids.tolist()]
        block["benign_only"] = bool(len(class_ids) == 1 and class_ids[0] == benign_id)
    return block


def structure_report(
    frame: pd.DataFrame,
    *,
    class_map: Mapping[str, int] | None = None,
    period_col: str = "_period",
    include_two_hop: bool = True,
) -> dict[str, Any]:
    """Compute global + per-period structure statistics (JSON-ready)."""
    if "_host_src_id" not in frame.columns or "_host_dst_id" not in frame.columns:
        raise KeyError("frame must be cleaned first (missing _host_src_id/_host_dst_id)")
    n_hosts_global = int(max(frame["_host_src_id"].max(), frame["_host_dst_id"].max()) + 1)
    global_block = _block(
        frame, period_col=period_col, n_hosts_global=n_hosts_global, class_map=class_map,
        two_hop=include_two_hop,
    )

    periods: dict[str, Any] = {}
    previous_hosts: set[int] = set()
    recurrences: list[float] = []
    turnovers: list[float] = []
    giant_fractions: list[float] = []
    benign_only: list[int] = []
    period_values = np.unique(frame[period_col].to_numpy(dtype="int64")) if period_col in frame.columns else np.array([0])
    for period in period_values:
        subset = frame[frame[period_col] == period] if period_col in frame.columns else frame
        block = _block(
            subset,
            period_col=period_col,
            n_hosts_global=n_hosts_global,
            class_map=class_map,
            two_hop=include_two_hop,
        )
        current = set(int(value) for value in block["host_ids"])
        block["hosts_new_vs_all_previous"] = len(current - previous_hosts)
        block["empty"] = block["n_edges"] == 0
        if previous_hosts:
            overlap = len(current & previous_hosts)
            block["host_recurrence"] = (overlap / len(current)) if current else None
            union = len(current | previous_hosts)
            block["host_turnover"] = (1.0 - overlap / union) if union else None
            block["hosts_previous_overlap"] = overlap
            block["hosts_union_with_previous"] = union
            recurrences.append(block["host_recurrence"] if block["host_recurrence"] is not None else 0.0)
            turnovers.append(block["host_turnover"] if block["host_turnover"] is not None else 1.0)
        else:
            block["host_recurrence"] = None
            block["host_turnover"] = None
            block["hosts_previous_overlap"] = 0
            block["hosts_union_with_previous"] = len(current)
        block["repeated_pair_fraction"] = block["repeated_host_pairs"]["fraction_edges"]
        block["giant_fraction"] = block["components"]["giant_fraction"]
        giant_fractions.append(block["giant_fraction"])
        if block.get("benign_only"):
            benign_only.append(int(period))
        periods[str(int(period))] = block
        previous_hosts |= current

    report = {
        "schema_version": 1,
        "period_column": period_col,
        "global": global_block,
        "periods": periods,
        "n_periods": int(len(period_values)),
        "host_recurrence": {
            "min": float(min(recurrences)) if recurrences else None,
            "median": float(np.median(recurrences)) if recurrences else None,
            "max": float(max(recurrences)) if recurrences else None,
        },
        "host_turnover": {
            "min": float(min(turnovers)) if turnovers else None,
            "median": float(np.median(turnovers)) if turnovers else None,
            "max": float(max(turnovers)) if turnovers else None,
        },
        "giant_component_per_period": {
            "min": float(min(giant_fractions)) if giant_fractions else None,
            "median": float(np.median(giant_fractions)) if giant_fractions else None,
            "values": {str(int(period)): float(value) for period, value in zip(period_values, giant_fractions)},
        },
        "benign_only_periods": benign_only,
        "notes": [],
    }
    if benign_only:
        report["notes"].append(
            "benign-only period(s) "
            + ", ".join(str(period + 1) for period in benign_only)
            + ": no attack flows, so no attack-class anchors can be drawn there (D-020)."
        )
    report["notes"].append(
        "Giant-component checks use the per-period graph (D-023); union-across-time components "
        "are reported in 'global' as a diagnostic."
    )
    return report


__all__ = ["MAX_HOSTS_FOR_EXACT_2HOP", "structure_report"]
