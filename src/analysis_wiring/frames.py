"""Build the S8-S12 input frames for the authoritative analysis toolkit (D-044).

The metric definitions are **not** re-implemented here: this module only
assembles the long frames that :func:`analysis.tables.generate_all` consumes, by
calling ``analysis.metrics_drift`` / ``analysis.noise_floor`` /
``analysis.rq3`` for every number. The frame schemas deliberately mirror
``analysis.synthetic`` (the tested protocol-shaped fixtures), so the analysis
writers, validators and figures apply unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis import metrics_cl as mc
from src.analysis import metrics_drift as md
from src.analysis import noise_floor as nf
from src.utils.artifact_paths import infer_results_root, resolve_artifact_path

FEATURE_KS: tuple[int, ...] = (3, 5, 10)
EDGE_M = 5


class AnalysisWiringError(RuntimeError):
    """Raised when constructed frames violate the analysis contract."""


# ---------------------------------------------------------------------------
# S8: real drift + noise floors
# ---------------------------------------------------------------------------
def _pairwise_metrics(
    explanation_a: np.ndarray,
    explanation_b: np.ndarray,
    *,
    metric_set: str,
    ks: Sequence[int] = FEATURE_KS,
    edge_m: int = EDGE_M,
    keys_a: Sequence[Any] | None = None,
    keys_b: Sequence[Any] | None = None,
) -> dict[str, Any]:
    if metric_set == "feature":
        similarity = md.attribution_similarity(explanation_a, explanation_b, ks=tuple(ks))
        drift = md.attribution_drift(explanation_a, explanation_b, ks=tuple(ks))
        # docs/05 Ã‚Â§8 + src/analysis/config_defaults/drift.yaml: primary feature metric is
        # 1 - Jaccard(top-5 feature set) for IG attributions (all metrics still reported).
        primary = "drift_jaccard_top5"
    elif metric_set == "edge":
        similarity = md.edge_mask_similarity(explanation_a, explanation_b, m=edge_m, keys_a=keys_a, keys_b=keys_b)
        drift = md.edge_mask_drift(explanation_a, explanation_b, m=edge_m, keys_a=keys_a, keys_b=keys_b)
        primary = f"drift_edge_jaccard_top{edge_m}"
    else:
        raise AnalysisWiringError(f"unknown metric_set {metric_set!r}; expected 'feature' or 'edge'")
    row: dict[str, Any] = {}
    for key, value in similarity.items():
        row[f"sim_{key}"] = float(value)
    for key, value in drift.items():
        row[f"drift_{key}"] = float(value)
    if primary not in row:
        raise AnalysisWiringError(f"primary metric {primary!r} missing from analysis.metrics_drift output")
    row["sim_primary"] = float(row[primary.replace("drift_", "sim_", 1)])
    row["drift_primary"] = float(row[primary])
    return row


def build_real_drift_frame(
    explanations: Mapping[tuple[int, Any], np.ndarray],
    anchor_meta: Mapping[Any, Mapping[str, Any]],
    *,
    strategy: str,
    model: str,
    explainer: str,
    seed: int,
    metric_set: str = "feature",
    ks: Sequence[int] = FEATURE_KS,
    edge_m: int = EDGE_M,
) -> pd.DataFrame:
    """Long real-update drift frame: one row per (anchor, checkpoint_j, checkpoint_t).

    ``checkpoint_j`` is the anchor's **own period** (the canonical anchor set it
    belongs to), not every explanation checkpoint.  For a canonical anchor from
    period j the valid comparisons are t = j, j+1, â€¦, T (diagonal included); this
    fixes the S8 row multiplication where an anchor appeared under intermediate
    checkpoint_j values and produced 4,413 instead of 3,211 pairs/group.
    """
    rows: list[dict[str, Any]] = []
    grouped: dict[Any, dict[int, np.ndarray]] = {}
    for (checkpoint, anchor_id), array in explanations.items():
        grouped.setdefault(anchor_id, {})[int(checkpoint)] = np.asarray(array)
    for anchor_id in sorted(grouped, key=str):
        meta = anchor_meta.get(anchor_id)
        if meta is None:
            raise AnalysisWiringError(f"anchor {anchor_id!r} has an explanation but no anchor metadata")
        anchor_period = meta.get("anchor_period")
        if anchor_period is None:
            raise AnalysisWiringError(f"anchor {anchor_id!r} has no anchor_period metadata")
        anchor_period = int(anchor_period)
        if anchor_period not in grouped[anchor_id]:
            raise AnalysisWiringError(
                f"anchor {anchor_id!r} has no diagonal explanation at its anchor period {anchor_period}"
            )
        explanation_j = grouped[anchor_id][anchor_period]
        for checkpoint_t in sorted(grouped[anchor_id]):
            if checkpoint_t < anchor_period:
                continue
            explanation_t = grouped[anchor_id][checkpoint_t]
            row: dict[str, Any] = {
                "strategy": strategy,
                "model": model,
                "explainer": explainer,
                "seed": int(seed),
                "anchor_id": anchor_id,
                "class": str(meta.get("class", meta.get("label"))),
                "anchor_period": anchor_period,
                "checkpoint_j": int(anchor_period),
                "checkpoint_t": int(checkpoint_t),
            }
            row.update(
                _pairwise_metrics(
                    np.asarray(explanation_j),
                    np.asarray(explanation_t),
                    metric_set=metric_set,
                    ks=ks,
                    edge_m=edge_m,
                    keys_a=meta.get("sub_edge_ids"),
                    keys_b=meta.get("sub_edge_ids"),
                )
            )
            rows.append(row)
    columns = [
        "strategy", "model", "explainer", "seed", "anchor_id", "class", "anchor_period",
        "checkpoint_j", "checkpoint_t", "sim_cosine", "drift_cosine",
        "sim_jaccard_top3", "drift_jaccard_top3", "sim_jaccard_top5", "drift_jaccard_top5",
        "sim_jaccard_top10", "drift_jaccard_top10", "sim_spearman", "drift_spearman",
        "sim_edge_jaccard_top5", "drift_edge_jaccard_top5", "sim_edge_cosine", "drift_edge_cosine",
        "sim_primary", "drift_primary",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]


def build_model_null_noise_frame(
    reference: Mapping[Any, np.ndarray],
    reseeded: Mapping[Any, Mapping[Any, np.ndarray]],
    anchor_meta: Mapping[Any, Mapping[str, Any]],
    *,
    strategy: str,
    model: str,
    explainer: str,
    seed: int,
    metric_set: str = "feature",
    ks: Sequence[int] = FEATURE_KS,
    edge_m: int = EDGE_M,
    channel: str = nf.MODEL_NULL,
    cfg: nf.NoiseFloorConfig | None = None,
) -> pd.DataFrame:
    """S8.2 null-update noise frame via analysis.noise_floor.null_update_drift."""
    class_by_anchor = {anchor_id: str(meta.get("class", meta.get("label"))) for anchor_id, meta in anchor_meta.items()}
    frame = nf.null_update_drift(
        reference,
        reseeded,
        class_by_anchor=class_by_anchor,
        metric_set=metric_set,
        ks=tuple(ks),
        m=edge_m,
        cfg=cfg,
    )
    frame["strategy"] = strategy
    frame["model"] = model
    frame["explainer"] = explainer
    frame["seed"] = int(seed)
    frame["channel"] = channel
    return frame


def build_explainer_repeat_noise_frame(
    repeats: Mapping[Any, Sequence[np.ndarray]],
    anchor_meta: Mapping[Any, Mapping[str, Any]],
    *,
    strategy: str,
    model: str,
    explainer: str,
    metric_set: str = "edge",
    ks: Sequence[int] = FEATURE_KS,
    edge_m: int = EDGE_M,
    channel: str = nf.EXPLAINER_REPEAT,
    cfg: nf.NoiseFloorConfig | None = None,
) -> pd.DataFrame:
    """S8.1 explainer-repeat noise frame via analysis.noise_floor.explainer_repeat_drift."""
    class_by_anchor = {anchor_id: str(meta.get("class", meta.get("label"))) for anchor_id, meta in anchor_meta.items()}
    frame = nf.explainer_repeat_drift(
        repeats,
        class_by_anchor=class_by_anchor,
        metric_set=metric_set,
        ks=tuple(ks),
        m=edge_m,
        cfg=cfg,
    )
    frame["strategy"] = strategy
    frame["model"] = model
    frame["explainer"] = explainer
    frame["channel"] = channel
    return frame


# ---------------------------------------------------------------------------
# S10: structural neighbourhood change (2-hop host/edge-set Jaccard)
# ---------------------------------------------------------------------------
def _edge_rows_by_pair(graph: Any) -> dict[tuple[int, int], list[int]]:
    """Undirected host-pair -> edge-row-id list, built once per checkpoint graph."""
    mapping: dict[tuple[int, int], list[int]] = {}
    for source, target, row_id in zip(
        graph.edge_index[0].tolist(),
        graph.edge_index[1].tolist(),
        graph.edge_row_ids.tolist(),
    ):
        key = (min(int(source), int(target)), max(int(source), int(target)))
        mapping.setdefault(key, []).append(int(row_id))
    return mapping


def _two_hop_sets(
    graph: Any,
    host_ids: Sequence[int],
    *,
    fanout: int = 10,
    hops: int = 2,
    neighbours: Mapping[int, Sequence[int]] | None = None,
    edge_rows_by_pair: Mapping[tuple[int, int], Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Bounded BFS 2-hop host set and edge-row-id set around a set of hosts.

    T49 performance fix: pass the per-checkpoint adjacency / pair->edge-id index
    built once by :func:`build_neighbourhood_frame`; the old implementation rebuilt
    the 1.1M-edge adjacency and rescanned every edge for every anchor.
    """
    node_set = {int(value) for value in host_ids}
    frontier = sorted(node_set)
    if neighbours is None:
        neighbours = data_adjacency(graph)
    for _hop in range(int(hops)):
        next_frontier: list[int] = []
        for node in frontier:
            candidates = [value for value in neighbours.get(node, ()) if value not in node_set]
            for candidate in candidates[: int(fanout)]:
                node_set.add(candidate)
                next_frontier.append(candidate)
        frontier = sorted(next_frontier)
        if not frontier:
            break
    if edge_rows_by_pair is None:
        edge_rows_by_pair = _edge_rows_by_pair(graph)
    nodes = sorted(node_set)
    edges: set[int] = set()
    for position, source in enumerate(nodes):
        for target in nodes[position + 1 :]:
            edges.update(edge_rows_by_pair.get((source, target), ()))
    degree = sum(len(neighbours.get(node, ())) for node in set(host_ids))
    return {"hosts": node_set, "edges": edges, "degree": float(degree)}


def data_adjacency(graph: Any) -> dict[int, tuple[int, ...]]:
    adjacency: dict[int, set[int]] = {}
    for source, target in zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()):
        adjacency.setdefault(int(source), set()).add(int(target))
        adjacency.setdefault(int(target), set()).add(int(source))
    return {node: tuple(sorted(values)) for node, values in adjacency.items()}


def build_neighbourhood_frame(
    graphs: Mapping[int, Any],
    anchor_sets: Mapping[int, Mapping[str, Any]],
    *,
    strategy: str,
    model: str,
    explainer: str,
    seed: int,
    fanout: int = 10,
    hops: int = 2,
) -> pd.DataFrame:
    """S10 neighbourhood-change frame: 2-hop host/edge-set Jaccard per anchor pair."""
    rows: list[dict[str, Any]] = []
    graph_index: dict[int, dict[str, Any]] = {}
    for checkpoint_t, graph in graphs.items():
        graph_index[checkpoint_t] = {
            "neighbours": data_adjacency(graph),
            "edge_rows_by_pair": _edge_rows_by_pair(graph),
            "global_ids": graph.node_global_ids.tolist(),
            "lookup": {
                int(value): position
                for position, value in enumerate(graph.node_global_ids.tolist())
            },
        }
    for anchor_period, anchor_set in sorted(anchor_sets.items()):
        for anchor in anchor_set.get("anchors", []):
            host_ids = sorted(
                {int(anchor["host_src_global"]), int(anchor["host_dst_global"])}
            )
            base_hosts = {int(value) for value in anchor.get("sub_node_global_ids", host_ids)}
            base_edges = {int(value) for value in anchor.get("sub_edge_ids", [])}
            for checkpoint_t in sorted(graphs):
                if checkpoint_t < anchor_period:
                    continue
                graph = graphs[checkpoint_t]
                index = graph_index[checkpoint_t]
                # map global host ids to local node ids in the checkpoint graph
                lookup = index["lookup"]
                local_hosts = [lookup[value] for value in host_ids if value in lookup]
                if not local_hosts:
                    continue
                current = _two_hop_sets(
                    graph,
                    local_hosts,
                    fanout=fanout,
                    hops=hops,
                    neighbours=index["neighbours"],
                    edge_rows_by_pair=index["edge_rows_by_pair"],
                )
                current_hosts_global = {
                    int(graph.node_global_ids[local]) for local in current["hosts"]
                }
                host_change = 1.0 - md.jaccard_index(base_hosts, current_hosts_global)
                edge_change = 1.0 - md.jaccard_index(base_edges, current["edges"])
                rows.append(
                    {
                        "strategy": strategy,
                        "model": model,
                        "explainer": explainer,
                        "seed": int(seed),
                        "anchor_id": anchor["anchor_id"],
                        "class": str(anchor.get("label")),
                        "checkpoint_j": int(anchor_period),
                        "checkpoint_t": int(checkpoint_t),
                        "neighbourhood_change": float(host_change),
                        "degree": float(current["degree"]),
                        "edge_set_change": float(edge_change),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "strategy", "model", "explainer", "seed", "anchor_id", "class",
            "checkpoint_j", "checkpoint_t", "neighbourhood_change", "degree", "edge_set_change",
        ],
    )


# ---------------------------------------------------------------------------
# S5 tensors + retention + S12 EAR frames
# ---------------------------------------------------------------------------
def build_cl_tensors_from_s5(cl_matrix_path: str | Path) -> dict[tuple[str, str, int], mc.RMetricTensor]:
    """Load the engineer S5 writer output into the analysis metric tensor type."""
    tensor = mc.load_cl_matrix_json(cl_matrix_path)
    key = (str(tensor.strategy), str(tensor.model), int(tensor.seed) if tensor.seed is not None else -1)
    return {key: tensor}


def build_retention_frame(
    anchor_sets: Mapping[int, Mapping[str, Any]],
    predictions: Mapping[tuple[int, int], Mapping[Any, int]],
    *,
    strategy: str,
    model: str,
) -> pd.DataFrame:
    """Retained-fraction frame (strategy, model, class, checkpoint_t, retained_fraction)."""
    rows: list[dict[str, Any]] = []
    for anchor_period, anchor_set in sorted(anchor_sets.items()):
        anchors = anchor_set.get("anchors", [])
        if not anchors:
            continue
        for (checkpoint_t, anchor_period_key), predicted_by_anchor in sorted(predictions.items()):
            if checkpoint_t <= anchor_period or anchor_period_key != anchor_period:
                continue
            buckets: dict[str, list[int]] = {}
            for anchor in anchors:
                label = str(anchor.get("label"))
                retained = int(predicted_by_anchor.get(anchor["anchor_id"], -1) == int(anchor.get("label")))
                buckets.setdefault(label, []).append(retained)
            for label, values in sorted(buckets.items()):
                rows.append(
                    {
                        "strategy": strategy,
                        "model": model,
                        "class": label,
                        "checkpoint_t": int(checkpoint_t),
                        "retained_fraction": float(np.mean(values)),
                    }
                )
    return pd.DataFrame(
        rows, columns=["strategy", "model", "class", "checkpoint_t", "retained_fraction"]
    )


def build_ear_frames(records: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build S12 comparison/ablation frames from run_train EAR records."""
    comparison_rows = []
    ablation_rows = []
    for record in records:
        base = {
            "model": record.get("model"),
            "variant": record.get("variant"),
            "seed": int(record.get("seed", -1)),
            "drift_mean": float(record.get("drift_mean", np.nan)),
            "macro_f1": float(record.get("macro_f1", np.nan)),
            "forgetting": float(record.get("forgetting", np.nan)),
            "bwt": float(record.get("bwt", np.nan)),
            "runtime_s": float(record.get("runtime_s", np.nan)),
        }
        comparison_rows.append(base)
        if record.get("ablation"):
            ablation_rows.append(
                {
                    "model": base["model"],
                    "variant": base["variant"],
                    "buffer_size": int(record.get("buffer_size", 0)),
                    "beta": float(record.get("beta", np.nan)),
                    "attribution_source": record.get("attribution_source"),
                    "drift_mean": base["drift_mean"],
                    "macro_f1": base["macro_f1"],
                    "runtime_s": base["runtime_s"],
                }
            )
    return (
        pd.DataFrame(comparison_rows, columns=["model", "variant", "seed", "drift_mean", "macro_f1", "forgetting", "bwt", "runtime_s"]),
        pd.DataFrame(ablation_rows, columns=["model", "variant", "buffer_size", "beta", "attribution_source", "drift_mean", "macro_f1", "runtime_s"]),
    )


def _expected_cache_k(explainer: str, k: int) -> int:
    """GNNExplainer edge-mask entries use k=0; IG uses k=5/10."""
    return 0 if explainer in ("gnn_explainer", "gnnexplainer") else int(k)


def _group_filter_match(
    metadata: Mapping[str, Any],
    *,
    strategy: str | None,
    model: str | None,
    seed: int | None,
    source: Path,
) -> bool:
    """D-048/T49 group filter from entry JSON metadata (canonical index has no columns)."""
    for field, expected in (("strategy", strategy), ("model", model), ("seed", seed)):
        if expected is None:
            continue
        if field not in metadata:
            raise AnalysisWiringError(
                f"entry metadata in {source} lacks {field!r}; cannot filter the "
                "canonical cross-checkpoint index by group (expected D-048 fields)"
            )
        if str(metadata.get(field)) != str(expected):
            return False
    return True


def load_explanations_from_cache(
    cache_root: str | Path,
    *,
    explainer: str,
    k: int = 5,
    strategy: str | None = None,
    model: str | None = None,
    seed: int | None = None,
) -> tuple[dict[tuple[int, Any], np.ndarray], dict[Any, dict[str, Any]]]:
    """Load cached S7 explanations into ``{(checkpoint_period, anchor_id): array}``.

    T49 fixes: remote index paths are relocated against the assembled local
    ``results/`` root; entry JSON metadata supplies the D-048 group fields
    (strategy/model/seed) so a multi-group canonical index can be loaded for one
    group at a time.  ``k`` is normalised to 0 for GNNExplainer edge masks.
    """
    import csv
    import json

    root = Path(cache_root)
    # D-048: accept either the operational cache dir or the canonical committed
    # evidence CSV (results/text/tables/explain_index.csv).
    index_path = root if root.is_file() else root / "index.csv"
    if not index_path.is_file():
        raise AnalysisWiringError(f"S7 cache index not found: {index_path}")
    results_root = infer_results_root(index_path)
    expected_k = _expected_cache_k(explainer, k)
    explanations: dict[tuple[int, Any], np.ndarray] = {}
    anchor_meta: dict[Any, dict[str, Any]] = {}
    missing_paths = 0
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("explainer")) != explainer:
                continue
            try:
                row_k = int(row.get("k", -1))
            except (TypeError, ValueError):
                continue
            if row_k != expected_k:
                continue
            json_path = resolve_artifact_path(row["json_path"], index_path, results_root=results_root)
            npz_path = resolve_artifact_path(row["npz_path"], index_path, results_root=results_root)
            if not json_path.is_file() or not npz_path.is_file():
                missing_paths += 1
                continue
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            if not _group_filter_match(metadata, strategy=strategy, model=model, seed=seed, source=index_path):
                continue
            with np.load(npz_path) as archive:
                array = archive["attributions"] if "attributions" in archive.files else archive["edge_mask"]
            checkpoint_period = int(metadata.get("checkpoint_period", -1))
            anchor_id = metadata.get("anchor_id", row.get("anchor_id"))
            explanations[(checkpoint_period, anchor_id)] = np.asarray(array)
            anchor_meta[anchor_id] = {
                "anchor_period": int(metadata.get("anchor_period", checkpoint_period)),
                "class": str(metadata.get("class", metadata.get("label"))),
                "label": metadata.get("label"),
                "strategy": metadata.get("strategy"),
                "model": metadata.get("model"),
                "seed": metadata.get("seed"),
                "checkpoint_sha": row.get("checkpoint_sha"),
            }
    if not explanations:
        raise AnalysisWiringError(
            f"no cached {explainer!r} explanations with k={expected_k} found under {root}"
            + (f" ({missing_paths} row(s) missing after path relocation)" if missing_paths else "")
        )
    return explanations, anchor_meta


def load_noise_explanations_from_cache(
    cache_root: str | Path,
    *,
    explainer: str,
    k: int = 5,
    strategy: str | None = None,
    model: str | None = None,
    seed: int | None = None,
) -> dict[Any, list[dict[str, Any]]]:
    """Load noise-channel explanations grouped per anchor, preserving all reseeds/repeats.

    T49: the producer records ``checkpoint_period = anchor period`` for every null
    reseed / repeat; the old ``(checkpoint_period, anchor_id)`` key collapsed R=5
    reseeds and R=10 repeats to one value.  This loader keys by ``anchor_id`` and
    keeps every row, ordered by its recorded ``seed`` (and checkpoint sha), so the
    noise-frame builders receive the full R values.
    """
    import csv
    import json

    root = Path(cache_root)
    index_path = root if root.is_file() else root / "index.csv"
    if not index_path.is_file():
        raise AnalysisWiringError(f"noise cache index not found: {index_path}")
    results_root = infer_results_root(index_path)
    expected_k = _expected_cache_k(explainer, k)
    grouped: dict[Any, list[dict[str, Any]]] = {}
    missing_paths = 0
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("explainer")) != explainer:
                continue
            try:
                row_k = int(row.get("k", -1))
            except (TypeError, ValueError):
                continue
            if row_k != expected_k:
                continue
            json_path = resolve_artifact_path(row["json_path"], index_path, results_root=results_root)
            npz_path = resolve_artifact_path(row["npz_path"], index_path, results_root=results_root)
            if not json_path.is_file() or not npz_path.is_file():
                missing_paths += 1
                continue
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            if not _group_filter_match(metadata, strategy=strategy, model=model, seed=seed, source=index_path):
                continue
            with np.load(npz_path) as archive:
                array = archive["attributions"] if "attributions" in archive.files else archive["edge_mask"]
            run_seed = metadata.get("seed", metadata.get("null_seed", metadata.get("repeat_seed", row.get("seed"))))
            try:
                run_seed = int(run_seed)
            except (TypeError, ValueError):
                run_seed = -1
            anchor_id = metadata.get("anchor_id", row.get("anchor_id"))
            grouped.setdefault(anchor_id, []).append(
                {
                    "anchor_id": anchor_id,
                    "explainer": explainer,
                    "seed": run_seed,
                    "checkpoint_sha": row.get("checkpoint_sha") or metadata.get("checkpoint_sha"),
                    "checkpoint_period": int(metadata.get("checkpoint_period", -1)),
                    "anchor_period": int(metadata.get("anchor_period", -1)),
                    "class": str(metadata.get("class", metadata.get("label"))),
                    "label": metadata.get("label"),
                    "strategy": metadata.get("strategy"),
                    "model": metadata.get("model"),
                    "array": np.asarray(array),
                }
            )
    if not grouped:
        raise AnalysisWiringError(
            f"no cached {explainer!r} noise explanations with k={expected_k} found under {root}"
            + (f" ({missing_paths} row(s) missing after path relocation)" if missing_paths else "")
        )
    for anchor_id, records in grouped.items():
        records.sort(key=lambda item: (int(item["seed"]), str(item.get("checkpoint_sha"))))
    return grouped


__all__ = [
    "AnalysisWiringError",
    "build_cl_tensors_from_s5",
    "build_ear_frames",
    "build_explainer_repeat_noise_frame",
    "build_model_null_noise_frame",
    "build_neighbourhood_frame",
    "build_real_drift_frame",
    "build_retention_frame",
    "load_explanations_from_cache",
    "load_noise_explanations_from_cache",
]
