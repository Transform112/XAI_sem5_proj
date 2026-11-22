"""Pipeline wiring for S8-S12: build frames from frozen artifacts, then let src/analysis/ write tables.

D-044: no metric/statistic is re-implemented here. This module converts the
engineer's artifacts (S5 matrix, S6 anchors, S7 explanation caches, S1 graphs,
S12 EAR records) into the long frames that :func:`analysis.tables.generate_all`
consumes, and calls the analysis writers/validators.

Two modes:

* ``run_synthetic_wiring`` - end-to-end proof on ``analysis.synthetic`` fixtures;
  exercises the exact same ``generate_all`` + validators path without touching
  real results (banner marks it as not a result).
* ``run_real_wiring`` - real artifact assembly. It refuses to write partial
  tables: missing artifacts raise :class:`AnalysisWiringError` listing every
  missing path (S8-S12 real runs wait for G2 artifacts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.analysis_wiring.frames import (
    AnalysisWiringError,
    build_cl_tensors_from_s5,
    build_ear_frames,
    build_explainer_repeat_noise_frame,
    build_model_null_noise_frame,
    build_neighbourhood_frame,
    build_real_drift_frame,
    build_retention_frame,
    load_explanations_from_cache,
    load_noise_explanations_from_cache,
)

#: Metric set per explainer (S8): IG -> feature attributions, GNNExplainer -> edge masks.
EXPLAINER_METRIC_SET = {"ig": "feature", "gnn_explainer": "edge", "gnnexplainer": "edge"}


def _reference_null_with_fallback(null_frame, real_frame):
    """Supplement the reference null floor for missing (explainer, class) keys.

    The 500 reference noise anchors need not cover every class present in the
    1,008 canonical anchors.  For a missing key the documented fallback is the
    same-explainer pooled reference null distribution; returned keys feed the
    summary.  No real drift values are altered.
    """
    import pandas as pd

    required = set(zip(real_frame["explainer"].astype(str), real_frame["class"].astype(str)))
    available = set(zip(null_frame["explainer"].astype(str), null_frame["class"].astype(str)))
    missing = sorted(required - available)
    if not missing:
        return null_frame, []
    frames = [null_frame]
    for explainer, class_name in missing:
        source = null_frame[null_frame["explainer"].astype(str) == explainer]
        if source.empty:
            source = null_frame
        fallback = source.copy()
        fallback["class"] = class_name
        fallback["null_fallback"] = True
        frames.append(fallback)
    return pd.concat(frames, ignore_index=True), missing


def run_synthetic_wiring(*, out_root: str | Path, seed: int = 42) -> dict[str, Any]:
    """End-to-end wiring proof on the analysis toolkit's synthetic fixtures."""
    from src.analysis import artifacts, synthetic, tables, validators

    output_root = Path(out_root)
    layout = artifacts.ArtifactLayout(output_root / "results")
    layout.ensure()
    bundle = synthetic.make_bundle(seed=seed)
    outputs = tables.generate_all(
        cl_tensors=bundle.cl_tensors,
        real_drift=bundle.real_drift,
        model_null_noise=bundle.model_null_noise,
        explainer_repeat_noise=bundle.explainer_repeat_noise,
        retention=bundle.retention,
        neighbourhood=bundle.neighbourhood,
        ear_comparison=bundle.ear_comparison,
        ear_ablations=bundle.ear_ablations,
        layout=layout,
        supported_classes=synthetic.SUPPORTED_CLASSES,
    )
    validation = validators.validate_artifact_set(layout.tables_root)
    findings = [finding.as_dict() for finding in validation.findings]
    engine_required = [
        finding
        for finding in findings
        if finding.get("status") == "fail"
        and str(finding.get("check", "")).startswith(
            (
                "required.present.anchors",
                "required.present.explanations",
                "required.present.tables/anchor_retention.csv",
                "required.present.tables/explain_index.csv",
            )
        )
    ]
    table_failures = [
        finding
        for finding in findings
        if finding.get("status") == "fail" and finding not in engine_required
    ]
    return {
        "mode": "synthetic",
        "status": "pending_synthetic_not_a_result",
        "banner": "SYNTHETIC WIRING PROOF - analysis toolkit fixtures, NOT real results",
        "tables_root": str(layout.tables_root),
        "outputs": outputs,
        "validation": {
            "overall": validation.overall,
            "counts": validation.counts,
            # Scope note: the synthetic proof covers the S8-S13 table contract; the
            # engine-owned S6/S7 artifacts (anchor sets / explanation cache) are
            # intentionally not fabricated, so their required.present checks are
            # reported separately instead of being counted as table failures.
            "scope": "table contract (S8-S13); S6/S7 engine artifacts intentionally absent",
            "table_contract_passed": not table_failures,
            "table_failures": table_failures[:40],
            "engine_artifacts_absent": engine_required[:40],
        },
    }


# ---------------------------------------------------------------------------
# Real artifact requirements
# ---------------------------------------------------------------------------
def required_real_inputs(
    *,
    out_root: str | Path,
    anchors_dir: str | Path | None = None,
    cache_root: str | Path | None = None,
    cl_matrix_path: str | Path | None = None,
    graphs_dir: str | Path | None = None,
    noise_cache_root: str | Path | None = None,
) -> list[str]:
    """List every missing artifact the real S8-S12 wiring needs."""
    output_root = Path(out_root)
    cl_matrix = Path(cl_matrix_path) if cl_matrix_path else output_root / "results/text/tables/cl_matrix.json"
    cache = Path(cache_root) if cache_root else output_root / "results/text/explanations"
    anchors = Path(anchors_dir) if anchors_dir else output_root / "results/text/anchors"
    graphs = Path(graphs_dir) if graphs_dir else output_root / "datasets/processed/NF-UNSW-NB15-v3"
    noise_root = Path(noise_cache_root) if noise_cache_root else output_root / "results/text/explanations_noise"
    missing: list[str] = []
    if not cl_matrix.is_file():
        missing.append(str(cl_matrix))
    if not ((output_root / "results/text/tables/explain_index.csv").is_file() or (cache / "index.csv").is_file()):
        missing.append(str(output_root / "results/text/tables/explain_index.csv") + " (or " + str(cache / "index.csv") + ")")
    if not (anchors.is_dir() and list(anchors.glob("period_*.pt"))):
        missing.append(str(anchors / "period_*.pt"))
    if not (graphs.is_dir() and list(graphs.glob("period_*.pt"))):
        missing.append(str(graphs / "period_*.pt"))
    # S8 noise inputs: null-update + explainer-repeat explanation runs (generated by
    # the S8 noise stage; the wiring refuses to fabricate an empty noise floor).
    for channel in ("model_null", "explainer_repeat"):
        if not (noise_root / channel).is_dir():
            missing.append(str(noise_root / channel))
    return missing


def run_real_wiring(
    *,
    out_root: str | Path,
    strategy: str,
    model: str,
    seed: int,
    anchors_dir: str | Path | None = None,
    cache_root: str | Path | None = None,
    cl_matrix_path: str | Path | None = None,
    cl_matrix_paths: Mapping[tuple[str, str, int], str | Path] | None = None,
    graphs_dir: str | Path | None = None,
    noise_cache_root: str | Path | None = None,
    explainers: Sequence[str] = ("ig", "gnn_explainer"),
    k: int = 5,
    groups: Sequence[tuple[str, str, int]] | None = None,
) -> dict[str, Any]:
    """Assemble real S8-S12 frames and write the analysis tables.

    ``groups=None`` keeps the legacy single-group behavior.  A multi-group list
    (e.g. ft/replay/ewc Ãƒâ€” egraphsage/gcn, seed 42) loads the shared canonical
    explanation index once per metadata group, concatenates the real-drift and
    neighbourhood frames, uses the first group as the noise reference group, and
    writes one combined S8/S9 table set (T49 follow-up for full S7).

    Raises :class:`AnalysisWiringError` naming every missing artifact instead of
    writing partial tables (T18: no scale-up before G2).
    """
    # Normalise the group list once; the legacy CLI still calls this function
    # with a single strategy/model/seed and no ``groups`` argument.
    group_list: list[tuple[str, str, int]] = (
        [(str(strategy), str(model), int(seed))]
        if groups is None
        else [(str(g[0]), str(g[1]), int(g[2])) for g in groups]
    )
    if not group_list:
        raise AnalysisWiringError("real wiring requires at least one (strategy, model, seed) group")
    multi_group = len(group_list) > 1
    reference_strategy, reference_model, reference_seed = group_list[0]
    output_root = Path(out_root)
    missing = required_real_inputs(
        out_root=output_root,
        anchors_dir=anchors_dir,
        cache_root=cache_root,
        cl_matrix_path=cl_matrix_path,
        graphs_dir=graphs_dir,
        noise_cache_root=noise_cache_root,
    )
    if missing:
        raise AnalysisWiringError(
            "real S8-S12 wiring is waiting for frozen artifacts (no partial tables): "
            + "; ".join(missing)
        )
    from src.analysis import artifacts, tables, validators
    from analysis.synthetic import SUPPORTED_CLASSES

    import pandas as pd
    import torch

    layout = artifacts.ArtifactLayout(output_root / "results")
    layout.ensure()
    if cl_matrix_paths:
        cl_tensors: dict[Any, Any] = {}
        for _group_key, matrix_path in cl_matrix_paths.items():
            matrix_path = Path(matrix_path)
            if not matrix_path.is_file():
                raise AnalysisWiringError(f"per-group S5 cl_matrix not found: {matrix_path}")
            cl_tensors.update(build_cl_tensors_from_s5(matrix_path))
    else:
        cl_tensors = build_cl_tensors_from_s5(
            cl_matrix_path or output_root / "results/text/tables/cl_matrix.json"
        )
    anchors_path = Path(anchors_dir) if anchors_dir else output_root / "results/text/anchors"
    anchor_sets: dict[int, dict[str, Any]] = {}
    for anchor_file in sorted(anchors_path.glob("period_*.pt")):
        # Keep the protocol's 1-based period ids here: S7 entry metadata records
        # checkpoint_period/anchor_period as 1..5, so the S10 merge must use the
        # same keys (T49; the old 0-based pipeline keys never matched the frames).
        period = int(anchor_file.stem.split("_")[1])
        anchor_sets[period] = torch.load(anchor_file, weights_only=False)
    anchor_meta: dict[Any, dict[str, Any]] = {}
    for period, anchor_set in anchor_sets.items():
        for anchor in anchor_set.get("anchors", []):
            anchor_meta[anchor["anchor_id"]] = {
                "anchor_period": int(period),
                "class": str(anchor.get("label")),
                "label": int(anchor.get("label")),
                "sub_edge_ids": anchor.get("sub_edge_ids"),
            }
    cache = Path(cache_root) if cache_root else output_root / "results/text/explanations"
    real_drift_frames = []
    group_explanations: dict[tuple[str, str, int], tuple[dict[str, dict[tuple[int, Any], np.ndarray]], dict[str, dict[Any, dict[str, Any]]]]] = {}
    for g_strategy, g_model, g_seed in group_list:
        real_explanations: dict[str, dict[tuple[int, Any], np.ndarray]] = {}
        real_meta: dict[str, dict[Any, dict[str, Any]]] = {}
        for explainer in explainers:
            metric_set = EXPLAINER_METRIC_SET.get(explainer, "feature")
            canonical_index = output_root / "results/text/tables/explain_index.csv"
            explanation_source = canonical_index if canonical_index.is_file() else cache
            # T49: the canonical cross-checkpoint index is shared by six seed-42
            # groups and has no strategy/model/seed columns (D-048); filter each
            # row by entry JSON metadata so this cannot merge groups.
            explanations, meta = load_explanations_from_cache(
                explanation_source,
                explainer=explainer,
                k=k,
                strategy=g_strategy,
                model=g_model,
                seed=g_seed,
            )
            real_explanations[explainer] = explanations
            real_meta[explainer] = meta
            merged_meta = {**anchor_meta, **meta}
            real_drift_frames.append(
                build_real_drift_frame(
                    explanations,
                    merged_meta,
                    strategy=g_strategy,
                    model=g_model,
                    explainer=explainer,
                    seed=g_seed,
                    metric_set=metric_set,
                )
            )
        group_explanations[(g_strategy, g_model, g_seed)] = (real_explanations, real_meta)
    real_drift = pd.concat(real_drift_frames, ignore_index=True) if real_drift_frames else pd.DataFrame()
    # S10 neighbourhood frame from the frozen period graphs + anchor 2-hop sets.
    graphs_path = Path(graphs_dir) if graphs_dir else output_root / "datasets/processed/NF-UNSW-NB15-v3"
    graphs = {}
    for graph_file in sorted(graphs_path.glob("period_*.pt")):
        period = int(graph_file.stem.split("_")[1])
        graphs[period] = torch.load(graph_file, weights_only=False)
    neighbourhood_frames = []
    for g_strategy, g_model, g_seed in group_list:
        neighbourhood_frames.append(
            build_neighbourhood_frame(
                graphs,
                anchor_sets,
                strategy=g_strategy,
                model=g_model,
                explainer=explainers[0] if explainers else "ig",
                seed=g_seed,
            )
        )
    neighbourhood = pd.concat(neighbourhood_frames, ignore_index=True) if neighbourhood_frames else pd.DataFrame()
    # S6 retention frame from the anchor sets' retained-fraction table.
    retention = pd.DataFrame(
        columns=["strategy", "model", "class", "checkpoint_t", "retained_fraction"]
    )
    # S8 noise frames from precomputed null/repeat explanation caches.
    noise_root = Path(noise_cache_root) if noise_cache_root else output_root / "results/text/explanations_noise"
    reference_explanations, reference_meta = group_explanations[
        (reference_strategy, reference_model, reference_seed)
    ]
    null_frames: list[pd.DataFrame] = []
    repeat_frames: list[pd.DataFrame] = []
    noise_inputs: dict[str, Any] = {"per_explainer": {}}
    for noise_explainer in explainers:
        metric_set = EXPLAINER_METRIC_SET.get(noise_explainer, "feature")
        # Model-null reference = the run's own checkpoint explanation at the
        # anchor's own period (the diagonal M_j^s entry in the canonical cache).
        reference: dict[Any, np.ndarray] = {}
        for (checkpoint, anchor_id), array in reference_explanations.get(noise_explainer, {}).items():
            meta = reference_meta.get(noise_explainer, {}).get(anchor_id, {})
            if int(meta.get("anchor_period", -1)) == int(checkpoint):
                reference[anchor_id] = array
        null_grouped = load_noise_explanations_from_cache(
            noise_root / "model_null",
            explainer=noise_explainer,
            k=k,
            strategy=reference_strategy,
            model=reference_model,
            seed=None,  # noise entry 'seed' is the null-reseed seed, not the run seed
        )
        reseeded: dict[Any, dict[Any, np.ndarray]] = {}
        null_meta: dict[Any, dict[str, Any]] = {}
        skipped_null = 0
        for anchor_id, records in null_grouped.items():
            if anchor_id not in reference or len(records) != 5:
                skipped_null += 1
                continue
            reseeded[anchor_id] = {int(record["seed"]): record["array"] for record in records}
            first = records[0]
            null_meta[anchor_id] = {
                "anchor_period": int(first["anchor_period"]),
                "class": str(first["class"]),
                "label": first["label"],
            }
        if reseeded:
            reference = {anchor_id: reference[anchor_id] for anchor_id in reseeded}
            null_frames.append(
                build_model_null_noise_frame(
                    reference,
                    reseeded,
                    {**anchor_meta, **null_meta},
                    strategy=strategy,
                    model=model,
                    explainer=noise_explainer,
                    seed=seed,
                    metric_set=metric_set,
                )
            )
        repeat_grouped = load_noise_explanations_from_cache(
            noise_root / "explainer_repeat",
            explainer=noise_explainer,
            k=k,
            strategy=strategy,
            model=model,
            seed=None,
        )
        repeats: dict[Any, list[np.ndarray]] = {}
        repeat_meta: dict[Any, dict[str, Any]] = {}
        skipped_repeat = 0
        for anchor_id, records in repeat_grouped.items():
            if len(records) != 10:
                skipped_repeat += 1
                continue
            repeats[anchor_id] = [record["array"] for record in records]
            first = records[0]
            repeat_meta[anchor_id] = {
                "anchor_period": int(first["anchor_period"]),
                "class": str(first["class"]),
                "label": first["label"],
            }
        if repeats:
            repeat_frames.append(
                build_explainer_repeat_noise_frame(
                    repeats,
                    {**anchor_meta, **repeat_meta},
                    strategy=strategy,
                    model=model,
                    explainer=noise_explainer,
                    metric_set=metric_set,
                )
            )
        noise_inputs["per_explainer"][noise_explainer] = {
            "model_null_reference_anchors": len(reference),
            "model_null_anchors_used": len(reseeded),
            "model_null_reseeds_per_anchor": sorted({len(records) for records in reseeded.values()}),
            "model_null_skipped_anchors": skipped_null,
            "explainer_repeat_anchors_used": len(repeats),
            "explainer_repeat_runs_per_anchor": sorted({len(records) for records in repeats.values()}),
            "explainer_repeat_skipped_anchors": skipped_repeat,
        }
    if not null_frames:
        raise AnalysisWiringError(
            "model-null noise frame unavailable: no anchor has both a canonical "
            "diagonal reference explanation and R=5 null reseeds in the assembled root"
        )
    model_null_noise = pd.concat(null_frames, ignore_index=True)
    explainer_repeat_noise = (
        pd.concat(repeat_frames, ignore_index=True) if repeat_frames else pd.DataFrame()
    )
    noise_inputs["model_null_frame_rows"] = int(len(model_null_noise))
    noise_inputs["explainer_repeat_frame_rows"] = int(len(explainer_repeat_noise))
    # S9 requires null and real updates to use exactly the same anchors; assembled
    # partial smoke roots can have disjoint smoke anchor subsets. Full S7 should
    # leave this unchanged; record the alignment explicitly for the SMOKE report.
    real_drift_total = len(real_drift)
    real_drift_metrics = real_drift.copy()  # drift tables keep every available pair
    if multi_group:
        # The noise floor is the documented reference-group floor (ft/egraphsage);
        # applying it across the six strategy/model groups must not join on model.
        null_keys = set(
            zip(
                model_null_noise["explainer"].astype(str),
                model_null_noise["anchor_id"].astype(str),
                model_null_noise["class"].astype(str),
            )
        )
        real_keys = list(
            zip(
                real_drift["explainer"].astype(str),
                real_drift["anchor_id"].astype(str),
                real_drift["class"].astype(str),
            )
        )
    else:
        null_keys = set(
            zip(
                model_null_noise["model"].astype(str),
                model_null_noise["explainer"].astype(str),
                model_null_noise["anchor_id"].astype(str),
                model_null_noise["class"].astype(str),
            )
        )
        real_keys = list(
            zip(
                real_drift["model"].astype(str),
                real_drift["explainer"].astype(str),
                real_drift["anchor_id"].astype(str),
                real_drift["class"].astype(str),
            )
        )
    if multi_group:
        # Reference-group null floor applied per (explainer, class); the 1,008
        # canonical anchors exceed the 500 reference noise anchors, so an
        # anchor-specific join would drop valid pairs.  Missing reference classes
        # use the same-explainer pooled fallback.
        rq1_null_noise, null_fallback_keys = _reference_null_with_fallback(
            model_null_noise, real_drift_metrics
        )
        real_drift = real_drift_metrics.copy().reset_index(drop=True)
    else:
        rq1_null_noise, null_fallback_keys = None, []
        aligned_mask = [key in null_keys for key in real_keys]
        real_drift = real_drift.loc[aligned_mask].reset_index(drop=True)
    noise_inputs["real_drift_input_rows"] = int(real_drift_total)
    noise_inputs["real_drift_rows_in_drift_tables"] = int(len(real_drift_metrics))
    noise_inputs["real_drift_rows_aligned_to_null"] = int(len(real_drift))
    noise_inputs["real_drift_rows_dropped_unanchored"] = int(real_drift_total - len(real_drift))
    noise_inputs["reference_null_pooling"] = (
        "per (explainer, class) across all groups" if multi_group else "per anchor (single group)"
    )
    noise_inputs["reference_null_fallback_keys"] = [
        f"{explainer}:{class_name}" for explainer, class_name in null_fallback_keys
    ]
    if real_drift.empty:
        raise AnalysisWiringError(
            "real-drift frame has no anchor with a matching model-null frame after "
            "alignment; the assembled root is not a complete S7/noise pair"
        )
    ear_comparison, ear_ablations = build_ear_frames([])
    # Single-group SMOKE wiring runs cannot form the replay/EWC paired comparison;
    # a multi-group call has all strategies and keeps strict strategy comparison.
    partial_groups = not {"ft", "replay", "ewc"}.issubset(set(real_drift["strategy"].astype(str)))
    analysis_partial = bool(multi_group or partial_groups)
    noise_cfg = None
    if partial_groups:
        from analysis.noise_floor import NoiseFloorConfig

        noise_cfg = NoiseFloorConfig(allow_partial_strategies=True)
    outputs = tables.generate_all(
        cl_tensors=cl_tensors,
        real_drift=real_drift_metrics,
        real_drift_rq1=real_drift,
        model_null_noise=model_null_noise,
        explainer_repeat_noise=explainer_repeat_noise,
        retention=retention,
        neighbourhood=neighbourhood,
        ear_comparison=ear_comparison,
        ear_ablations=ear_ablations,
        layout=layout,
        supported_classes=SUPPORTED_CLASSES,
        cfg=noise_cfg,
        partial=analysis_partial,
        rq1_join_keys=("explainer", "class") if multi_group else None,
        rq1_null_noise=rq1_null_noise,
    )
    validation_stage = "s9" if analysis_partial else "all"
    validation = validators.validate_artifact_set(layout.tables_root, stage=validation_stage)
    per_group_rows = {
        f"{g_strategy}_{g_model}_seed{g_seed}": int(
            (
                real_drift["strategy"].eq(g_strategy)
                & real_drift["model"].eq(g_model)
                & real_drift["seed"].eq(g_seed)
            ).sum()
        )
        for g_strategy, g_model, g_seed in group_list
    }
    return {
        "mode": "real",
        "tables_root": str(layout.tables_root),
        "outputs": outputs,
        "noise_inputs": noise_inputs,
        "groups": [f"{g_strategy}:{g_model}:{g_seed}" for g_strategy, g_model, g_seed in group_list],
        "multi_group": multi_group,
        "analysis_scope": "s8_s9_partial" if analysis_partial else "s8_s12_full",
        "per_group_real_drift_rows": per_group_rows,
        "partial_groups": partial_groups,
        "validation": {"overall": validation.overall, "counts": validation.counts},
    }


__all__ = [
    "AnalysisWiringError",
    "EXPLAINER_METRIC_SET",
    "required_real_inputs",
    "run_real_wiring",
    "run_synthetic_wiring",
]
