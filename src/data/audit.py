"""S-D1 data-audit orchestrator: one call -> scorecard + frozen artifacts.

Pipeline (ordered, each step timed and recorded):

    load raw CSV -> clean (dedupe/sentinel/sort/global maps) -> 5 chronological
    periods -> leakage-safe grouped splits (+ host-pair-disjoint robustness
    variant) -> preprocess fitted on period-1 train only -> structure checks ->
    quality audit -> threshold evaluation -> artifacts + run manifest.

Artifacts (under ``out_dir``):
    scorecard.json / scorecard.md     verdict, checks, metric summaries
    feature_manifest.csv              55-column kept/dropped/transformed table
    split_manifest.json               frozen mask hashes (grouped + host-pair)
    structure.json                    D-020/D-023 structure facts
    quality_audit.json                full quality metrics
    preprocessor_state.json           train-only fitted state
    manifest.json                     run manifest (git/config/dataset hashes)

Status vocabulary (D-023): ``pass`` | ``flag`` | ``fail``.
``fail`` blocks; ``flag`` needs an explanation and is resolved at G1c.

Thresholds are **required** from the configuration file (``configs/audit.yaml``);
missing threshold keys raise instead of silently using a hard-coded value.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.clean import clean_frame, load_raw_frame
from src.data.features import build_feature_manifest
from src.data.periods import DEFAULT_SESSION_MIN_GAP_SECONDS, assign_periods
from src.data.preprocess import Preprocessor
from src.data.quality import quality_audit
from src.data.registry import DatasetSpec
from src.data.splits import build_splits, freeze_split_masks, write_split_masks
from src.data.structure import structure_report
from src.utils.config import Config, hash_config
from src.utils.logging import get_logger
from src.utils.manifest import RunManifest, sha256_file, utc_now, write_json
from src.utils.seed import seed_everything

LOGGER = get_logger("xdrift.audit")

#: Threshold keys that MUST be present in the configuration file.
REQUIRED_THRESHOLDS: tuple[str, ...] = (
    "nan_inf_fraction_max",
    "near_duplicate_cross_split_max",
    "duplicate_fraction_max",
    "label_conflict_fraction_max",
    "constant_feature_count_max",
    "adversarial_auc_within_period_max",
    "median_host_degree_min",
    "giant_component_fraction_min",
    "host_recurrence_min",
    "repeated_host_pair_fraction_min",
    "max_self_loop_fraction_max",
    "graph_lift_macro_f1_min",
    "graph_lift_vs_randomised_min",
)

#: D-023 default: hard giant-component check applies only when a period has at
#: least this many hosts; below it the check is emitted as ``flag``. The value
#: is read from configs/audit.yaml (structure.small_host_flag_threshold).
DEFAULT_SMALL_HOST_FLAG_THRESHOLD = 100

#: Section fallbacks for non-threshold settings. The canonical values live in
#: configs/audit.yaml; these only make library use in tests convenient.
SECTION_DEFAULTS: dict[str, dict[str, Any]] = {
    "periods": {
        "count": 5,
        "strategy": "equal_counts",
        "min_flows_per_period": 1,
        "min_session_gap_seconds": 3600,
    },
    "dedupe": {"exact": True, "keep": "first"},
    "near_duplicate": {
        "time_bucket_ms": 1000,
        "keys": ["host_src", "host_dst", "port_src", "port_dst", "protocol"],
    },
    "sentinels": {"negative_to_nan": True, "inf_to_nan": True, "explicit_values": []},
    "splits": {
        "fractions": {"train": 0.7, "val": 0.1, "test": 0.2},
        "seed": 42,
        "group_by": "near_duplicate",
        "host_pair_disjoint": {"enabled": True, "min_groups_per_split": 10, "min_share": 0.05},
        "frozen": True,
    },
    "preprocess": {
        "categorical_encoding": "label",
        "unknown_category_index": 0,
        "impute_strategy": "median",
        "missing_indicator_threshold": 0.001,
        "log1p_clip_min": 0.0,
        "standardize": True,
        "clip_std": 10.0,
    },
    "structure": {
        "giant_component_undirected": True,
        "degree_mode": "undirected_pair",
        "repeated_host_pairs": True,
        "self_loops": True,
    },
    "quality": {},
    "graph_lift": {"enabled": False},
    "identity": {},
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(f"audit config section {name!r} must be a mapping")
    resolved = dict(SECTION_DEFAULTS.get(name, {}))
    resolved.update(dict(value))
    return resolved


def _thresholds(config: Mapping[str, Any]) -> dict[str, float]:
    raw = config.get("thresholds")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "audit config is missing the 'thresholds' mapping; thresholds are never hard-coded "
            "(see configs/audit.yaml)"
        )
    missing = [key for key in REQUIRED_THRESHOLDS if key not in raw]
    if missing:
        raise ValueError(f"audit config is missing required thresholds: {missing}")
    return {key: float(raw[key]) for key in REQUIRED_THRESHOLDS}


# ---------------------------------------------------------------------------
# Check construction
# ---------------------------------------------------------------------------
def _check(
    name: str,
    section: str,
    status: str,
    *,
    measured: Any,
    threshold: Any = None,
    detail: str = "",
) -> dict[str, Any]:
    if status not in {"pass", "flag", "fail"}:
        raise ValueError(f"invalid check status {status!r}")
    return {
        "name": name,
        "section": section,
        "status": status,
        "measured": measured,
        "threshold": threshold,
        "detail": detail,
    }


def _pass_fail(passed: bool) -> str:
    return "pass" if passed else "fail"


def _giant_component_check(
    thresholds: Mapping[str, float],
    structure: Mapping[str, Any],
    *,
    small_host_flag_threshold: int = DEFAULT_SMALL_HOST_FLAG_THRESHOLD,
) -> dict[str, Any]:
    """D-023: per-period giant component; hard fail only at/above the host threshold."""
    per_period = structure.get("periods", {}) or {}
    evaluations = []
    worst = "pass"
    for period_label, block in per_period.items():
        hosts = int(block.get("hosts", 0))
        fraction = float(block.get("giant_fraction", 0.0))
        if hosts >= small_host_flag_threshold:
            status = "pass" if fraction >= thresholds["giant_component_fraction_min"] else "fail"
        else:
            status = "pass" if fraction >= thresholds["giant_component_fraction_min"] else "flag"
        if status == "fail":
            worst = "fail"
        elif status == "flag" and worst == "pass":
            worst = "flag"
        evaluations.append(
            {
                "period": int(period_label) + 1,
                "hosts": hosts,
                "giant_fraction": round(fraction, 6),
                "status": status,
            }
        )
    union_fraction = float(structure.get("global", {}).get("components", {}).get("giant_fraction", 0.0))
    detail = (
        "per-period graphs (D-023); host_count < "
        f"{small_host_flag_threshold} emits flag not fail; union-across-time "
        f"giant_fraction={union_fraction:.4f} reported as diagnostic"
    )
    return _check(
        "giant_component_fraction",
        "structure",
        worst,
        measured={
            "per_period": evaluations,
            "min_period_fraction": structure.get("giant_component_per_period", {}).get("min"),
            "union_fraction": round(union_fraction, 6),
        },
        threshold=thresholds["giant_component_fraction_min"],
        detail=detail,
    )


def evaluate_checks(
    *,
    thresholds: Mapping[str, float],
    quality: Mapping[str, Any],
    structure: Mapping[str, Any],
    splits_grouped: Any,
    splits_host_pair: Any | None,
    periods: Mapping[str, Any],
    pilot: Mapping[str, Any] | None,
    structure_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Translate audit measurements into pass/flag/fail checks."""
    structure_cfg = dict(structure_config or {})
    small_host_flag_threshold = int(
        structure_cfg.get("small_host_flag_threshold", DEFAULT_SMALL_HOST_FLAG_THRESHOLD)
    )
    checks: list[dict[str, Any]] = []

    nan_inf = quality["nan_inf"]
    checks.append(
        _check(
            "nan_inf_fraction",
            "quality",
            _pass_fail(nan_inf["fraction"] <= thresholds["nan_inf_fraction_max"]),
            measured=nan_inf["fraction"],
            threshold=thresholds["nan_inf_fraction_max"],
            detail=f"{nan_inf['total']} non-finite cells in {nan_inf['scope']}",
        )
    )
    duplicates = quality["duplicates"]
    checks.append(
        _check(
            "duplicate_fraction_remaining",
            "quality",
            _pass_fail(
                duplicates["duplicate_fraction_remaining"] <= thresholds["duplicate_fraction_max"]
            ),
            measured=duplicates["duplicate_fraction_remaining"],
            threshold=thresholds["duplicate_fraction_max"],
            detail=(
                "D-023: threshold applies to duplicates remaining after dedupe; "
                f"raw diagnostic rate={duplicates['duplicate_fraction_raw']:.6f}"
            ),
        )
    )
    conflict = quality["label_conflict"]
    checks.append(
        _check(
            "label_conflict_fraction",
            "quality",
            _pass_fail(conflict["fraction"] <= thresholds["label_conflict_fraction_max"]),
            measured=conflict["fraction"],
            threshold=thresholds["label_conflict_fraction_max"],
            detail=f"{conflict['n_conflicted_groups']}/{conflict['n_groups']} near-duplicate groups carry conflicting labels",
        )
    )
    constant = quality["constant_features"]
    checks.append(
        _check(
            "constant_feature_count",
            "quality",
            _pass_fail(constant["count"] <= thresholds["constant_feature_count_max"]),
            measured=constant["count"],
            threshold=thresholds["constant_feature_count_max"],
            detail=f"constant features: {constant['features']}",
        )
    )
    adversarial = quality["adversarial"]["within_period"]
    if adversarial.get("computed"):
        checks.append(
            _check(
                "adversarial_auc_within_period",
                "quality",
                _pass_fail(adversarial["auc"] <= thresholds["adversarial_auc_within_period_max"]),
                measured=adversarial["auc"],
                threshold=thresholds["adversarial_auc_within_period_max"],
                detail=f"{adversarial['folds']}-fold mean AUC; train vs test edges of period 1",
            )
        )
    else:
        checks.append(
            _check(
                "adversarial_auc_within_period",
                "quality",
                "flag",
                measured=None,
                threshold=thresholds["adversarial_auc_within_period_max"],
                detail=f"not computed: {adversarial.get('reason')}",
            )
        )
    crossing = int(splits_grouped.stats.get("groups_crossing_splits", 0))
    checks.append(
        _check(
            "near_duplicate_cross_split",
            "splits",
            _pass_fail(crossing <= thresholds["near_duplicate_cross_split_max"]),
            measured=crossing,
            threshold=thresholds["near_duplicate_cross_split_max"],
            detail="groups that appear in more than one split (must be 0)",
        )
    )
    checks.append(
        _check(
            "periods_time_disjoint",
            "periods",
            _pass_fail(bool(periods.get("time_disjoint", False))),
            measured=bool(periods.get("time_disjoint", False)),
            threshold=True,
            detail="later period must start at or after the previous period ends",
        )
    )

    per_period = structure.get("periods", {}) or {}
    median_degrees = [float(block["degree"]["median"]) for block in per_period.values()]
    min_median_degree = min(median_degrees) if median_degrees else 0.0
    checks.append(
        _check(
            "median_host_degree",
            "structure",
            _pass_fail(min_median_degree >= thresholds["median_host_degree_min"]),
            measured=min_median_degree,
            threshold=thresholds["median_host_degree_min"],
            detail="minimum across periods of the per-period median incident-flow degree",
        )
    )
    checks.append(
        _giant_component_check(
            thresholds, structure, small_host_flag_threshold=small_host_flag_threshold
        )
    )

    recurrences = [
        float(block["host_recurrence"])
        for block in per_period.values()
        if block.get("host_recurrence") is not None
    ]
    if recurrences:
        checks.append(
            _check(
                "host_recurrence",
                "structure",
                _pass_fail(min(recurrences) >= thresholds["host_recurrence_min"]),
                measured=min(recurrences),
                threshold=thresholds["host_recurrence_min"],
                detail="minimum share of period-t hosts also present in period t-1 (D-020: min 0.571 expected)",
            )
        )
    else:
        checks.append(
            _check("host_recurrence", "structure", "flag", measured=None, threshold=thresholds["host_recurrence_min"], detail="single period: recurrence undefined")
        )

    repeated = [
        float(block.get("repeated_pair_fraction", 0.0)) for block in per_period.values()
    ]
    min_repeated = min(repeated) if repeated else 0.0
    checks.append(
        _check(
            "repeated_host_pair_fraction",
            "structure",
            _pass_fail(min_repeated >= thresholds["repeated_host_pair_fraction_min"]),
            measured=min_repeated,
            threshold=thresholds["repeated_host_pair_fraction_min"],
            detail="share of edges whose directed host pair carries >= 2 flows (min across periods)",
        )
    )
    self_loops = float(structure.get("global", {}).get("self_loops", {}).get("fraction", 0.0))
    checks.append(
        _check(
            "self_loop_fraction",
            "structure",
            _pass_fail(self_loops <= thresholds["max_self_loop_fraction_max"]),
            measured=self_loops,
            threshold=thresholds["max_self_loop_fraction_max"],
            detail="self-loops are kept as normal edges; only an unusual rate is flagged",
        )
    )
    if splits_host_pair is not None:
        feasibility = splits_host_pair.stats.get("feasible")
        checks.append(
            _check(
                "host_pair_disjoint_feasibility",
                "splits",
                "pass" if feasibility else "flag",
                measured=bool(feasibility),
                threshold=True,
                detail="; ".join(splits_host_pair.stats.get("infeasible_reasons", []))[:500]
                or "robustness variant is feasible",
            )
        )
    if pilot is not None:
        verdict = pilot.get("verdict") or {}
        for check_name, verdict_key, threshold_key in (
            ("graph_lift_macro_f1", "graph_lift_macro_f1", "graph_lift_macro_f1_min"),
            ("graph_lift_vs_randomised", "graph_lift_vs_randomised", "graph_lift_vs_randomised_min"),
        ):
            block = verdict.get(verdict_key) or {}
            measured = block.get("mean_delta_points")
            if block.get("pass") is None:
                status = "flag"
                detail = "pilot ran but produced no D-024 verdict block"
            else:
                status = "pass" if bool(block["pass"]) else "fail"
                detail = (
                    "D-024: 3-seed mean delta "
                    f"{measured:+.2f} macro-F1 points, 95% CI "
                    f"{block.get('ci95_points')}, threshold >= {block.get('threshold_points')} points; "
                    "E-GraphSAGE minus MLP" if verdict_key == "graph_lift_macro_f1" else
                    "D-024: 3-seed mean delta "
                    f"{measured:+.2f} macro-F1 points, 95% CI "
                    f"{block.get('ci95_points')}, threshold >= {block.get('threshold_points')} points; "
                    "E-GraphSAGE minus structure-randomised control"
                )
            checks.append(
                _check(
                    check_name,
                    "graph_lift",
                    status,
                    measured=measured,
                    threshold=thresholds[threshold_key],
                    detail=detail,
                )
            )
    else:
        for check_name, threshold_key in (
            ("graph_lift_macro_f1", "graph_lift_macro_f1_min"),
            ("graph_lift_vs_randomised", "graph_lift_vs_randomised_min"),
        ):
            checks.append(
                _check(
                    check_name,
                    "graph_lift",
                    "flag",
                    measured=None,
                    threshold=thresholds[threshold_key],
                    detail=(
                        "graph-lift pilot not run in this invocation (decisive G1c item; run "
                        "--with-pilot on the legacy/canonical lane)"
                    ),
                )
            )
    return checks


def _overall_status(checks: Sequence[Mapping[str, Any]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "fail"
    if any(check["status"] == "flag" for check in checks):
        return "flag"
    return "pass"


#: D-039 gate semantics: which checks belong to which gate, so the scorecard can
#: separate data-quality evidence (G1b) from the structure-informativeness
#: decision input (G1c). The numeric checks themselves are unchanged.
GATE_SECTIONS: dict[str, dict[str, Any]] = {
    "g1b_data_integrity": {
        "checks": [
            "nan_inf_fraction",
            "duplicate_fraction_remaining",
            "label_conflict_fraction",
            "constant_feature_count",
            "adversarial_auc_within_period",
            "near_duplicate_cross_split",
            "periods_time_disjoint",
        ],
        "note": "G1b: data integrity / leakage / split validity (D-039)",
    },
    "g1c_graph_lift_decision": {
        "checks": ["graph_lift_macro_f1", "graph_lift_vs_randomised"],
        "note": "G1c decision input: structure-informativeness verdict, pre-registered "
        "D-024/R1b; not a data-quality failure (D-039)",
    },
    "structure_reporting": {
        "checks": [
            "median_host_degree",
            "giant_component_fraction",
            "host_recurrence",
            "repeated_host_pair_fraction",
            "self_loop_fraction",
            "host_pair_disjoint_feasibility",
        ],
        "note": "G1b structure reporting per D-020/D-023",
    },
}


def _decision_inputs(checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Machine-readable separation of gate evidence (D-039)."""
    by_name = {check["name"]: check for check in checks}
    result: dict[str, Any] = {}
    for gate, spec in GATE_SECTIONS.items():
        entries = [by_name[name] for name in spec["checks"] if name in by_name]
        result[gate] = {
            "checks": [entry["name"] for entry in entries],
            "status": _overall_status(entries) if entries else "missing",
            "note": spec["note"],
        }
    result["data_state"] = {
        "note": "data state is frozen (D-034/D-036); the graph-lift verdict must not "
        "trigger data/threshold changes (D-039)"
    }
    return result


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------
def _feature_manifest_table(
    spec: DatasetSpec,
    preprocessor: Preprocessor,
    *,
    final_feature_names: Sequence[str],
) -> pd.DataFrame:
    manifest = build_feature_manifest(spec)
    state = preprocessor.state_dict()
    rows: list[dict[str, Any]] = []
    final_index = {name: index for index, name in enumerate(final_feature_names)}
    columns_state = state.get("columns", {})
    categorical_state = state.get("categorical", {})
    for _, row in manifest.iterrows():
        record = row.to_dict()
        name = str(record["name"])
        column_state = columns_state.get(name, {})
        record["in_final_matrix"] = name in final_index
        record["final_index"] = final_index.get(name)
        record["train_missing_rate"] = column_state.get("missing_rate")
        record["impute_value"] = column_state.get("impute_value")
        record["scaler_mean"] = column_state.get("mean")
        record["scaler_std"] = column_state.get("std")
        record["vocab_size"] = (
            categorical_state.get(name, {}).get("n_categories")
            if name in categorical_state
            else None
        )
        record["missing_indicator"] = column_state.get("missing_indicator")
        rows.append(record)
    table = pd.DataFrame(rows)
    indicator_rows = []
    for base, indicator in state.get("missing_indicators", {}).items():
        indicator_rows.append(
            {
                "name": indicator,
                "role": "feature",
                "source_group": "derived_indicator",
                "transform": "binary_indicator",
                "graph_role": "edge_feature",
                "reason": f"train-fitted missing indicator for {base}",
                "declared_in": "preprocess.missing_indicator_threshold",
                "in_final_matrix": True,
                "final_index": final_index.get(indicator),
            }
        )
    if indicator_rows:
        table = pd.concat([table, pd.DataFrame(indicator_rows)], ignore_index=True)
    return table


def _pilot_summary(pilot: Mapping[str, Any]) -> dict[str, Any]:
    """Compact pilot report for embedding in the scorecard (drops bootstrap arrays)."""
    summary = {
        key: pilot[key]
        for key in (
            "status",
            "reason",
            "dataset",
            "period",
            "device",
            "seeds",
            "sample",
            "mean_over_seeds",
            "verdict",
            "verdict_thresholds",
            "timing",
            "torch",
            "generated_at_utc",
            "randomised_structure_semantics",
            "ci_method",
            "pinning",
        )
        if key in pilot
    }
    summary["per_seed"] = [
        {key: value for key, value in entry.items() if not key.startswith("bootstrap")}
        for entry in pilot.get("per_seed", [])
    ]
    return summary


def _scorecard_markdown(scorecard: Mapping[str, Any]) -> str:
    status = str(scorecard["status"]).upper()
    lines = [
        f"# Data audit scorecard: {scorecard['dataset']}",
        "",
        f"- **Verdict**: **{status}**",
        f"- **Generated (UTC)**: {scorecard['generated_at_utc']}",
        f"- **Config SHA-256**: `{scorecard['config']['sha256']}`",
        f"- **Rows**: {scorecard['rows']['after_clean']:,} after dedupe "
        f"(raw {scorecard['rows']['raw']:,}, published {scorecard['rows']['published']}, "
        f"removed {scorecard['rows']['exact_duplicates_removed']:,} exact duplicates)",
        "",
        "## Checks",
        "",
        "| Check | Section | Status | Measured | Threshold |",
        "| --- | --- | --- | --- | --- |",
    ]
    for check in scorecard["checks"]:
        measured = check["measured"]
        if isinstance(measured, (dict, list)):
            measured_text = "(see scorecard.json)"
        elif isinstance(measured, float):
            measured_text = f"{measured:.6g}"
        else:
            measured_text = str(measured)
        lines.append(
            f"| {check['name']} | {check['section']} | **{check['status'].upper()}** | "
            f"{measured_text} | {check['threshold']} |"
        )
    lines += [
        "",
        "## Periods",
        "",
        "| Period | Rows | Time range (UTC) | Classes present | Benign only |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for label, block in scorecard["periods"]["per_period"].items():
        lines.append(
            f"| {label} | {block['rows']:,} | {block.get('ts_min_utc')} -> {block.get('ts_max_utc')} | "
            f"{block.get('classes_present') or '-'} | "
            f"{'yes' if int(label) - 1 in scorecard.get('benign_only_periods', []) else 'no'} |"
        )
    lines += [
        "",
        "## Structure (D-020 / D-023)",
        "",
        "| Period | Hosts | Edges | Median degree | Giant fraction | Components | Recurrence | Turnover | Repeated-pair share |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for label, block in scorecard["structure"]["periods"].items():
        recurrence = block.get("host_recurrence")
        turnover = block.get("host_turnover")
        lines.append(
            f"| {int(label) + 1} | {block['n_hosts']} | {block['n_edges']:,} | "
            f"{block['degree']['median']:.1f} | {block['components']['giant_fraction']:.4f} | "
            f"{block['components']['sizes']} | "
            f"{'-' if recurrence is None else f'{recurrence:.3f}'} | "
            f"{'-' if turnover is None else f'{turnover:.3f}'} | "
            f"{block.get('repeated_pair_fraction', 0.0):.4f} |"
        )
    lines += [
        "",
        f"Union-across-time diagnostic: {scorecard['structure']['global']['n_hosts']} hosts, "
        f"components {scorecard['structure']['global']['components']['sizes']}, "
        f"giant fraction {scorecard['structure']['global']['components']['giant_fraction']:.4f}.",
        "",
        "## Quality",
        "",
        f"- NaN/inf after preprocessing: {scorecard['quality']['nan_inf']['fraction']:.6g} "
        f"({scorecard['quality']['nan_inf']['total']} cells)",
        f"- Duplicate fraction raw (diagnostic): {scorecard['quality']['duplicates']['duplicate_fraction_raw']:.6f}; "
        f"remaining after dedupe: {scorecard['quality']['duplicates']['duplicate_fraction_remaining']:.6g}",
        f"- Label-conflict fraction: {scorecard['quality']['label_conflict']['fraction']:.6f}"
        f" ({scorecard['quality']['label_conflict']['n_conflicted_groups']} groups)",
        f"- Constant features: {scorecard['quality']['constant_features']['count']} "
        f"{scorecard['quality']['constant_features']['features']}",
        f"- Adversarial AUC within period 1: {scorecard['quality']['adversarial']['within_period'].get('auc')}",
        f"- KS within period (median/max): {scorecard['quality']['ks']['within_period'].get('median')}/"
        f"{scorecard['quality']['ks']['within_period'].get('max')}",
        "",
        "## Splits (frozen)",
        "",
        f"- Grouped protocol frozen hash: `{scorecard['splits']['grouped']['frozen_hash']}`",
        f"- Host-pair-disjoint variant feasible: {scorecard['splits']['host_pair_disjoint'].get('feasible')}",
        f"- Frozen masks: `{scorecard['splits']['mask_files']}`",
        "",
        "## Graph lift (decisive G1c item)",
        "",
    ]
    pilot = scorecard.get("graph_lift")
    if pilot and pilot.get("status") == "success":
        pilot_models = pilot.get("mean_over_seeds", {}).get("models", {})
        pilot_mean = pilot.get("mean_over_seeds", {})
        verdict_block = pilot.get("verdict", {})
        lines += [
            f"- MLP macro-F1: {pilot_models.get('mlp', float('nan')):.4f}",
            f"- E-GraphSAGE macro-F1: {pilot_models.get('egraphsage', float('nan')):.4f}",
            f"- Randomised-structure control macro-F1: "
            f"{pilot_models.get('egraphsage_randomised', float('nan')):.4f}",
            f"- Delta(E-GS - MLP): {pilot_mean.get('delta_points_egsage_minus_mlp', float('nan')):+.2f} "
            f"points (CI {verdict_block.get('graph_lift_macro_f1', {}).get('ci95_points')})",
            f"- Delta(E-GS - randomised): "
            f"{pilot_mean.get('delta_points_egsage_minus_randomised', float('nan')):+.2f} points "
            f"(CI {verdict_block.get('graph_lift_vs_randomised', {}).get('ci95_points')})",
            f"- **D-024 overall verdict: {verdict_block.get('overall', 'n/a').upper()}**",
        ]
    elif pilot:
        lines.append(f"- Pilot status: {pilot.get('status')} - {pilot.get('reason')}")
    else:
        lines.append(
            "- Not run in this invocation. The graph-lift pilot is the decisive item for the primary-dataset "
            "decision (D-020/G1c); run it with `--with-pilot` on the legacy lane (small) and canonical lane (full)."
        )
    lines += ["", "## Artifacts", ""]
    for name, path in scorecard["artifacts"].items():
        lines.append(f"- `{name}`: `{path}`")
    if scorecard.get("notes"):
        lines += ["", "## Notes", ""]
        lines += [f"- {note}" for note in scorecard["notes"]]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_audit(
    *,
    spec: DatasetSpec,
    csv_path: str | Path,
    config: Mapping[str, Any] | Config,
    out_dir: str | Path,
    splits_dir: str | Path,
    dataset: str | None = None,
    seed: int | None = None,
    with_pilot: bool = False,
    pilot_runner: Callable[..., dict[str, Any]] | None = None,
    write_run_manifest: bool = True,
    paths: Any | None = None,
) -> dict[str, Any]:
    """Run the complete S-D1 data audit for one dataset."""
    config_dict = config.data if isinstance(config, Config) else dict(config)
    dataset_name = dataset or spec.name
    seed = int(seed if seed is not None else config_dict.get("seed", 42))
    out_path = Path(out_dir)
    splits_path = Path(splits_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    splits_path.mkdir(parents=True, exist_ok=True)

    thresholds = _thresholds(config_dict)
    periods_cfg = _section(config_dict, "periods")
    dedupe_cfg = _section(config_dict, "dedupe")
    near_dup_cfg = _section(config_dict, "near_duplicate")
    sentinel_cfg = _section(config_dict, "sentinels")
    splits_cfg = _section(config_dict, "splits")
    preprocess_cfg = _section(config_dict, "preprocess")
    structure_cfg = _section(config_dict, "structure")
    quality_cfg = _section(config_dict, "quality")
    graph_lift_cfg = _section(config_dict, "graph_lift")

    seed_state = seed_everything(seed)
    timings: dict[str, float] = {}
    notes: list[str] = []
    start_total = time.monotonic()
    # T17/T18 manifest lifecycle: start before the expensive stages.
    run_manifest: RunManifest | None = None
    if write_run_manifest:
        run_manifest = RunManifest(
            "data_audit",
            paths=paths,
            config=Config(data=config_dict),
            lane=getattr(paths, "lane", None),
            seed=seed,
            dataset=dataset_name,
            out_path=out_path / "manifest.json",
            extra={"stage": "S-D1 data audit", "started_before_pipeline": True},
        )
        run_manifest.start()

    def timed(name: str) -> Callable[[], float]:
        return lambda: time.monotonic()

    t = time.monotonic()
    raw = load_raw_frame(spec, csv_path)
    timings["load_s"] = round(time.monotonic() - t, 3)
    raw_rows = int(len(raw))

    t = time.monotonic()
    clean = clean_frame(
        raw,
        spec,
        sentinel_policy=sentinel_cfg,
        near_duplicate=near_dup_cfg,
        dedupe=bool(dedupe_cfg.get("exact", True)),
        label_conflict_policy=str(dedupe_cfg.get("label_conflict_policy", "report_only")),
    )
    timings["clean_s"] = round(time.monotonic() - t, 3)
    frame = clean.frame
    del raw

    t = time.monotonic()
    periods_result = assign_periods(
        frame,
        n_periods=int(periods_cfg["count"]),
        strategy=str(periods_cfg["strategy"]),
        min_flows_per_period=int(periods_cfg.get("min_flows_per_period", 1)),
        min_session_gap_seconds=float(
            periods_cfg.get("min_session_gap_seconds", DEFAULT_SESSION_MIN_GAP_SECONDS)
        ),
    )
    frame = periods_result.frame
    timings["periods_s"] = round(time.monotonic() - t, 3)

    t = time.monotonic()
    fractions = dict(splits_cfg.get("fractions", {"train": 0.7, "val": 0.1, "test": 0.2}))
    host_pair_cfg = dict(splits_cfg.get("host_pair_disjoint", {}))
    splits_grouped = build_splits(
        frame,
        fractions=fractions,
        seed=seed,
        mode="grouped",
        min_groups_per_split=int(host_pair_cfg.get("min_groups_per_split", 10)),
        min_share=float(host_pair_cfg.get("min_share", 0.05)),
    )
    splits_host_pair = None
    if bool(host_pair_cfg.get("enabled", True)):
        splits_host_pair = build_splits(
            frame,
            fractions=fractions,
            seed=seed + 1,
            mode="host_pair_disjoint",
            min_groups_per_split=int(host_pair_cfg.get("min_groups_per_split", 10)),
            min_share=float(host_pair_cfg.get("min_share", 0.05)),
        )
    timings["splits_s"] = round(time.monotonic() - t, 3)

    t = time.monotonic()
    train_period = min(splits_grouped.masks)
    train_positions = splits_grouped.masks[train_period]["train"]
    preprocessor = Preprocessor(spec, config=preprocess_cfg)
    if len(train_positions) == 0:
        raise RuntimeError("period-1 train mask is empty; cannot fit the preprocessor")
    preprocessor.fit(frame.iloc[train_positions])
    preprocessor.save(out_path / "preprocessor_state.json")
    timings["preprocess_fit_s"] = round(time.monotonic() - t, 3)

    t = time.monotonic()
    structure = structure_report(frame, class_map=clean.class_map)
    write_json(out_path / "structure.json", structure)
    timings["structure_s"] = round(time.monotonic() - t, 3)

    t = time.monotonic()
    quality = quality_audit(
        frame,
        splits_grouped,
        preprocessor,
        spec,
        config=quality_cfg,
        seed=seed,
        clean_stats=clean.stats,
    )
    write_json(out_path / "quality_audit.json", quality)
    timings["quality_s"] = round(time.monotonic() - t, 3)

    # Frozen splits: manifests + per-period npz masks (both variants).
    grouped_manifest = freeze_split_masks(splits_grouped, frame, dataset=dataset_name)
    mask_files = write_split_masks(splits_grouped, splits_path, dataset=dataset_name)
    host_pair_manifest: dict[str, Any] = {
        "feasible": None,
        "note": "host-pair-disjoint robustness variant disabled in config",
    }
    if splits_host_pair is not None:
        host_pair_manifest = freeze_split_masks(splits_host_pair, frame, dataset=dataset_name)
        host_pair_manifest["feasible"] = splits_host_pair.stats.get("feasible")
        host_pair_manifest["infeasible_reasons"] = splits_host_pair.stats.get("infeasible_reasons", [])
        host_pair_dir = splits_path / "host_pair_disjoint"
        host_pair_masks = write_split_masks(splits_host_pair, host_pair_dir, dataset=dataset_name)
        host_pair_manifest["mask_files"] = host_pair_masks["mask_files"]
    split_manifest = {
        "schema_version": 1,
        "dataset": dataset_name,
        "generated_at_utc": utc_now(),
        "protocol": "S-D1 leakage-safe split v1",
        "frozen": True,
        "mode": grouped_manifest["mode"],
        "period_strategy": periods_result.stats["strategy"],
        "session_boundary": periods_result.stats.get("session"),
        "frozen_hash": grouped_manifest["frozen_hash"],
        "grouped": grouped_manifest,
        "host_pair_disjoint": host_pair_manifest,
        "mask_files": mask_files["mask_files"],
        "seed": seed,
        "fractions": fractions,
        "class_weight_rule": "balanced class weights computed from each period's train mask only",
        "preprocessing_rule": "all learned statistics fitted on period-1 train only (see preprocessor_state.json)",
    }
    write_json(out_path / "split_manifest.json", split_manifest)

    # Feature manifest with fitted statistics.
    feature_table = _feature_manifest_table(
        spec, preprocessor, final_feature_names=preprocessor.feature_names
    )
    feature_table.to_csv(out_path / "feature_manifest.csv", index=False)

    # Optional graph-lift pilot (decisive G1c item).
    pilot: dict[str, Any] | None = None
    if with_pilot:
        try:
            if pilot_runner is None:
                from src.pilot.graph_lift import run_graph_lift_pilot  # local import: optional torch

                pilot_runner = run_graph_lift_pilot
            t = time.monotonic()
            pilot = pilot_runner(
                frame,
                splits_grouped,
                preprocessor,
                spec,
                config=graph_lift_cfg,
                seed=seed,
                dataset=dataset_name,
                out_dir=out_path,
                thresholds=thresholds,
                period_strategy=str(periods_result.stats["strategy"]),
                config_sha256=hash_config(config_dict),
            )
            timings["graph_lift_s"] = round(time.monotonic() - t, 3)
        except Exception as exc:  # noqa: BLE001 - pilot must not destroy the audit
            LOGGER.warning("graph-lift pilot failed: %s", exc)
            notes.append(f"graph-lift pilot failed: {type(exc).__name__}: {exc}")
            pilot = None

    checks = evaluate_checks(
        thresholds=thresholds,
        quality=quality,
        structure=structure,
        splits_grouped=splits_grouped,
        splits_host_pair=splits_host_pair,
        periods=periods_result.stats,
        pilot=pilot,
        structure_config=structure_cfg,
    )
    status = _overall_status(checks)
    if pilot is None:
        notes.append(
            "graph-lift pilot not run in this invocation; it is the decisive G1c item (D-020). "
            "Run: python src/scripts/run_data_audit.py --dataset <name> --with-pilot"
        )
    if periods_result.stats.get("class_presence"):
        benign_periods = structure.get("benign_only_periods", [])
        if benign_periods:
            notes.append(
                "benign-only period(s) "
                + ", ".join(str(int(p) + 1) for p in benign_periods)
                + ": no attack-class anchors can be drawn there (D-020); anchor rule must handle it at S6."
            )
    conflict_stats = clean.stats.get("label_conflict") or {}
    if conflict_stats.get("policy") == "drop_groups" and conflict_stats.get("rows_dropped"):
        notes.append(
            f"D-032 label-conflict cleaning: dropped {conflict_stats['rows_dropped']:,} rows from "
            f"{conflict_stats['n_conflicted_groups']:,} label-conflicting near-duplicate groups "
            f"before period/split assignment (raw->clean rows "
            f"{clean.stats['rows_in']:,}->{len(frame):,}); examples in label_conflict_examples.json. "
            "The pre-registered label_conflict_fraction threshold is unchanged."
        )

    artifact_paths = {
        "scorecard_json": out_path / "scorecard.json",
        "scorecard_md": out_path / "scorecard.md",
        "feature_manifest_csv": out_path / "feature_manifest.csv",
        "split_manifest_json": out_path / "split_manifest.json",
        "structure_json": out_path / "structure.json",
        "quality_json": out_path / "quality_audit.json",
        "preprocessor_state_json": out_path / "preprocessor_state.json",
    }
    conflict_examples = (clean.stats.get("label_conflict") or {}).get("examples", [])
    if conflict_examples:
        write_json(out_path / "label_conflict_examples.json", {
            "schema_version": 1,
            "dataset": dataset_name,
            "policy": (clean.stats.get("label_conflict") or {}).get("policy"),
            "n_conflicted_groups": (clean.stats.get("label_conflict") or {}).get("n_conflicted_groups"),
            "rows_dropped": (clean.stats.get("label_conflict") or {}).get("rows_dropped"),
            "decision_ref": "D-032 (docs/DECISIONS.md)",
            "examples": conflict_examples,
        })
        artifact_paths["label_conflict_examples_json"] = out_path / "label_conflict_examples.json"
    if pilot is not None:
        artifact_paths["graph_lift_pilot_json"] = out_path / "graph_lift_pilot.json"
        artifact_paths["graph_lift_pilot_md"] = out_path / "graph_lift_pilot.md"
        artifact_paths["graph_lift_pilot_manifest_json"] = out_path / "graph_lift_pilot_manifest.json"

    csv_path_obj = Path(csv_path)
    scorecard: dict[str, Any] = {
        "schema_version": 1,
        "dataset": dataset_name,
        "generated_at_utc": utc_now(),
        "status": status,
        "config": {
            "sha256": hash_config(config_dict),
            "path": str(config.path) if isinstance(config, Config) and config.path else None,
            "sections_used": sorted(
                ["periods", "dedupe", "near_duplicate", "sentinels", "splits", "preprocess",
                 "structure", "quality", "graph_lift", "thresholds"]
            ),
        },
        "seed": seed,
        "seed_state": seed_state,
        "dataset_source": {
            "csv": str(csv_path_obj),
            "bytes": csv_path_obj.stat().st_size if csv_path_obj.is_file() else None,
            "sha256": sha256_file(csv_path_obj) if csv_path_obj.is_file() else None,
        },
        "rows": {
            "raw": raw_rows,
            "after_clean": int(len(frame)),
            "exact_duplicates_removed": int(clean.stats["exact_duplicates_removed"]),
            "published": spec.published_flows,
            "delta_vs_published": (
                None if spec.published_flows is None else int(len(frame)) - int(spec.published_flows)
            ),
        },
        "cleaning": {
            "duplicates": {
                "rows_in": int(clean.stats["rows_in"]),
                "exact_duplicates_removed": int(clean.stats["exact_duplicates_removed"]),
                "rows_after_dedupe": int(clean.stats["rows_after_dedupe"]),
            },
            "label_conflict": clean.stats.get("label_conflict"),
        },
        "periods": {
            "strategy": periods_result.stats["strategy"],
            "n_periods": periods_result.stats["n_periods"],
            "period_rows": periods_result.stats["period_rows"],
            "time_disjoint": periods_result.stats["time_disjoint"],
            "session_boundary": periods_result.stats.get("session"),
            "per_period": {
                str(int(period) + 1): {
                    "rows": block["rows"],
                    "ts_min_utc": block["ts_min_utc"],
                    "ts_max_utc": block["ts_max_utc"],
                    "classes_present": sorted(block["class_counts"]),
                    "class_counts": block["class_counts"],
                }
                for period, block in periods_result.stats["per_period"].items()
            },
        },
        "benign_only_periods": structure.get("benign_only_periods", []),
        "structure": structure,
        "quality": quality,
        "splits": {
            "grouped": grouped_manifest,
            "host_pair_disjoint": host_pair_manifest,
            "mask_files": mask_files["mask_files"],
        },
        "feature_manifest": {
            "n_columns": int(len(feature_table)),
            "n_features_final": len(preprocessor.feature_names),
            "path": str(artifact_paths["feature_manifest_csv"]),
        },
        "preprocessor": {
            "n_features": len(preprocessor.feature_names),
            "missing_indicators": preprocessor.used_missing_indicators(),
            "fit_scope": "period-1 train mask only",
        },
        "graph_lift": _pilot_summary(pilot) if pilot is not None else None,
        "thresholds": thresholds,
        "checks": checks,
        "decision_inputs": _decision_inputs(checks),
        "notes": notes,
        "timings": timings,
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
    }
    scorecard["timings"]["total_s"] = round(time.monotonic() - start_total, 3)
    write_json(artifact_paths["scorecard_json"], scorecard)
    artifact_paths["scorecard_md"].write_text(_scorecard_markdown(scorecard), encoding="utf-8")

    if write_run_manifest and run_manifest is not None:
        run_manifest.payload.setdefault("extra", {}).update(
            {"scorecard_status": status, "timings": timings, "started_before_pipeline": True}
        )
        run_manifest.set_dataset(dataset_name, csv_path_obj.parent, patterns=(csv_path_obj.name,))
        for path in artifact_paths.values():
            run_manifest.add_artifact(path, role="data_audit")
        if pilot is not None:
            for name in ("graph_lift_pilot.json", "graph_lift_pilot.md"):
                candidate = out_path / name
                if candidate.is_file():
                    run_manifest.add_artifact(candidate, role="graph_lift_pilot")
        run_manifest.finish("success" if status != "fail" else "failed")
        run_manifest.write()

    if pilot is not None:
        # PR-5/D-034: the graph-lift gate artifact needs its own pinned manifest
        # (git SHA via full-path fallback, config hash, code fingerprint) so a
        # stale stream-A manifest can never be mistaken for the current run.
        pilot_manifest = RunManifest(
            "graph_lift_pilot",
            paths=None,
            config=Config(data=config_dict),
            lane=None,
            seed=seed,
            dataset=dataset_name,
            out_path=out_path / "graph_lift_pilot_manifest.json",
            extra={
                "pinning": pilot.get("pinning"),
                "verdict": pilot.get("verdict"),
                "sample": pilot.get("sample"),
                "period_strategy": pilot.get("period_strategy"),
                "config_sha256": pilot.get("config_sha256"),
                "scorecard_status": status,
            },
        )
        pilot_manifest.start()
        pilot_manifest.set_dataset(dataset_name, csv_path_obj.parent, patterns=(csv_path_obj.name,))
        for name in ("graph_lift_pilot.json", "graph_lift_pilot.md"):
            candidate = out_path / name
            if candidate.is_file():
                pilot_manifest.add_artifact(candidate, role="graph_lift_pilot")
        pilot_manifest.finish("success" if pilot.get("status") == "success" else "failed")
        pilot_manifest.write()

    LOGGER.info(
        "data audit finished: dataset=%s status=%s checks=%s->%s total=%.1fs",
        dataset_name,
        status,
        {key: sum(1 for check in checks if check["status"] == key) for key in ("pass", "flag", "fail")},
        artifact_paths["scorecard_json"],
        timings["total_s"],
    )
    return scorecard


__all__ = [
    "GIANT_COMPONENT_HOST_COUNT_HARD_MIN",
    "REQUIRED_THRESHOLDS",
    "evaluate_checks",
    "run_audit",
]
