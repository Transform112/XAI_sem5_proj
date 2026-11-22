"""Dataset quality audit (T4 / S-D1).

Records the quantitative evidence behind the audit thresholds and the diagnostic
metrics that inform the dataset review:

* NaN / infinity after cleaning and after preprocessing (must be 0),
* duplicate rates: raw exact duplicates (diagnostic) vs duplicates remaining
  after dedupe (D-023: the 0.1% threshold applies to the remaining rate),
* near-duplicate leakage groups and label-conflict rate (a group whose flows
  carry different classes is a labelling problem, not just a duplicate),
* constant features after preprocessing,
* per-feature Kolmogorov-Smirnov statistics within a period (train vs test) and
  between periods (period-1 train vs each later train set),
* single-feature one-vs-rest AUC and mutual information against the class label
  (how much signal the edge features carry without the graph),
* adversarial validation within a period (can a classifier tell train from test
  edges? the grouped split should make this near chance) and between periods
  (real temporal drift is expected and is reported, not thresholded).

All sampling is seeded and the sample sizes are recorded in the report so the
numbers are reproducible.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from src.data.preprocess import Preprocessor
from src.data.registry import DatasetSpec
from src.utils.logging import get_logger

LOGGER = get_logger("xdrift.quality")

DEFAULT_CONFIG: dict[str, Any] = {
    "ks_sample_rows": 50_000,
    "ks_minimum_rows": 1_000,
    "mutual_information": True,
    "mi_sample_rows": 20_000,
    "single_feature_auc": True,
    "auc_sample_rows": 50_000,
    "adversarial_sample_rows": 50_000,
    "adversarial_model": "hist_gradient_boosting",
    "adversarial_max_iter": 80,
    "adversarial_cv": 3,
    "constant_feature_unique_max": 1,
    "label_conflict_key": "near_duplicate",
}


def _resolve_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    resolved = dict(DEFAULT_CONFIG)
    if config:
        for key, value in dict(config).items():
            if value is not None:
                resolved[key] = value
    return resolved


def _sample_positions(n: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.sort(rng.choice(n, size=limit, replace=False))


def _period_local_masks(
    frame: pd.DataFrame,
    splits: Any,
    period: int,
    period_col: str = "_period",
) -> dict[str, np.ndarray]:
    """Map global frame positions to positions local to one period."""
    positions = np.flatnonzero(frame[period_col].to_numpy(dtype="int64") == period)
    local: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        global_mask = splits.masks[period][name]
        if len(global_mask) == 0:
            local[name] = np.array([], dtype="int64")
            continue
        mapping = np.searchsorted(positions, global_mask)
        if not np.array_equal(positions[mapping], global_mask):
            raise RuntimeError("split mask is inconsistent with the period assignment")
        local[name] = mapping.astype("int64")
    return local


def _transform_rows(preprocessor: Preprocessor, frame: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    if len(positions) == 0:
        return np.empty((0, len(preprocessor.feature_names)), dtype="float64")
    return preprocessor.transform(frame.iloc[positions])


def _constant_features(matrix: np.ndarray, names: Sequence[str], unique_max: int) -> dict[str, Any]:
    constant: list[str] = []
    unique_counts = []
    for index, name in enumerate(names):
        values = matrix[:, index]
        unique = int(np.unique(values).size)
        unique_counts.append(unique)
        if unique <= unique_max:
            constant.append(name)
    return {
        "count": len(constant),
        "features": constant,
        "unique_max": int(unique_max),
        "n_features": len(names),
    }


def _ks_block(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    names: Sequence[str],
    *,
    minimum_rows: int,
) -> dict[str, Any]:
    if len(matrix_a) < minimum_rows or len(matrix_b) < minimum_rows:
        return {"computed": False, "reason": f"need >= {minimum_rows} rows per side"}
    stats = {}
    for index, name in enumerate(names):
        result = ks_2samp(matrix_a[:, index], matrix_b[:, index])
        stats[name] = {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
    values = np.array([entry["statistic"] for entry in stats.values()])
    return {
        "computed": True,
        "n_features": len(stats),
        "n_above_0_1": int((values > 0.1).sum()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "per_feature": stats,
    }


def _single_feature_auc(
    matrix: np.ndarray,
    labels: np.ndarray,
    names: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    classes = np.unique(labels)
    if len(classes) < 2:
        return {"computed": False, "reason": "need at least two classes"}
    per_feature: dict[str, float] = {}
    for index, name in enumerate(names):
        scores = []
        for class_id in classes:
            binary = labels == class_id
            if binary.all() or not binary.any():
                continue
            scores.append(float(roc_auc_score(binary, matrix[:, index])))
        if scores:
            per_feature[name] = float(np.mean(scores))
    ordered = sorted(per_feature.items(), key=lambda item: item[1], reverse=True)
    return {
        "computed": True,
        "classes": [int(value) for value in classes.tolist()],
        "top": [{"feature": name, "macro_ovr_auc": round(score, 6)} for name, score in ordered[:10]],
        "per_feature": {name: round(score, 6) for name, score in per_feature.items()},
    }


def _mutual_information(
    matrix: np.ndarray,
    labels: np.ndarray,
    names: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    if len(np.unique(labels)) < 2:
        return {"computed": False, "reason": "need at least two classes"}
    values = mutual_info_classif(matrix, labels, discrete_features=False, random_state=seed)
    ordered = sorted(zip(names, values.tolist()), key=lambda item: item[1], reverse=True)
    return {
        "computed": True,
        "top": [{"feature": name, "mi": round(float(score), 6)} for name, score in ordered[:10]],
        "per_feature": {name: round(float(score), 6) for name, score in zip(names, values.tolist())},
    }


def _adversarial_auc(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    *,
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    limit = int(config["adversarial_sample_rows"])
    if len(matrix_a) < 50 or len(matrix_b) < 50:
        return {"computed": False, "reason": "need >= 50 rows per side"}
    rng = np.random.default_rng(seed)
    index_a = _sample_positions(len(matrix_a), limit, rng)
    index_b = _sample_positions(len(matrix_b), limit, rng)
    features = np.concatenate([matrix_a[index_a], matrix_b[index_b]], axis=0)
    labels = np.concatenate([np.zeros(len(index_a)), np.ones(len(index_b))]).astype("int8")
    model = HistGradientBoostingClassifier(
        max_iter=int(config["adversarial_max_iter"]),
        random_state=seed,
    )
    folds = int(config["adversarial_cv"])
    folds = max(2, min(folds, int(min(np.bincount(labels)))))
    scores = cross_val_score(model, features, labels, cv=folds, scoring="roc_auc")
    return {
        "computed": True,
        "auc": float(np.mean(scores)),
        "fold_aucs": [float(value) for value in scores.tolist()],
        "folds": folds,
        "n_per_side": int(min(len(index_a), len(index_b))),
        "model": "hist_gradient_boosting",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def quality_audit(
    frame: pd.DataFrame,
    splits: Any,
    preprocessor: Preprocessor,
    spec: DatasetSpec,
    *,
    config: Mapping[str, Any] | None = None,
    seed: int = 42,
    clean_stats: Mapping[str, Any] | None = None,
    period_col: str = "_period",
) -> dict[str, Any]:
    """Run the quality audit; returns a JSON-ready report."""
    cfg = _resolve_config(config)
    rng = np.random.default_rng(seed)
    names = preprocessor.feature_names
    periods = sorted(splits.masks)
    train_period = periods[0]
    last_period = periods[-1]
    local = _period_local_masks(frame, splits, train_period, period_col=period_col)
    period_frame = frame[frame[period_col] == train_period].reset_index(drop=True)
    labels = period_frame["_class_id"].to_numpy(dtype="int64")

    # Full period-1 transformed matrix (also the NaN/inf verification sample).
    matrix_train_period = preprocessor.transform(period_frame)
    finite = np.isfinite(matrix_train_period)
    nan_inf = {
        "total": int((~finite).sum()),
        "fraction": float((~finite).mean()) if finite.size else 0.0,
        "n_rows": int(matrix_train_period.shape[0]),
        "n_features": int(matrix_train_period.shape[1]),
        "scope": "full period-1 transform (all its train/val/test rows)",
    }

    # Constant features: fitted-train rows of the first period.
    train_matrix = matrix_train_period[local["train"]]
    constant = _constant_features(
        train_matrix, names, int(cfg["constant_feature_unique_max"])
    )

    # Duplicates and label conflicts.
    duplicate_raw = 0.0
    rows_in = None
    if clean_stats:
        rows_in = clean_stats.get("rows_in")
        removed = int(clean_stats.get("exact_duplicates_removed", 0))
        duplicate_raw = (removed / rows_in) if rows_in else 0.0
    duplicate_remaining = 0.0
    if clean_stats and "exact_duplicates_remaining" in clean_stats:
        duplicate_remaining = (
            clean_stats["exact_duplicates_remaining"] / clean_stats.get("rows_after_dedupe", 1)
        )
    conflict = _label_conflict(frame, period_col=period_col)
    conflict_diagnostic = _label_conflict_diagnostic(frame, spec, time_bucket_ms=1)

    # KS: within period (train vs test) and between periods (p1 train vs later train).
    rng_ks = np.random.default_rng(seed + 1)
    train_sample = local["train"][
        _sample_positions(len(local["train"]), int(cfg["ks_sample_rows"]), rng_ks)
    ]
    test_sample = local["test"][
        _sample_positions(len(local["test"]), int(cfg["ks_sample_rows"]), rng_ks)
    ]
    within_a = _transform_rows(preprocessor, period_frame, train_sample)
    within_b = _transform_rows(preprocessor, period_frame, test_sample)
    ks_within = _ks_block(within_a, within_b, names, minimum_rows=int(cfg["ks_minimum_rows"]))
    train_reference = matrix_train_period[train_sample]
    ks_between: dict[str, Any] = {}
    between_rows: dict[str, int] = {}
    for period in periods[1:]:
        period_subset = frame[frame[period_col] == period]
        if len(period_subset) == 0:
            continue
        positions = _sample_positions(len(period_subset), int(cfg["ks_sample_rows"]), rng_ks)
        other = preprocessor.transform(period_subset.iloc[positions])
        block = _ks_block(
            train_reference,
            other,
            names,
            minimum_rows=int(cfg["ks_minimum_rows"]),
        )
        ks_between[str(int(period) + 1)] = block  # 1-based for the report
        between_rows[str(int(period) + 1)] = int(len(positions))
        del other
    ks = {
        "within_period": ks_within,
        "between_periods": ks_between,
        "between_period_row_counts": between_rows,
    }

    # Single-feature AUC and mutual information on period-1 train rows.
    rng_auc = np.random.default_rng(seed + 2)
    auc_positions = local["train"][
        _sample_positions(len(local["train"]), int(cfg["auc_sample_rows"]), rng_auc)
    ]
    auc_matrix = matrix_train_period[auc_positions]
    auc_labels = labels[auc_positions]
    single_feature_auc = (
        _single_feature_auc(auc_matrix, auc_labels, names, seed=seed)
        if cfg.get("single_feature_auc", True)
        else {"computed": False, "reason": "disabled in config"}
    )
    rng_mi = np.random.default_rng(seed + 3)
    mi_positions = local["train"][
        _sample_positions(len(local["train"]), int(cfg["mi_sample_rows"]), rng_mi)
    ]
    mutual_information = (
        _mutual_information(
            matrix_train_period[mi_positions], labels[mi_positions], names, seed=seed
        )
        if cfg.get("mutual_information", True)
        else {"computed": False, "reason": "disabled in config"}
    )

    # Adversarial validation: within period-1 and between period-1 / last period.
    adversarial_within = _adversarial_auc(
        matrix_train_period[local["train"]],
        matrix_train_period[local["test"]],
        config=cfg,
        seed=seed,
    )
    last_subset = frame[frame[period_col] == last_period]
    adversarial_between: dict[str, Any] = {"computed": False, "reason": "single period"}
    if len(last_subset) > 0 and last_period != train_period:
        rng_adv = np.random.default_rng(seed + 4)
        last_positions = _sample_positions(len(last_subset), int(cfg["adversarial_sample_rows"]), rng_adv)
        last_matrix = preprocessor.transform(last_subset.iloc[last_positions])
        adversarial_between = _adversarial_auc(
            matrix_train_period[local["train"]],
            last_matrix,
            config=cfg,
            seed=seed + 4,
        )

    # Class distribution per period/split (from the split statistics).
    class_distribution = {
        str(period + 1): splits.stats["periods"].get(period, {}).get("splits", {})
        for period in periods
    }

    report = {
        "schema_version": 1,
        "dataset": spec.name,
        "seed": seed,
        "config": cfg,
        "nan_inf": nan_inf,
        "duplicates": {
            "duplicate_fraction_raw": duplicate_raw,
            "duplicate_fraction_remaining": duplicate_remaining,
            "raw_duplicate_rows": int((clean_stats or {}).get("exact_duplicates_removed", 0)),
            "rows_in": rows_in,
            "note": "D-023: the 0.1% threshold applies to the remaining-after-dedupe rate; "
            "the raw rate is diagnostic only",
        },
        "label_conflict": conflict,
        "label_conflict_diagnostic_finer_time_bucket": conflict_diagnostic,
        "constant_features": constant,
        "ks": ks,
        "single_feature_auc": single_feature_auc,
        "mutual_information": mutual_information,
        "adversarial": {
            "within_period": adversarial_within,
            "within_period_auc": adversarial_within.get("auc"),
            "between_periods": {
                "first_period": int(train_period) + 1,
                "last_period": int(last_period) + 1,
                **adversarial_between,
            },
        },
        "class_distribution": class_distribution,
        "per_feature": _per_feature_table(
            names, ks, single_feature_auc, mutual_information
        ),
    }
    LOGGER.info(
        "quality audit: nan_inf=%s duplicates_raw=%.5f remaining=%.5f conflict=%.5f adv_within=%s",
        nan_inf["total"],
        duplicate_raw,
        duplicate_remaining,
        conflict["fraction"],
        adversarial_within.get("auc"),
    )
    return report


def _label_conflict(frame: pd.DataFrame, *, period_col: str = "_period") -> dict[str, Any]:
    """Near-duplicate groups (config-defined canonical key) whose rows carry >1 class.

    D-034: the key is exactly the config-defined one
    (``near_duplicate.keys`` + ``time_bucket_ms``), computed in
    :func:`src.data.clean.clean_frame` into ``_group_id`` using global host ids.
    This is the pre-registered metric behind ``label_conflict_fraction_max``.
    """
    pairs = frame[["_group_id", "_class_id"]].drop_duplicates()
    group_counts = pairs.groupby("_group_id").size()
    conflicted_groups = group_counts[group_counts > 1]
    unique_groups = int(frame["_group_id"].nunique())
    conflicted_rows = 0
    if len(conflicted_groups):
        conflicted_rows = int(np.isin(frame["_group_id"].to_numpy(), conflicted_groups.index.to_numpy()).sum())
    n_edges = int(len(frame))
    return {
        "key": "config near_duplicate key (host_src|host_dst|port_src|port_dst|protocol + time_bucket)",
        "n_groups": unique_groups,
        "n_conflicted_groups": int(len(conflicted_groups)),
        "fraction": float(len(conflicted_groups) / unique_groups) if unique_groups else 0.0,
        "n_conflicted_edges": conflicted_rows,
        "fraction_edges": float(conflicted_rows / n_edges) if n_edges else 0.0,
    }


def _label_conflict_diagnostic(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    *,
    time_bucket_ms: int = 1,
) -> dict[str, Any]:
    """Informational only: label conflicts under a FINER time bucket (D-034).

    Clearly not thresholded; reported so the paper can state how sensitive the
    conflict rate is to the canonical key's 1-second bucket.
    """
    src = frame["_host_src_id"].to_numpy(dtype="int64")
    dst = frame["_host_dst_id"].to_numpy(dtype="int64")
    sport = pd.to_numeric(frame[spec.column("port_src")], errors="coerce").fillna(-1).astype("int64").to_numpy()
    dport = pd.to_numeric(frame[spec.column("port_dst")], errors="coerce").fillna(-1).astype("int64").to_numpy()
    protocol = pd.to_numeric(frame[spec.column("protocol")], errors="coerce").fillna(-1).astype("int64").to_numpy()
    ts_bucket = (frame["_ts"].to_numpy(dtype="int64") // max(int(time_bucket_ms), 1)).astype("int64")
    hashes = pd.util.hash_pandas_object(
        pd.DataFrame({"src": src, "dst": dst, "sport": sport, "dport": dport, "protocol": protocol, "ts": ts_bucket}),
        index=False,
    ).to_numpy(dtype="uint64")
    _, group_ids = np.unique(hashes, return_inverse=True)
    pairs = pd.DataFrame({"_g": group_ids, "_c": frame["_class_id"].to_numpy(dtype="int64")}).drop_duplicates()
    counts = pairs.groupby("_g").size()
    conflicted = counts[counts > 1]
    n_groups = int(len(counts))
    conflicted_edges = 0
    if len(conflicted):
        conflicted_edges = int(np.isin(group_ids, conflicted.index.to_numpy()).sum())
    n_edges = int(len(frame))
    return {
        "time_bucket_ms": int(time_bucket_ms),
        "n_groups": n_groups,
        "n_conflicted_groups": int(len(conflicted)),
        "fraction": float(len(conflicted) / n_groups) if n_groups else 0.0,
        "fraction_edges": float(conflicted_edges / n_edges) if n_edges else 0.0,
        "note": "informational diagnostic (finer time bucket); not thresholded (D-034)",
    }


def _per_feature_table(
    names: Sequence[str],
    ks: Mapping[str, Any],
    auc: Mapping[str, Any],
    mi: Mapping[str, Any],
) -> list[dict[str, Any]]:
    within = ks.get("within_period", {}).get("per_feature", {}) or {}
    auc_values = auc.get("per_feature", {}) or {}
    mi_values = mi.get("per_feature", {}) or {}
    table = []
    for name in names:
        entry: dict[str, Any] = {"name": name}
        if name in within:
            entry["ks_within"] = round(float(within[name]["statistic"]), 6)
        if name in auc_values:
            entry["auc_macro_ovr"] = round(float(auc_values[name]), 6)
        if name in mi_values:
            entry["mutual_information"] = round(float(mi_values[name]), 6)
        between = ks.get("between_periods", {}) or {}
        for period_label, block in between.items():
            per_feature = (block or {}).get("per_feature", {}) or {}
            if name in per_feature:
                entry[f"ks_between_{period_label}"] = round(float(per_feature[name]["statistic"]), 6)
        table.append(entry)
    return table


__all__ = ["DEFAULT_CONFIG", "quality_audit"]
