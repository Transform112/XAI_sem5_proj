"""RQ3: does update-induced drift add predictive value for later per-class forgetting?

Authority
---------
* ``docs/05_EXPERIMENT_PROTOCOL.md`` §11 "RQ3 — drift as early-warning (S11,
  exploratory)": per-class F1 matrix from S5; later-forgetting definition
  ``F_{c,j}(i) = max_{k<=i} R[k,j,c] - R[i,j,c]`` and the **pre-registered**
  binary later-drop label (drop >= delta = 0.05 within the next update or final,
  no post-hoc variants); features = mean/90th-percentile excess drift of
  class-c anchors at checkpoint i, retained fraction, neighbourhood change, plus
  controls (anchor count, class F1 at i); **competing baselines** = anchor
  confidence/margin statistics, retained-anchor fraction, and the class's F1
  level; L2 logistic regression, leave-one-period-out CV where possible, AUC with
  bootstrap CI, ΔAUC (drift + baselines vs baselines only) with CI, effect sizes,
  exploratory caveat (T = 5).
* ``docs/source_documents/Doc 2b`` §S11 mirrors the above and requires the pre-registered
  rule in ``docs/05`` (it is there), no future-F1 leakage into features, and
  association-only wording.
* ``docs/DECISIONS.md`` D-019 (G3 action): RQ3 must show incremental value
  beyond confidence/margin/retention, otherwise H3 is reported as not supported.

Ambiguities flagged to the Lead (not resolved silently)
-------------------------------------------------------
* **A8 — analysis unit.** docs/05 §11 describes features "at checkpoint i for
  class c" but the label is defined per ``(i, j, c)`` (class c evaluated on
  period j). This module joins class-level features at checkpoint i onto the
  label rows ``(i, j, c)`` and reports the unit in the output; the alternative
  (one row per ``(i, c)`` with the label aggregated over j) is not implemented
  because it would average labels across periods and could hide a real drop.
* **A9 — what "leave-one-period-out" means here.** Default CV group is the
  checkpoint index (the same index that drives the label's "later" horizon);
  ``group_column='eval_period'`` is available as a declared robustness variant
  and both are reported when requested. Neither changes the pre-registered rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.analysis import metrics_cl as mc
from src.analysis import stats as st

#: Feature columns derived from drift aggregates (docs/05 §11).
DRIFT_FEATURE_COLUMNS: tuple[str, ...] = ("mean_excess_drift", "p90_excess_drift")

#: Competing-baseline columns (docs/05 §11): confidence/margin, retention, F1 level.
BASELINE_FEATURE_COLUMNS: tuple[str, ...] = ("confidence_mean", "margin_mean", "retained_fraction", "f1")

#: Controls that are reported but are not the competing baseline family.
CONTROL_COLUMNS: tuple[str, ...] = ("anchor_count",)

#: Pre-registered label column written by :func:`build_forgetting_labels`.
LABEL_COLUMN: str = "label"


class RQ3Error(ValueError):
    """Raised when the RQ3 analysis plan cannot be executed on the given inputs."""


@dataclass(frozen=True)
class RQ3Config:
    """Frozen RQ3 analysis defaults (docs/05 §11 + §13)."""

    C: float = 1.0
    alpha: float = st.DEFAULT_ALPHA
    n_resamples: int = st.DEFAULT_N_RESAMPLES
    #: Coefficient CIs are supplementary (the main AUC/delta CIs use
    #: ``n_resamples``); refitting the model is expensive, so this is a declared
    #: separate budget and the count used is recorded in the table.
    n_bootstrap_coef: int = 200
    seed: int = st.DEFAULT_SEED
    group_column: str = "checkpoint"  # ambiguity A9: or "eval_period"
    delta: float = mc.DEFAULT_FORGETTING_DELTA
    horizon: str = "next_or_final"
    dropna: bool = True

    def __post_init__(self) -> None:
        if self.delta <= 0:
            raise RQ3Error(f"delta must be positive, got {self.delta}")
        if self.horizon not in {"next_or_final", "any_later"}:
            raise RQ3Error(f"unknown horizon {self.horizon!r}")


@dataclass(frozen=True)
class RQ3Result:
    """Tables produced by :func:`competing_baseline_analysis`."""

    auc_table: pd.DataFrame
    delta_auc_table: pd.DataFrame
    coefficient_table: pd.DataFrame
    n_rows: int
    n_positive: int
    group_column: str
    notes: tuple[str, ...] = ()

    def as_frames(self) -> dict[str, pd.DataFrame]:
        return {
            "rq3_auc": self.auc_table,
            "rq3_delta_auc": self.delta_auc_table,
            "rq3_coefficients": self.coefficient_table,
        }


# ---------------------------------------------------------------------------
# Labels and features
# ---------------------------------------------------------------------------
def build_forgetting_labels(
    tensor: mc.RMetricTensor,
    *,
    delta: float = mc.DEFAULT_FORGETTING_DELTA,
    horizon: str = "next_or_final",
) -> pd.DataFrame:
    """Per ``(checkpoint, eval_period, class)`` later-drop labels (docs/05 §11).

    Wraps :func:`analysis.metrics_cl.later_drop_rows`; the rule (delta = 0.05,
    next update or final) is pre-registered in ``docs/05`` and must not be
    varied after seeing outcomes.
    """
    rows = mc.later_drop_rows(tensor, delta=delta, horizon=horizon)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RQ3Error(
            "no later-forgetting labels were produced; the per-class tensor needs at least two checkpoints "
            "with future observations"
        )
    return frame


def build_rq3_features(
    labels_frame: pd.DataFrame,
    drift_frame: pd.DataFrame,
    *,
    retention_frame: pd.DataFrame | None = None,
    rq2_frame: pd.DataFrame | None = None,
    drift_value_column: str = "excess",
    anchor_id_column: str = "anchor_id",
    class_column: str = "class",
    checkpoint_column: str = "checkpoint",
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Join checkpoint-level class features onto the ``(i, j, c)`` label rows.

    ``drift_frame`` must contain one row per anchor-update at
    ``checkpoint`` (the update endpoint ``i``); columns ``excess`` and,
    optionally, ``drift_primary``/``confidence``/``margin`` are aggregated to
    ``mean_excess_drift``, ``p90_excess_drift``, ``mean_drift_primary``,
    ``confidence_mean``, ``margin_mean``. ``retention_frame`` and ``rq2_frame``
    optionally add ``retained_fraction`` and ``mean_neighbourhood_change``.
    Labels are never recomputed here; future F1 never enters a feature
    (docs/05 §11 check).
    """
    required_label_columns = {label_column, class_column, checkpoint_column, "f1"}
    missing = required_label_columns - set(labels_frame.columns)
    if missing:
        raise RQ3Error(f"labels frame lacks columns {sorted(missing)}")
    required_drift_columns = {class_column, checkpoint_column, drift_value_column}
    missing_drift = required_drift_columns - set(drift_frame.columns)
    if missing_drift:
        raise RQ3Error(f"drift frame lacks columns {sorted(missing_drift)}")

    strategy_keys = [column for column in ("strategy", "model") if column in drift_frame.columns and column in labels_frame.columns]
    explainer_key = "explainer" if "explainer" in drift_frame.columns else None
    group_keys = strategy_keys + [checkpoint_column, class_column]
    if explainer_key:
        group_keys.append(explainer_key)

    drift = drift_frame.copy()
    if drift_value_column not in drift.columns:
        raise RQ3Error(f"drift frame lacks {drift_value_column!r}")
    aggregations: dict[str, Any] = {
        "mean_excess_drift": (drift_value_column, "mean"),
        "p90_excess_drift": (drift_value_column, lambda values: float(np.nanpercentile(values, 90))),
    }
    if "drift_primary" in drift.columns:
        aggregations["mean_drift_primary"] = ("drift_primary", "mean")
    if "confidence" in drift.columns:
        aggregations["confidence_mean"] = ("confidence", "mean")
    if "margin" in drift.columns:
        aggregations["margin_mean"] = ("margin", "mean")
    if anchor_id_column in drift.columns:
        aggregations["anchor_count"] = (anchor_id_column, "nunique")
    else:
        aggregations["anchor_count"] = (drift_value_column, "count")
    features = drift.groupby(group_keys, dropna=False, sort=True).agg(**aggregations).reset_index()

    if retention_frame is not None:
        retention_keys = [column for column in group_keys if column in retention_frame.columns]
        if "retained_fraction" not in retention_frame.columns:
            raise RQ3Error("retention_frame lacks 'retained_fraction'")
        retention = (
            retention_frame.groupby(retention_keys, dropna=False, sort=True)["retained_fraction"].mean().reset_index()
        )
        features = features.merge(retention, on=retention_keys, how="left", validate="many_to_one")
    if rq2_frame is not None:
        rq2_keys = [column for column in group_keys if column in rq2_frame.columns]
        change_column = None
        for candidate in ("neighbourhood_change", "drift_primary", "excess"):
            if candidate in rq2_frame.columns and candidate != drift_value_column:
                change_column = candidate
                break
        if change_column is None:
            raise RQ3Error("rq2_frame needs a 'neighbourhood_change' column (or the configured fallback)")
        change = rq2_frame.groupby(rq2_keys, dropna=False, sort=True)[change_column].mean().reset_index()
        change = change.rename(columns={change_column: "mean_neighbourhood_change"})
        features = features.merge(change, on=rq2_keys, how="left", validate="many_to_one")

    merge_keys = [column for column in group_keys if column in labels_frame.columns]
    # ``f1`` is the class's F1 level at checkpoint i (a baseline), not a future value.
    labels = labels_frame.copy()
    merged = labels.merge(features, on=merge_keys, how="inner", validate="many_to_one", suffixes=("", "_drift"))
    if merged.empty:
        raise RQ3Error(f"labels and drift features share no rows on keys {merge_keys}")
    merged[LABEL_COLUMN] = merged[label_column].astype(int)
    if not set(np.unique(merged[LABEL_COLUMN])).issubset({0, 1}):
        raise RQ3Error("label column must be binary after joining")
    return merged


# ---------------------------------------------------------------------------
# Competing-baseline analysis
# ---------------------------------------------------------------------------
def competing_baseline_analysis(
    features_frame: pd.DataFrame,
    *,
    drift_columns: Sequence[str] = DRIFT_FEATURE_COLUMNS,
    baseline_columns: Sequence[str] = BASELINE_FEATURE_COLUMNS,
    label_column: str = LABEL_COLUMN,
    cfg: RQ3Config | None = None,
    extra_control_columns: Sequence[str] = (),
) -> RQ3Result:
    """H3 test: does drift add AUC beyond confidence/margin/retention/F1?

    Fits four L2-logistic models under leave-one-period-out CV (docs/05 §11):
    ``majority`` (constant, AUC = 0.5 reference), ``baselines_only`` (competing
    predictors), ``drift_only`` and ``drift_plus_baselines``. The headline
    delta is ``AUC(drift_plus_baselines) - AUC(baselines_only)`` with a paired
    bootstrap CI and p-value; coefficients are reported in standardised units
    with bootstrap CIs. If the delta CI contains 0 the result is reported as
    "H3 not supported" — the function never hides that outcome.
    """
    config = cfg or RQ3Config()
    available_drift = [column for column in drift_columns if column in features_frame.columns]
    available_baseline = [column for column in baseline_columns if column in features_frame.columns]
    available_extra = [column for column in extra_control_columns if column in features_frame.columns]
    if not available_drift:
        raise RQ3Error(f"no drift feature columns present; expected some of {list(drift_columns)}")
    if not available_baseline:
        raise RQ3Error(f"no competing-baseline columns present; expected some of {list(baseline_columns)}")
    if label_column not in features_frame.columns:
        raise RQ3Error(f"features frame lacks label column {label_column!r}")
    if config.group_column not in features_frame.columns:
        raise RQ3Error(f"features frame lacks CV group column {config.group_column!r}")

    frame = features_frame.copy()
    needed = list(dict.fromkeys(available_drift + available_baseline + available_extra + [label_column, config.group_column]))
    if config.dropna:
        frame = frame.dropna(subset=needed)
    elif frame[needed].isna().any().any():
        raise RQ3Error("RQ3 features contain NaN; pass dropna=True (default) or fix the upstream aggregation")
    if frame.empty:
        raise RQ3Error(f"no complete rows for label={label_column!r} and features {needed}")

    y = frame[label_column].to_numpy(dtype=int)
    groups = frame[config.group_column].to_numpy()
    if np.unique(y).size < 2:
        raise RQ3Error(
            "the later-drop label has a single class; H3 cannot be evaluated (report as not supported, not as AUC=0.5)"
        )
    if len(np.unique(groups)) < 2:
        raise RQ3Error(f"leave-one-period-out CV needs at least two {config.group_column!r} groups")
    if len(frame) < 8:
        notes = (f"very small sample (n={len(frame)}); exploratory only (docs/05 §11)",)
    else:
        notes = ()

    drift_X = frame[available_drift].to_numpy(dtype=float)
    baseline_X = frame[available_baseline].to_numpy(dtype=float)
    combined_X = np.column_stack([baseline_X, drift_X, frame[available_extra].to_numpy(dtype=float)]) if available_extra else np.column_stack([baseline_X, drift_X])
    baseline_with_extra = (
        np.column_stack([baseline_X, frame[available_extra].to_numpy(dtype=float)]) if available_extra else baseline_X
    )

    fits: dict[str, st.AucResult] = {}
    fits["baselines_only"] = st.logistic_lopo_auc(baseline_with_extra, y, groups, C=config.C)
    fits["drift_only"] = st.logistic_lopo_auc(drift_X, y, groups, C=config.C)
    fits["drift_plus_baselines"] = st.logistic_lopo_auc(combined_X, y, groups, C=config.C)

    auc_rows: list[dict[str, Any]] = [
        {
            "model": "majority_reference",
            "role": "reference",
            "auc": 0.5,
            "ci_lo": 0.5,
            "ci_hi": 0.5,
            "n": int(len(frame)),
            "n_pos": int(np.sum(y == 1)),
            "n_neg": int(np.sum(y == 0)),
            "n_folds": int(len(np.unique(groups))),
            "baseline_accuracy": st.majority_rate(y),
            "note": "constant predictor: AUC = 0.5 by definition; accuracy baseline = majority rate",
        }
    ]
    for role, model_name in (("competing_baseline", "baselines_only"), ("drift_only", "drift_only"), ("combined", "drift_plus_baselines")):
        result = fits[model_name]
        interval = st.bootstrap_auc(result.y, result.oof_scores, n_resamples=config.n_resamples, alpha=config.alpha, seed=config.seed)
        auc_rows.append(
            {
                "model": model_name,
                "role": role,
                "auc": float(result.auc),
                "ci_lo": interval.lo,
                "ci_hi": interval.hi,
                "n": int(result.n),
                "n_pos": int(result.n_pos),
                "n_neg": int(result.n_neg),
                "n_folds": int(result.n_folds),
                "baseline_accuracy": st.majority_rate(y),
                "note": "; ".join(result.notes) if result.notes else "",
            }
        )
    auc_table = pd.DataFrame(auc_rows)

    delta_rows: list[dict[str, Any]] = []
    for comparison, result, other in (
        ("drift_plus_baselines_vs_baselines_only", fits["drift_plus_baselines"], fits["baselines_only"]),
        ("drift_only_vs_baselines_only", fits["drift_only"], fits["baselines_only"]),
    ):
        delta = st.paired_auc_delta_bootstrap(
            y,
            result.oof_scores,
            other.oof_scores,
            n_resamples=config.n_resamples,
            alpha=config.alpha,
            seed=config.seed,
        )
        supported = delta.lo > 0.0
        delta_rows.append(
            {
                "comparison": comparison,
                "auc_a": delta.auc_a,
                "auc_b": delta.auc_b,
                "delta_auc": delta.delta,
                "ci_lo": delta.lo,
                "ci_hi": delta.hi,
                "p_value": delta.p_value,
                "n": int(delta.n),
                "n_resamples": int(delta.n_resamples),
                "h3_supported": bool(supported),
                "caveat": "exploratory, T=5, association only (docs/05 §11)",
            }
        )
    delta_table = pd.DataFrame(delta_rows)

    coefficient_rows: list[dict[str, Any]] = []
    for model_name, design_columns, design_matrix in (
        ("baselines_only", available_baseline + available_extra, baseline_with_extra),
        ("drift_plus_baselines", available_baseline + available_extra + available_drift, combined_X),
    ):
        coefficient_rows.extend(
            _bootstrap_coefficients(
                design_matrix,
                y,
                design_columns,
                model_name=model_name,
                n_resamples=config.n_bootstrap_coef,
                alpha=config.alpha,
                seed=config.seed,
                C=config.C,
            )
        )
    coefficient_table = pd.DataFrame(coefficient_rows)

    return RQ3Result(
        auc_table=auc_table,
        delta_auc_table=delta_table,
        coefficient_table=coefficient_table,
        n_rows=int(len(frame)),
        n_positive=int(np.sum(y == 1)),
        group_column=config.group_column,
        notes=notes,
    )


def _bootstrap_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    model_name: str,
    n_resamples: int,
    alpha: float,
    seed: int,
    C: float,
) -> list[dict[str, Any]]:
    """Standardised logistic coefficients with case-control bootstrap CIs."""
    point = st.fit_logistic_l2(X, y, C=C, standardize=True, random_state=seed)
    coefficients = point["coef_standardized"]
    rng = np.random.default_rng(seed)
    groups = [np.where(y == value)[0] for value in np.unique(y)]
    boot = np.empty((n_resamples, coefficients.size), dtype=float)
    used = 0
    for _ in range(n_resamples):
        indices = np.concatenate([rng.choice(group, size=group.size, replace=True) for group in groups])
        try:
            fitted = st.fit_logistic_l2(X[indices], y[indices], C=C, standardize=True, random_state=seed)
        except st.StatisticsError:  # pragma: no cover - degenerate resample
            continue
        boot[used] = fitted["coef_standardized"]
        used += 1
    boot = boot[:used]
    rows: list[dict[str, Any]] = []
    for index, feature_name in enumerate(feature_names):
        if used:
            lo = float(np.percentile(boot[:, index], 100.0 * alpha / 2.0))
            hi = float(np.percentile(boot[:, index], 100.0 * (1.0 - alpha / 2.0)))
        else:  # pragma: no cover - defensive
            lo = hi = float("nan")
        rows.append(
            {
                "model": model_name,
                "feature": str(feature_name),
                "coef_standardized": float(coefficients[index]),
                "ci_lo": lo,
                "ci_hi": hi,
                "n_bootstrap": int(used),
            }
        )
    return rows


__all__ = [
    "BASELINE_FEATURE_COLUMNS",
    "CONTROL_COLUMNS",
    "DRIFT_FEATURE_COLUMNS",
    "LABEL_COLUMN",
    "RQ3Config",
    "RQ3Error",
    "RQ3Result",
    "build_forgetting_labels",
    "build_rq3_features",
    "competing_baseline_analysis",
]
