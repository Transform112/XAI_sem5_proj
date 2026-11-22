"""Noise floor, excess drift and RQ1/H1 statistics (stages S8/S9).

Authority
---------
* ``docs/source_documents/Doc 2b`` Â§S8 "Drift metrics and noise floor": three noise-floor
  components â€” (1) *explainer noise* = repeated GNNExplainer runs (R = 10 seeds)
  on the same checkpoint/anchor, pairwise drift distribution â†’ q95;
  (2) *model noise (null update)* = retrain on the same period-j data with a
  different seed (FT on j, R = 5 seeds), explain the same anchors with the same
  explainer seed â†’ drift ``M_j^s -> M_j^s'`` â†’ q95; (3) *combined null*. Per-class
  noise threshold ``q95(anchor)`` from the pooled null distribution of its class
  and explainer (record the pooling rule; few anchors per class â†’ documented
  fallback). ``excess = drift_real - mean(drift_null)``;
  ``above_floor = drift_real > q95``. Both are reported.
* ``docs/source_documents/Doc 2b`` Â§S9 / ``docs/05`` Â§9: H1 = fraction of anchors above the
  null-update q95, stratified bootstrap CI, comparison to the 5 % expected by
  construction (binomial test **and** normal approximation), paired Wilcoxon
  signed-rank on excess vs 0, Holm correction within the
  (strategy/model/explainer) family, raw/null/excess reported side by side,
  retained-anchor fraction reported alongside.
* ``docs/05`` Â§13: 1000 stratified resamples, percentile CIs, effect sizes.
* ``docs/DECISIONS.md`` D-030: the hierarchical pilot bootstrap (500 anchor
  resamples/seed â†’ 1000 seed-level resamples) is implemented in ``stats`` and is
  the *only* accepted deviation; it is not used here.

Ambiguities resolved (reported to the Lead, not hidden)
-------------------------------------------------------
* **A6 â€” which null distribution defines the H1 floor.** Doc 2b Â§S8 lists three
  components and then says "q95 from the pooled null distribution"; the C1 claim
  text says "null-update 95th-percentile floor". Default here is
  ``channel='model_null'`` (the claim wording), with ``explainer_repeat`` and
  ``combined`` available as declared robustness channels. Nothing is
  cherry-picked: every configured channel is reported.
* **A7 â€” pooling fallback.** Classes with fewer than
  ``min_anchors_per_class`` null pairs (default 20) fall back to the *global*
  pooled q95 and the fallback is recorded per row in the ``threshold_source``
  column, so the verifier can see exactly where the class threshold was not used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.analysis import metrics_drift as md
from src.analysis import stats as st

#: Noise-floor channels (Doc 2b Â§S8 components 1â€“3).
MODEL_NULL: str = "model_null"
EXPLAINER_REPEAT: str = "explainer_repeat"
COMBINED: str = "combined"
CHANNELS: tuple[str, ...] = (MODEL_NULL, EXPLAINER_REPEAT, COMBINED)

#: Default channel for H1 (C1 claim wording; ambiguity A6).
PRIMARY_CHANNEL: str = MODEL_NULL

#: Expected fraction above q95 by construction (docs/05 Â§9 H1).
EXPECTED_ABOVE_FRACTION: float = 0.05


class NoiseFloorError(ValueError):
    """Raised when noise-floor inputs violate the S8/S9 contract."""


@dataclass(frozen=True)
class NoiseFloorConfig:
    """Frozen S8/S9 noise-floor defaults (docs/05 Â§9, Doc 2b Â§S8).

    ``n_null_reseeds = 5`` and ``n_explainer_repeats = 10`` are protocol
    constants (docs/05 Â§9); the bootstrap defaults come from docs/05 Â§13.
    """

    n_null_reseeds: int = 5
    n_explainer_repeats: int = 10
    quantile: float = 0.95
    n_resamples: int = st.DEFAULT_N_RESAMPLES
    alpha: float = st.DEFAULT_ALPHA
    seed: int = st.DEFAULT_SEED
    min_anchors_per_class: int = 20
    pooling: str = "class"  # "class" | "global"
    primary_channel: str = PRIMARY_CHANNEL
    # T49: a one-group SMOKE wiring run cannot form the replay/EWC paired
    # comparison; production runs keep this False (missing strategies still error).
    allow_partial_strategies: bool = False

    def __post_init__(self) -> None:
        if self.pooling not in {"class", "global"}:
            raise NoiseFloorError(f"pooling must be 'class' or 'global', got {self.pooling!r}")
        if self.primary_channel not in CHANNELS:
            raise NoiseFloorError(f"unknown channel {self.primary_channel!r}; expected one of {CHANNELS}")
        if not 0.5 < self.quantile < 1.0:
            raise NoiseFloorError(f"quantile must be in (0.5, 1), got {self.quantile}")


# ---------------------------------------------------------------------------
# Drift-row helpers
# ---------------------------------------------------------------------------
def _drift_columns(
    vector_a: Any,
    vector_b: Any,
    *,
    metric_set: str,
    ks: Sequence[int],
    m: int,
) -> dict[str, float]:
    if metric_set == "feature":
        similarities = md.attribution_similarity(vector_a, vector_b, ks=ks)
    elif metric_set == "edge":
        similarities = md.edge_mask_similarity(vector_a, vector_b, m=m)
    else:
        raise NoiseFloorError(f"metric_set must be 'feature' or 'edge', got {metric_set!r}")
    row: dict[str, float] = {}
    for name, value in similarities.items():
        row[f"sim_{name}"] = float(value)
        row[f"drift_{name}"] = float("nan") if not np.isfinite(value) else 1.0 - float(value)
    primary = md.PRIMARY_FEATURE_METRIC if metric_set == "feature" else md.PRIMARY_EDGE_METRIC
    row["sim_primary"] = row[f"sim_{primary}"]
    row["drift_primary"] = row[f"drift_{primary}"]
    return row


def _base_row(anchor_id: Any, cls: Any) -> dict[str, Any]:
    return {"anchor_id": anchor_id, "class": cls}


# ---------------------------------------------------------------------------
# S8 component 1 â€” explainer-repeat drift
# ---------------------------------------------------------------------------
def explainer_repeat_drift(
    explanations: Mapping[Any, Sequence[Any]],
    *,
    class_by_anchor: Mapping[Any, str] | None = None,
    metric_set: str = "feature",
    ks: Sequence[int] = md.DEFAULT_TOP_K,
    m: int = md.DEFAULT_EDGE_M,
    cfg: NoiseFloorConfig | None = None,
) -> pd.DataFrame:
    """Pairwise drift among ``R`` repeated explainer runs per anchor (S8.1).

    ``explanations`` maps ``anchor_id -> [explanation_0, ..., explanation_{R-1}]``
    (feature attribution vectors, or edge masks when ``metric_set='edge'``).
    Every unordered repeat pair contributes one row, so the pooled distribution
    has ``n_anchors * C(R, 2)`` entries. IG is deterministic (docs/05 Â§6), so the
    resulting drift must be exactly 0 â€” see :func:`ig_repeat_canary`.
    """
    config = cfg or NoiseFloorConfig()
    rows: list[dict[str, Any]] = []
    for anchor_id, runs in explanations.items():
        run_list = list(runs)
        if len(run_list) < 2:
            raise NoiseFloorError(
                f"anchor {anchor_id!r} has {len(run_list)} explainer repetitions; at least 2 are required"
            )
        for i in range(len(run_list)):
            for j in range(i + 1, len(run_list)):
                row = _base_row(anchor_id, (class_by_anchor or {}).get(anchor_id))
                row.update(
                    {
                        "channel": EXPLAINER_REPEAT,
                        "repeat_i": i,
                        "repeat_j": j,
                    }
                )
                row.update(_drift_columns(run_list[i], run_list[j], metric_set=metric_set, ks=ks, m=m))
                rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise NoiseFloorError("explainer_repeat_drift produced no rows")
    return frame


def ig_repeat_canary(
    explanations: Mapping[Any, Sequence[Any]],
    *,
    metric_set: str = "feature",
    tol: float = 0.0,
) -> dict[str, float]:
    """S8 check: repeated IG explanations of the same input must be identical.

    Returns the maximum absolute drift over every feature metric and every pair;
    the protocol requires exactly 0 (``tol=0``). Any non-zero value is a bug in
    the explainer path, not a finding.
    """
    frame = explainer_repeat_drift(explanations, metric_set=metric_set, cfg=NoiseFloorConfig())
    drift_columns = [column for column in frame.columns if column.startswith("drift_")]
    if not drift_columns:
        return {"max_abs_drift": float("nan"), "n_pairs": float(len(frame))}
    values = frame[drift_columns].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else float("nan")
    return {
        "max_abs_drift": max_abs,
        "n_pairs": float(len(frame)),
        "is_zero": float(max_abs <= tol) if np.isfinite(max_abs) else float("nan"),
    }


# ---------------------------------------------------------------------------
# S8 component 2 â€” null-update (model-noise) drift
# ---------------------------------------------------------------------------
def null_update_drift(
    reference: Mapping[Any, Any],
    reseeded: Mapping[Any, Mapping[Any, Any]],
    *,
    class_by_anchor: Mapping[Any, str] | None = None,
    metric_set: str = "feature",
    ks: Sequence[int] = md.DEFAULT_TOP_K,
    m: int = md.DEFAULT_EDGE_M,
    cfg: NoiseFloorConfig | None = None,
) -> pd.DataFrame:
    """Drift between the reference checkpoint and ``R`` same-period reseeds (S8.2).

    ``reference`` maps ``anchor_id -> explanation`` from the run's own checkpoint
    ``M_j^s``; ``reseeded`` maps ``anchor_id -> {null_seed: explanation}`` from
    the R null-update retrains (FT on period j, different seed, same explainer
    seed and stored subgraph). One row per (anchor, null seed).
    """
    config = cfg or NoiseFloorConfig()
    anchors = set(reference)
    reseeded_anchors = set(reseeded)
    if anchors != reseeded_anchors:
        raise NoiseFloorError(
            f"reference/reseeded anchor sets differ: {sorted(anchors - reseeded_anchors)[:5]} "
            f"vs {sorted(reseeded_anchors - anchors)[:5]}"
        )
    rows: list[dict[str, Any]] = []
    for anchor_id in sorted(reference, key=str):
        seeds = reseeded[anchor_id]
        if len(seeds) < 1:
            raise NoiseFloorError(f"anchor {anchor_id!r} has no null-update reseeds")
        if len(seeds) != config.n_null_reseeds:
            raise NoiseFloorError(
                f"anchor {anchor_id!r} has {len(seeds)} null-update reseeds; protocol requires "
                f"R = {config.n_null_reseeds} (docs/05 Â§9)"
            )
        for null_seed, vector in sorted(seeds.items(), key=lambda item: str(item[0])):
            row = _base_row(anchor_id, (class_by_anchor or {}).get(anchor_id))
            row.update({"channel": MODEL_NULL, "null_seed": null_seed})
            row.update(_drift_columns(reference[anchor_id], vector, metric_set=metric_set, ks=ks, m=m))
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise NoiseFloorError("null_update_drift produced no rows")
    return frame


# ---------------------------------------------------------------------------
# S8 component 3 â€” combined null (null update x explainer repeats)
# ---------------------------------------------------------------------------
def combined_null_drift(
    reference: Mapping[Any, Any],
    reseeded_repeats: Mapping[Any, Mapping[Any, Sequence[Any]]],
    *,
    class_by_anchor: Mapping[Any, str] | None = None,
    metric_set: str = "feature",
    ks: Sequence[int] = md.DEFAULT_TOP_K,
    m: int = md.DEFAULT_EDGE_M,
    cfg: NoiseFloorConfig | None = None,
) -> pd.DataFrame:
    """Combined null: null-update pairs with repeated explainer seeds (S8.3).

    ``reseeded_repeats`` maps ``anchor_id -> {null_seed: [explanation_repeat, ...]}``;
    every (null seed, repeat) pair against the reference explanation is one row,
    so this distribution mixes both noise sources (Doc 2b Â§S8 component 3).
    """
    config = cfg or NoiseFloorConfig()
    rows: list[dict[str, Any]] = []
    for anchor_id in sorted(reference, key=str):
        for null_seed in sorted(reseeded_repeats[anchor_id], key=str):
            repeats = list(reseeded_repeats[anchor_id][null_seed])
            if len(repeats) != config.n_explainer_repeats:
                raise NoiseFloorError(
                    f"anchor {anchor_id!r}/null_seed {null_seed!r} has {len(repeats)} explainer repeats; "
                    f"protocol requires R = {config.n_explainer_repeats} (docs/05 Â§9)"
                )
            for repeat_index, vector in enumerate(repeats):
                row = _base_row(anchor_id, (class_by_anchor or {}).get(anchor_id))
                row.update({"channel": COMBINED, "null_seed": null_seed, "repeat_index": repeat_index})
                row.update(_drift_columns(reference[anchor_id], vector, metric_set=metric_set, ks=ks, m=m))
                rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise NoiseFloorError("combined_null_drift produced no rows")
    return frame


# ---------------------------------------------------------------------------
# Per-class q95 thresholds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThresholdTable:
    """Noise-floor thresholds keyed by the protocol's pooling unit.

    The protocol pools the null distribution per **class and explainer**
    (Doc 2b Â§S8: "q95(anchor) from the pooled null distribution of its class and
    explainer"; docs/05 Â§9: "pooled per class"). ``group_columns`` records which
    columns form the key (default ``("explainer", "class")``) so a caller cannot
    accidentally reuse a feature-attribution floor for an edge-mask metric; the
    pooling fallback is recorded per key in ``source`` (ambiguity A7).
    """

    quantile: float
    channel: str
    value_column: str
    group_columns: tuple[str, ...]
    global_q95: float
    global_n: int
    q95: Mapping[tuple[str, ...], float]
    n: Mapping[tuple[str, ...], int]
    source: Mapping[tuple[str, ...], str]

    def lookup(self, key: Any) -> float:
        """q95 for a pooling key; unknown keys fall back to the global pool explicitly."""
        if not isinstance(key, tuple):
            key = (key,)
        normalised = tuple(str(value) for value in key)
        return float(self.q95.get(normalised, self.global_q95))

    def lookup_row(self, row: Mapping[str, Any]) -> float:
        """q95 for a row of a DataFrame with the threshold's group columns."""
        return self.lookup(tuple(row[column] for column in self.group_columns))

    def lookup_values(self, rows: Sequence[Sequence[Any]]) -> list[float]:
        """q95 for an iterable of key tuples in ``group_columns`` order."""
        return [self.lookup(tuple(value for value in row)) for row in rows]

    def source_for(self, key: Any) -> str:
        if not isinstance(key, tuple):
            key = (key,)
        return self.source.get(tuple(str(value) for value in key), "global")

    def as_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in sorted(self.q95):
            row: dict[str, Any] = dict(zip(self.group_columns, key))
            row.update(
                {
                    "channel": self.channel,
                    "n_null_pairs": int(self.n.get(key, 0)),
                    "quantile": float(self.quantile),
                    "q95": float(self.q95[key]),
                    "threshold_source": self.source.get(key, "global"),
                    "global_q95": float(self.global_q95),
                    "global_n": int(self.global_n),
                }
            )
            rows.append(row)
        return rows


def per_class_thresholds(
    noise_frame: pd.DataFrame,
    *,
    value_column: str = "drift_primary",
    class_column: str = "class",
    extra_group_columns: Sequence[str] = ("explainer",),
    cfg: NoiseFloorConfig | None = None,
    channel: str | None = None,
) -> ThresholdTable:
    """Pooled null q95 per (explainer, class) â€” the S8 per-anchor threshold.

    ``pooling='class'`` uses the group's own quantile when it has at least
    ``min_anchors_per_class`` noise pairs, otherwise (and for every group when
    ``pooling='global'``) the global pooled quantile; ``source`` records which
    was used (ambiguity A7). ``extra_group_columns=()`` gives a single pooled
    threshold (e.g. for a smoke test); the protocol default is class + explainer.
    """
    config = cfg or NoiseFloorConfig()
    if value_column not in noise_frame.columns:
        raise NoiseFloorError(f"{value_column!r} not in noise frame columns {sorted(noise_frame.columns)}")
    frame = noise_frame
    resolved_channel = channel or (str(frame["channel"].iloc[0]) if "channel" in frame.columns and len(frame) else config.primary_channel)
    group_columns = tuple(extra_group_columns) + (class_column,)
    missing_columns = [column for column in group_columns if column not in frame.columns]
    if missing_columns:
        raise NoiseFloorError(f"noise frame lacks threshold group column(s) {missing_columns}")
    values = frame[value_column].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise NoiseFloorError(f"no finite {value_column!r} values in the noise frame")
    global_q95 = float(np.percentile(finite, 100.0 * config.quantile))
    global_n = int(finite.size)
    q95: dict[tuple[str, ...], float] = {}
    n_by_key: dict[tuple[str, ...], int] = {}
    source: dict[tuple[str, ...], str] = {}
    if np.any(frame[list(group_columns)].isna().to_numpy()):
        raise NoiseFloorError(f"noise frame has anchors without a complete threshold key {group_columns}")
    for key, group in frame.groupby(list(group_columns), sort=True, dropna=False):
        key_tuple = tuple(str(value) for value in (key if isinstance(key, tuple) else (key,)))
        class_values = group[value_column].to_numpy(dtype=float)
        class_finite = class_values[np.isfinite(class_values)]
        n = int(class_finite.size)
        n_by_key[key_tuple] = n
        if config.pooling == "class" and n >= config.min_anchors_per_class and n > 0:
            q95[key_tuple] = float(np.percentile(class_finite, 100.0 * config.quantile))
            source[key_tuple] = "class"
        else:
            q95[key_tuple] = global_q95
            source[key_tuple] = "global"
    return ThresholdTable(
        quantile=config.quantile,
        channel=resolved_channel,
        value_column=value_column,
        group_columns=group_columns,
        global_q95=global_q95,
        global_n=global_n,
        q95=q95,
        n=n_by_key,
        source=source,
    )


# ---------------------------------------------------------------------------
# Excess drift (S9)
# ---------------------------------------------------------------------------
def compute_excess(
    real_frame: pd.DataFrame,
    noise_frame: pd.DataFrame,
    thresholds: ThresholdTable,
    *,
    join_keys: Sequence[str] = ("model", "explainer", "anchor_id"),
    value_column: str = "drift_primary",
    class_column: str = "class",
    cfg: NoiseFloorConfig | None = None,
) -> pd.DataFrame:
    """Attach null mean / q95 / excess / above-floor to every real-update row (S9).

    ``drift_null`` is the **mean null drift of the same anchor** (Doc 2b Â§S8), so
    the noise rows are aggregated per join key before joining. ``above_floor``
    compares the real drift to the per-class q95 (``floor_q95`` column records the
    exact threshold used). A real anchor with no null counterpart is an error:
    S8 checks require null and real updates to use the same anchors.
    """
    config = cfg or NoiseFloorConfig()
    if value_column not in real_frame.columns:
        raise NoiseFloorError(f"{value_column!r} not in real frame columns {sorted(real_frame.columns)}")
    for key in join_keys:
        if key not in real_frame.columns:
            raise NoiseFloorError(f"real frame lacks join key {key!r}")
        if key not in noise_frame.columns:
            raise NoiseFloorError(f"noise frame lacks join key {key!r}")
    if class_column not in real_frame.columns:
        raise NoiseFloorError(f"real frame lacks class column {class_column!r}")

    noise_values = noise_frame[value_column].to_numpy(dtype=float)
    if not np.any(np.isfinite(noise_values)):
        raise NoiseFloorError("noise frame has no finite drift values")
    aggregation = (
        noise_frame.assign(**{f"{value_column}_noise": noise_values})
        .groupby(list(join_keys), dropna=False)
        .agg(
            null_mean=(f"{value_column}_noise", "mean"),
            null_median=(f"{value_column}_noise", "median"),
            null_sd=(f"{value_column}_noise", "std"),
            n_null=(f"{value_column}_noise", "count"),
        )
        .reset_index()
    )
    merged = real_frame.merge(aggregation, on=list(join_keys), how="left", validate="many_to_one")
    missing = merged["null_mean"].isna()
    if missing.any():
        example = merged.loc[missing, list(join_keys)].head(3).to_dict("records")
        raise NoiseFloorError(
            f"{int(missing.sum())} real-update rows have no null-update counterpart for join keys {list(join_keys)}; "
            f"example(s): {example}"
        )
    threshold_columns = list(thresholds.group_columns)
    missing_threshold_columns = [column for column in threshold_columns if column not in merged.columns]
    if missing_threshold_columns:
        raise NoiseFloorError(
            f"threshold table pools by {threshold_columns} but the real frame lacks {missing_threshold_columns}"
        )
    key_rows = list(zip(*(merged[column].tolist() for column in threshold_columns))) if threshold_columns else []
    merged["floor_q95"] = thresholds.lookup_values(key_rows) if threshold_columns else thresholds.global_q95
    merged["threshold_source"] = [thresholds.source_for(key) for key in key_rows] if threshold_columns else "global"
    merged["excess"] = merged[value_column] - merged["null_mean"]
    merged["above_floor"] = merged[value_column] > merged["floor_q95"]
    merged["channel"] = thresholds.channel
    merged["quantile"] = thresholds.quantile
    merged["expected_above_fraction"] = EXPECTED_ABOVE_FRACTION
    return merged


# ---------------------------------------------------------------------------
# Aggregation with stratified bootstrap CIs (S9 tables)
# ---------------------------------------------------------------------------
def aggregate_drift(
    excess_frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("strategy", "model", "explainer", "class", "checkpoint_j", "checkpoint_t"),
    value_column: str = "drift_primary",
    class_column: str = "class",
    cfg: NoiseFloorConfig | None = None,
) -> pd.DataFrame:
    """Per-group raw/null/excess summary with stratified bootstrap CIs (S9).

    Groups are the protocol's aggregation unit (strategy x model x explainer x
    class x period pair). CIs resample anchors within class (docs/05 Â§13); when
    the group already contains a single class the stratification degenerates to
    that class, exactly as required. ``macro_across_classes`` rows are *not*
    produced here â€” macro aggregation over classes must be done explicitly with
    macro weights (Doc 2b Â§S8 "never average across classes without macro
    weighting"); use :func:`macro_average_classes`.
    """
    config = cfg or NoiseFloorConfig()
    for column in (*group_columns, value_column, class_column, "excess", "above_floor"):
        if column not in excess_frame.columns:
            raise NoiseFloorError(f"missing column {column!r} in excess frame")
    rows: list[dict[str, Any]] = []
    for key, group in excess_frame.groupby(list(group_columns), dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        row: dict[str, Any] = dict(zip(group_columns, key_values))
        values = group[value_column].to_numpy(dtype=float)
        excess = group["excess"].to_numpy(dtype=float)
        above = group["above_floor"].astype(float).to_numpy()
        strata = group[class_column].astype(str).to_numpy()
        real_ci = st.stratified_bootstrap_ci(values, strata, n_resamples=config.n_resamples, alpha=config.alpha, seed=config.seed)
        excess_ci = st.stratified_bootstrap_ci(excess, strata, n_resamples=config.n_resamples, alpha=config.alpha, seed=config.seed)
        fraction_ci = st.stratified_bootstrap_ci(above, strata, n_resamples=config.n_resamples, alpha=config.alpha, seed=config.seed)
        row.update(
            {
                "n_anchors": int(len(group)),
                "drift_real_mean": float(np.nanmean(values)),
                "drift_real_median": float(np.nanmedian(values)),
                "drift_real_ci_lo": real_ci.lo,
                "drift_real_ci_hi": real_ci.hi,
                "drift_null_mean": float(np.nanmean(group["null_mean"])),
                "drift_null_median": float(np.nanmedian(group["null_mean"])),
                "excess_mean": float(np.nanmean(excess)),
                "excess_ci_lo": excess_ci.lo,
                "excess_ci_hi": excess_ci.hi,
                "above_floor_fraction": fraction_ci.point,
                "above_floor_ci_lo": fraction_ci.lo,
                "above_floor_ci_hi": fraction_ci.hi,
                "expected_fraction": EXPECTED_ABOVE_FRACTION,
                "n_null_pairs": int(group["n_null"].sum()) if "n_null" in group.columns else int(len(group)),
                "bootstrap_resamples": config.n_resamples,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def macro_average_classes(
    aggregated: pd.DataFrame,
    *,
    class_column: str = "class",
    group_columns: Sequence[str] = ("strategy", "model", "explainer", "checkpoint_j", "checkpoint_t"),
    value_columns: Sequence[str] = ("drift_real_mean", "drift_null_mean", "excess_mean", "above_floor_fraction"),
) -> pd.DataFrame:
    """Macro-average class-level summaries (Doc 2b Â§S8: macro weighting required)."""
    for column in (*group_columns, class_column):
        if column not in aggregated.columns:
            raise NoiseFloorError(f"missing column {column!r} in aggregated frame")
    missing_values = [column for column in value_columns if column not in aggregated.columns]
    if missing_values:
        raise NoiseFloorError(f"missing value column(s) {missing_values}")
    rows: list[dict[str, Any]] = []
    for key, group in aggregated.groupby(list(group_columns), dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key_values))
        row["n_classes"] = int(group[class_column].nunique())
        for column in value_columns:
            row[column] = float(np.nanmean(group[column].to_numpy(dtype=float)))
        row["n_anchors_total"] = int(group.get("n_anchors", pd.Series(dtype=int)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H1 test and strategy family
# ---------------------------------------------------------------------------
def h1_test(
    excess_frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("strategy", "model", "explainer"),
    class_column: str = "class",
    cfg: NoiseFloorConfig | None = None,
    expected_fraction: float = EXPECTED_ABOVE_FRACTION,
    holm_within_family: bool = True,
) -> pd.DataFrame:
    """H1 per configuration: fraction above the null q95 vs 5 % (docs/05 Â§9).

    Reports for each (strategy, model, explainer): n anchors, n above floor,
    fraction with stratified bootstrap CI, exact binomial p, normal-approximation
    p, paired Wilcoxon signed-rank p on excess vs 0, rank-biserial effect size,
    mean/median excess, and Holm-adjusted p-values within the family of
    configurations (Doc 2b Â§S9 / docs/05 Â§13).
    """
    config = cfg or NoiseFloorConfig()
    for column in (*group_columns, "above_floor", "excess", class_column):
        if column not in excess_frame.columns:
            raise NoiseFloorError(f"missing column {column!r} in excess frame")
    rows: list[dict[str, Any]] = []
    for key, group in excess_frame.groupby(list(group_columns), dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        above = group["above_floor"].astype(float).to_numpy()
        excess = group["excess"].to_numpy(dtype=float)
        strata = group[class_column].astype(str).to_numpy()
        fraction_ci = st.stratified_bootstrap_ci(above, strata, n_resamples=config.n_resamples, alpha=config.alpha, seed=config.seed)
        successes = int(np.sum(above == 1.0))
        binomial = st.binomial_test_proportion(successes, int(above.size), p0=expected_fraction, alternative="greater")
        wilcoxon = st.wilcoxon_signed_rank(excess, np.zeros_like(excess))
        row: dict[str, Any] = dict(zip(group_columns, key_values))
        row.update(
            {
                "n_anchors": int(above.size),
                "n_above_floor": successes,
                "fraction_above_floor": float(fraction_ci.point),
                "fraction_ci_lo": fraction_ci.lo,
                "fraction_ci_hi": fraction_ci.hi,
                "expected_fraction": float(expected_fraction),
                "p_binomial": binomial["p_binomial"],
                "p_normal": binomial["p_normal"],
                "z_normal": binomial["z_normal"],
                "excess_mean": float(np.nanmean(excess)) if excess.size else float("nan"),
                "excess_median": float(np.nanmedian(excess)) if excess.size else float("nan"),
                "p_wilcoxon": wilcoxon["p_value"],
                "wilcoxon_statistic": wilcoxon["statistic"],
                "rank_biserial": st.rank_biserial_from_wilcoxon(excess, np.zeros_like(excess)),
                "bootstrap_resamples": config.n_resamples,
                "expected_by_construction": True,
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if holm_within_family and not frame.empty:
        frame["p_binomial_holm"] = st.holm_correction(frame["p_binomial"].to_numpy(dtype=float))
        frame["p_wilcoxon_holm"] = st.holm_correction(frame["p_wilcoxon"].to_numpy(dtype=float))
        frame["holm_family"] = "|".join(str(column) for column in group_columns)
    return frame


def strategy_comparison(
    excess_frame: pd.DataFrame,
    *,
    baseline: str = "ft",
    comparisons: Sequence[str] = ("replay", "ewc"),
    strategy_column: str = "strategy",
    value_columns: Sequence[str] = ("drift_primary", "excess"),
    pair_keys: Sequence[str] = ("model", "explainer", "anchor_id", "checkpoint_j", "checkpoint_t"),
    class_column: str = "class",
    cfg: NoiseFloorConfig | None = None,
) -> pd.DataFrame:
    """Paired strategy differences vs the fine-tuning baseline (claim C2, S9).

    Anchors are paired by ``pair_keys`` (plus ``seed`` when present); the
    difference is ``strategy - baseline`` and is reported with a paired bootstrap
    CI, bootstrap p-value and effect sizes. The direction is *not* assumed:
    negative drift differences are reported honestly (C2 wording).
    """
    config = cfg or NoiseFloorConfig()
    if strategy_column not in excess_frame.columns:
        raise NoiseFloorError(f"excess frame lacks {strategy_column!r}")
    present_strategies = set(excess_frame[strategy_column].astype(str).unique())
    empty_columns = [
        "comparison", "baseline", "metric", "n_pairs", "mean_diff",
        "ci_lo", "ci_hi", "p_value", "cohens_d", "rank_biserial",
        "bootstrap_resamples", "p_holm",
    ]
    if baseline not in present_strategies:
        if config.allow_partial_strategies:
            return pd.DataFrame(columns=empty_columns)
        raise NoiseFloorError(f"baseline strategy {baseline!r} not found in excess frame")
    keys = list(pair_keys)
    if "seed" in excess_frame.columns and "seed" not in keys:
        keys.append("seed")
    if class_column not in keys:
        keys.append(class_column)  # keep a single merged class column for stratification
    for key in (*keys, class_column):
        if key not in excess_frame.columns:
            raise NoiseFloorError(f"excess frame lacks pairing key {key!r}")
    baseline_frame = excess_frame[excess_frame[strategy_column].astype(str) == baseline]
    if baseline_frame.empty:
        raise NoiseFloorError(f"baseline strategy {baseline!r} not found in excess frame")
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        comparison_frame = excess_frame[excess_frame[strategy_column].astype(str) == comparison]
        if comparison_frame.empty:
            if config.allow_partial_strategies:
                continue
            raise NoiseFloorError(f"comparison strategy {comparison!r} not found in excess frame")
        merged = comparison_frame.merge(
            baseline_frame,
            on=keys,
            suffixes=("_cmp", "_base"),
            how="inner",
            validate="one_to_one",
        )
        if merged.empty:
            raise NoiseFloorError(f"no paired anchors between {comparison!r} and {baseline!r} on keys {keys}")
        for value_column in value_columns:
            column_cmp = f"{value_column}_cmp"
            column_base = f"{value_column}_base"
            if column_cmp not in merged.columns or column_base not in merged.columns:
                raise NoiseFloorError(f"cannot pair metric {value_column!r} (merge columns: {sorted(merged.columns)})")
            a = merged[column_cmp].to_numpy(dtype=float)
            b = merged[column_base].to_numpy(dtype=float)
            paired = st.paired_bootstrap_diff(
                a,
                b,
                n_resamples=config.n_resamples,
                alpha=config.alpha,
                seed=config.seed,
                stratum=merged[class_column].astype(str).to_numpy(),
            )
            effect = st.effect_size_summary(a, b, paired=True)
            rows.append(
                {
                    "comparison": comparison,
                    "baseline": baseline,
                    "metric": value_column,
                    "n_pairs": int(merged.shape[0]),
                    "mean_diff": paired.diff,
                    "ci_lo": paired.lo,
                    "ci_hi": paired.hi,
                    "p_value": paired.p_value,
                    "cohens_d": effect["cohens_d"],
                    "rank_biserial": effect["rank_biserial"],
                    "bootstrap_resamples": config.n_resamples,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=empty_columns)
    frame["p_holm"] = st.holm_correction(frame["p_value"].to_numpy(dtype=float))
    return frame


# ---------------------------------------------------------------------------
# Table rows
# ---------------------------------------------------------------------------
def noise_floor_rows(
    noise_frame: pd.DataFrame,
    *,
    thresholds: Mapping[str, ThresholdTable] | None = None,
    value_column: str = "drift_primary",
    class_column: str = "class",
    config: NoiseFloorConfig | None = None,
) -> pd.DataFrame:
    """Rows behind ``results/text/tables/noise_floor.csv`` (S8 output).

    One row per (channel, model, explainer, class): n null pairs, mean/median
    drift, q95, threshold source and the frozen protocol constants. Pass the
    per-channel :class:`ThresholdTable` mapping so the reported q95 is exactly
    the one used downstream (no recomputation drift).
    """
    cfg = config or NoiseFloorConfig()
    required = {value_column, "channel"}
    missing = required - set(noise_frame.columns)
    if missing:
        raise NoiseFloorError(f"noise frame lacks columns {sorted(missing)}")
    group_columns = [column for column in ("channel", "model", "explainer", class_column) if column in noise_frame.columns]
    if class_column not in group_columns:
        raise NoiseFloorError("noise_floor_rows requires a class column (per-class floors, docs/05 Â§9)")
    rows: list[dict[str, Any]] = []
    for key, group in noise_frame.groupby(group_columns, dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        info = dict(zip(group_columns, key_values))
        values = group[value_column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        channel = str(info["channel"])
        class_name = str(info[class_column])
        table = (thresholds or {}).get(channel)
        if table is not None and all(column in info for column in table.group_columns):
            threshold_key = tuple(info[column] for column in table.group_columns)
            q95 = table.lookup(threshold_key)
            source = table.source_for(threshold_key)
            global_q95 = table.global_q95
            threshold_n = table.n.get(tuple(str(value) for value in threshold_key), int(finite.size))
        else:
            q95 = float(np.percentile(finite, 100.0 * cfg.quantile))
            source = "class" if cfg.pooling == "class" else "global"
            global_q95 = q95
            threshold_n = int(finite.size)
        row = {
            "channel": channel,
            "model": info.get("model", ""),
            "explainer": info.get("explainer", ""),
            "class": class_name,
            "n_null_pairs": int(finite.size),
            "quantile": float(cfg.quantile),
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "q95": float(q95),
            "threshold_source": source,
            "threshold_n": int(threshold_n),
            "global_q95": float(global_q95),
        }
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "CHANNELS",
    "COMBINED",
    "EXPECTED_ABOVE_FRACTION",
    "EXPLAINER_REPEAT",
    "MODEL_NULL",
    "NoiseFloorConfig",
    "NoiseFloorError",
    "PRIMARY_CHANNEL",
    "ThresholdTable",
    "aggregate_drift",
    "combined_null_drift",
    "compute_excess",
    "explainer_repeat_drift",
    "h1_test",
    "ig_repeat_canary",
    "macro_average_classes",
    "noise_floor_rows",
    "null_update_drift",
    "per_class_thresholds",
    "strategy_comparison",
]
