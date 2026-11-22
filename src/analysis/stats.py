"""Statistics primitives for every analysis (docs/05 §13; Doc 2b §S9–§S13).

Authority
---------
* ``docs/05_EXPERIMENT_PROTOCOL.md`` §13 "Statistics (all analyses)":
  bootstrap with **1000 resamples**, **stratified by class/period**, **percentile
  95 % CIs**; paired comparisons use a **paired bootstrap**; **Holm correction
  within each RQ's family** (report unadjusted and adjusted); **effect sizes
  required for every comparison**; no "significant" claim without CI and effect
  size; missing/rare classes excluded explicitly.
* ``docs/05`` §9 (H1: bootstrap CI on the above-floor fraction, binomial test vs
  5 %, paired Wilcoxon on excess) and §11 (RQ3: L2 logistic regression,
  leave-one-period-out CV, AUC with bootstrap CI, ΔAUC with CI).
* ``docs/source_documents/Doc 2b`` §S10 (Spearman + partial Spearman + permutation test vs a
  shuffled-change null) and §S11 (logistic AUC, baselines, effect sizes).
* ``docs/09_DATA_AUDIT_AND_PREPROCESSING.md`` §8 / D-030: the **hierarchical**
  bootstrap for the pilot — 500 unstratified anchor resamples per seed → 1000
  seed-level resamples of per-seed means — is implemented here as a documented
  deviation that applies to the pilot only (main analyses use the stratified
  1000-resample percentile bootstrap above).

Implementation conventions
--------------------------
* Randomness is injected through ``numpy.random.default_rng(seed)`` and every
  result records the seed, so a CI is reproducible bit-for-bit.
* Percentile intervals use ``numpy.percentile``'s default ``linear`` method on
  the bootstrap distribution (the protocol does not name an estimator; this is
  convention A5 reported to the Lead and applied everywhere, never mixed).
* The paired-bootstrap p-value is the two-sided bootstrap sign test
  ``2 * min(P(diff <= 0), P(diff >= 0))`` with the +1 correction; it is a
  *bootstrap* p-value, not a model-based test, and is always reported next to
  the CI and the effect size.
* Degenerate inputs (n = 1, single class in a fold, all-zero differences) return
  NaN/inf with an explicit note instead of crashing a whole analysis table; the
  guard conditions are documented per function.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as _scipy_stats

#: Protocol default number of bootstrap resamples (docs/05 §13).
DEFAULT_N_RESAMPLES: int = 1000

#: Protocol default two-sided alpha (docs/05 §13: 95 % percentile CIs).
DEFAULT_ALPHA: float = 0.05

#: Default RNG seed for analysis-side resampling.
DEFAULT_SEED: int = 42

#: Pilot hierarchical-bootstrap defaults (docs/09 §8 / D-030).
PILOT_N_INNER: int = 500
PILOT_N_OUTER: int = 1000


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BootstrapResult:
    """Percentile bootstrap interval for one statistic."""

    point: float
    lo: float
    hi: float
    n: int
    n_resamples: int
    alpha: float
    seed: int
    statistic: str = "mean"
    method: str = "percentile"

    def as_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "ci_lo": self.lo,
            "ci_hi": self.hi,
            "n": self.n,
            "n_resamples": self.n_resamples,
            "alpha": self.alpha,
            "seed": self.seed,
            "statistic": self.statistic,
            "ci_method": self.method,
        }


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Paired-bootstrap difference ``a - b`` with percentile CI and bootstrap p."""

    diff: float
    lo: float
    hi: float
    p_value: float
    n: int
    n_resamples: int
    alpha: float
    seed: int
    method: str = "paired_percentile"

    def as_dict(self) -> dict[str, Any]:
        return {
            "diff": self.diff,
            "ci_lo": self.lo,
            "ci_hi": self.hi,
            "p_value": self.p_value,
            "n": self.n,
            "n_resamples": self.n_resamples,
            "alpha": self.alpha,
            "seed": self.seed,
            "method": self.method,
        }


@dataclass(frozen=True)
class PermutationResult:
    """Permutation test of an association statistic (Doc 2b §S10)."""

    observed: float
    p_value: float
    alternative: str
    n_permutations: int
    seed: int
    n: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed": self.observed,
            "p_value": self.p_value,
            "alternative": self.alternative,
            "n_permutations": self.n_permutations,
            "seed": self.seed,
            "n": self.n,
        }


@dataclass(frozen=True)
class AucResult:
    """Leave-one-period-out logistic-regression AUC (docs/05 §11)."""

    auc: float
    oof_scores: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    groups: np.ndarray = field(repr=False)
    per_fold_auc: Mapping[Any, float] = field(default_factory=dict)
    n: int = 0
    n_pos: int = 0
    n_neg: int = 0
    n_folds: int = 0
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "auc": self.auc,
            "n": self.n,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "n_folds": self.n_folds,
            "per_fold_auc": dict(self.per_fold_auc),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class AucDeltaResult:
    """Paired bootstrap of ``AUC(model_a) - AUC(model_b)`` on the same units."""

    auc_a: float
    auc_b: float
    delta: float
    lo: float
    hi: float
    p_value: float
    n: int
    n_resamples: int
    alpha: float
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "auc_a": self.auc_a,
            "auc_b": self.auc_b,
            "delta": self.delta,
            "ci_lo": self.lo,
            "ci_hi": self.hi,
            "p_value": self.p_value,
            "n": self.n,
            "n_resamples": self.n_resamples,
            "alpha": self.alpha,
            "seed": self.seed,
        }


class StatisticsError(ValueError):
    """Raised when a statistics input violates the protocol contract (e.g. wrong shapes)."""


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------
def _as_1d(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array


def _check_same_length(a: np.ndarray, b: np.ndarray, *, name_a: str = "a", name_b: str = "b") -> None:
    if a.shape != b.shape:
        raise StatisticsError(f"{name_a} and {name_b} must have the same length (got {a.size} vs {b.size})")


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Row-wise finite mask shared by 1-D vectors and 2-D control matrices."""
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for array in arrays:
        if array.ndim == 1:
            mask &= np.isfinite(array)
        else:
            mask &= np.all(np.isfinite(array), axis=tuple(range(1, array.ndim)))
    return mask


def _resolve_statistic(statistic: str | Callable[[np.ndarray], float], values: np.ndarray) -> tuple[Callable[[np.ndarray], float], str]:
    if callable(statistic):
        return statistic, getattr(statistic, "__name__", "callable")
    if statistic == "mean":
        return (lambda sample: float(np.mean(sample))), "mean"
    if statistic == "median":
        return (lambda sample: float(np.median(sample))), "median"
    raise ValueError(f"unknown statistic {statistic!r}; use 'mean', 'median' or a callable")


def _percentile_interval(distribution: np.ndarray, alpha: float) -> tuple[float, float]:
    finite = distribution[np.isfinite(distribution)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lower = float(np.percentile(finite, 100.0 * alpha / 2.0))
    upper = float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


# ---------------------------------------------------------------------------
# Vectorised row statistics (bootstrap performance; numerically equivalent)
# ---------------------------------------------------------------------------
def _apply_statistic_rows(func: Callable[[np.ndarray], float], samples: np.ndarray, name: str) -> np.ndarray:
    """Apply a sample statistic to every row of ``samples`` (fast paths for mean/median)."""
    if name == "mean":
        return np.mean(samples, axis=1)
    if name == "median":
        return np.median(samples, axis=1)
    return np.apply_along_axis(func, 1, samples)


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of two ``(B, n)`` matrices (NaN when degenerate)."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    numerator = np.sum(ac * bc, axis=1)
    denominator = np.sqrt(np.sum(ac * ac, axis=1) * np.sum(bc * bc, axis=1))
    result = np.full(numerator.shape, np.nan, dtype=float)
    valid = denominator > 0.0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _rank_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-wise average ranks (scipy ``rankdata(..., axis=1)``)."""
    return _scipy_stats.rankdata(matrix, axis=1)


def _auc_rows(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Row-wise AUC via the Mann-Whitney U identity (ties handled by average ranks).

    ``scores`` is ``(B, n)`` and ``y`` is the fixed ``(n,)`` binary outcome (a
    label-stratified bootstrap keeps the label composition fixed, so the same
    ``y`` vector applies to every bootstrap row). Equivalent to
    ``roc_auc_score`` for every row.
    """
    labels = np.asarray(y).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[0], np.nan)
    ranks = _rank_rows(np.atleast_2d(scores))
    positive_rank_sum = ranks[:, labels].sum(axis=1)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _resample_rows(indices: Sequence[np.ndarray], n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Row matrix ``(B, n)`` of resampled global positions, stratified by group."""
    parts = [group_indices[rng.integers(0, group_indices.size, size=(n_resamples, group_indices.size))] for group_indices in indices]
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap_ci(
    values: Any,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    statistic: str | Callable[[np.ndarray], float] = "mean",
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    dropna: bool = False,
) -> BootstrapResult:
    """Percentile bootstrap CI of one sample statistic (docs/05 §13).

    Use the stratified variant (:func:`stratified_bootstrap_ci`) whenever the
    sample is stratified by class/period, as the protocol requires; this
    function is the unstratified primitive (and the pilot's inner loop).
    """
    array = _as_1d(values, "values")
    if dropna:
        array = array[np.isfinite(array)]
    elif not np.all(np.isfinite(array)):
        raise StatisticsError("bootstrap_ci requires finite values; pass dropna=True to drop NaN/inf")
    if array.size == 0:
        raise StatisticsError("bootstrap_ci requires at least one value")
    func, name = _resolve_statistic(statistic, array)
    point = float(func(array))
    if array.size == 1:
        return BootstrapResult(point, point, point, 1, n_resamples, alpha, seed, name)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_resamples, array.size))
    samples = array[indices]
    distribution = _apply_statistic_rows(func, samples, name)
    lo, hi = _percentile_interval(distribution, alpha)
    return BootstrapResult(point, lo, hi, int(array.size), n_resamples, alpha, seed, name)


def stratify_indices(strata: Sequence[Any]) -> list[np.ndarray]:
    """Index groups for a stratification vector, in a deterministic order."""
    array = np.asarray(list(strata))
    if array.ndim != 1:
        raise StatisticsError("strata must be 1-D")
    unique = list(dict.fromkeys(array.tolist()))
    return [np.where(array == value)[0] for value in unique]


def stratified_bootstrap_ci(
    values: Any,
    strata: Sequence[Any],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    statistic: str | Callable[[np.ndarray], float] = "mean",
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    dropna: bool = False,
) -> BootstrapResult:
    """Stratified percentile bootstrap CI (docs/05 §13: resample within stratum).

    Each resample keeps every stratum's original size, so class/period
    composition is held fixed while anchors are resampled with replacement —
    exactly the protocol's "resampled within class" unit.
    """
    array = _as_1d(values, "values")
    if len(strata) != array.size:
        raise StatisticsError(f"strata length {len(strata)} != values length {array.size}")
    strata_array = np.asarray(list(strata))
    if dropna:
        mask = np.isfinite(array)
        array = array[mask]
        strata_array = strata_array[mask]
    elif not np.all(np.isfinite(array)):
        raise StatisticsError("stratified_bootstrap_ci requires finite values; pass dropna=True to drop NaN/inf")
    if array.size == 0:
        raise StatisticsError("stratified_bootstrap_ci requires at least one value")
    func, name = _resolve_statistic(statistic, array)
    point = float(func(array))
    groups = stratify_indices(strata_array)
    if array.size == 1:
        return BootstrapResult(point, point, point, 1, n_resamples, alpha, seed, name)
    rng = np.random.default_rng(seed)
    resample_positions = _resample_rows(groups, n_resamples, rng)
    samples = array[resample_positions]
    distribution = _apply_statistic_rows(func, samples, name)
    lo, hi = _percentile_interval(distribution, alpha)
    return BootstrapResult(point, lo, hi, int(array.size), n_resamples, alpha, seed, name)


def hierarchical_bootstrap_ci(
    per_seed_values: Sequence[Any],
    *,
    n_inner: int = PILOT_N_INNER,
    n_outer: int = PILOT_N_OUTER,
    statistic: str | Callable[[np.ndarray], float] = "mean",
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> BootstrapResult:
    """Hierarchical bootstrap for the graph-lift pilot (docs/09 §8, D-030).

    Exact documented procedure (must be reproduced exactly for the pilot):

    1. for each seed ``s`` (independent run/arm), draw ``n_inner`` unstratified
       anchor resamples and record the statistic of each (default mean macro-F1);
    2. draw ``n_outer`` seed resamples with replacement; for each, average one
       drawn inner replicate per drawn seed;
    3. report the percentile interval of the resulting outer distribution.

    This is the *only* analysis allowed to use an unstratified inner loop and
    the only place where the 500/1000 split replaces the standard 1000-resample
    stratified bootstrap; it applies to the pilot only (D-030).
    """
    groups = [np.asarray(values, dtype=float).reshape(-1) for values in per_seed_values]
    groups = [group[np.isfinite(group)] for group in groups]
    groups = [group for group in groups if group.size > 0]
    if not groups:
        raise StatisticsError("hierarchical_bootstrap_ci requires at least one non-empty seed sample")
    func, name = _resolve_statistic(statistic, groups[0])
    point = float(np.mean([func(group) for group in groups]))
    if len(groups) == 1 and groups[0].size == 1:
        return BootstrapResult(point, point, point, 1, n_outer, alpha, seed, name)
    rng = np.random.default_rng(seed)
    n_seeds = len(groups)
    inner = np.empty((n_seeds, n_inner), dtype=float)
    for index, group in enumerate(groups):
        if group.size == 1:
            inner[index, :] = group[0]
            continue
        draw = rng.integers(0, group.size, size=(n_inner, group.size))
        inner[index, :] = np.apply_along_axis(func, 1, group[draw])
    outer = np.empty(n_outer, dtype=float)
    for b in range(n_outer):
        seed_draw = rng.integers(0, n_seeds, size=n_seeds)
        inner_draw = rng.integers(0, n_inner, size=n_seeds)
        outer[b] = float(np.mean(inner[seed_draw, inner_draw]))
    lo, hi = _percentile_interval(outer, alpha)
    return BootstrapResult(point, lo, hi, int(sum(group.size for group in groups)), n_outer, alpha, seed, name)


def paired_bootstrap_diff(
    a: Any,
    b: Any,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    stratum: Sequence[Any] | None = None,
) -> PairedBootstrapResult:
    """Paired bootstrap of ``mean(a - b)`` with percentile CI and sign-test p.

    ``stratum`` optionally stratifies the resampling (docs/05 §13); paired
    comparisons always resample the *pair* as one unit (same anchor/class).
    """
    array_a = _as_1d(a, "a")
    array_b = _as_1d(b, "b")
    _check_same_length(array_a, array_b)
    mask = _finite_mask(array_a, array_b)
    array_a = array_a[mask]
    array_b = array_b[mask]
    if array_a.size == 0:
        raise StatisticsError("paired_bootstrap_diff requires at least one finite pair")
    differences = array_a - array_b
    point = float(np.mean(differences))
    if differences.size == 1:
        return PairedBootstrapResult(point, point, point, 1.0, 1, n_resamples, alpha, seed)
    strata_array = None if stratum is None else np.asarray(list(stratum))[mask]
    rng = np.random.default_rng(seed)
    if strata_array is None:
        groups = [np.arange(differences.size)]
    else:
        groups = stratify_indices(strata_array)
    resample_positions = _resample_rows(groups, n_resamples, rng)
    distribution = differences[resample_positions].mean(axis=1)
    lo, hi = _percentile_interval(distribution, alpha)
    return PairedBootstrapResult(point, lo, hi, _bootstrap_sign_p(distribution), int(differences.size), n_resamples, alpha, seed)


def _bootstrap_sign_p(distribution: np.ndarray) -> float:
    """Two-sided bootstrap sign-test p-value with the +1 correction."""
    finite = distribution[np.isfinite(distribution)]
    if finite.size == 0:
        return float("nan")
    count_le = int(np.sum(finite <= 0.0))
    count_ge = int(np.sum(finite >= 0.0))
    p_le = (1 + count_le) / (1 + finite.size)
    p_ge = (1 + count_ge) / (1 + finite.size)
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


# ---------------------------------------------------------------------------
# Correlation: Spearman, partial Spearman, permutation test
# ---------------------------------------------------------------------------
def spearman_correlation(x: Any, y: Any, *, dropna: bool = True) -> float:
    """Spearman rank correlation (Doc 2b §S10); NaN when undefined (constant input)."""
    array_x = _as_1d(x, "x")
    array_y = _as_1d(y, "y")
    _check_same_length(array_x, array_y, name_a="x", name_b="y")
    mask = _finite_mask(array_x, array_y)
    xs, ys = array_x[mask], array_y[mask]
    if not dropna and mask.sum() != array_x.size:
        raise StatisticsError("spearman_correlation: NaN/inf present; pass dropna=True (default) to drop pairs")
    if xs.size < 3 or np.ptp(xs) == 0.0 or np.ptp(ys) == 0.0:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(_scipy_stats.spearmanr(xs, ys).statistic)


def spearman_correlation_ci(
    x: Any,
    y: Any,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> BootstrapResult:
    """Bootstrap CI of the Spearman correlation (resample pairs, docs/05 §13)."""
    array_x = _as_1d(x, "x")
    array_y = _as_1d(y, "y")
    _check_same_length(array_x, array_y, name_a="x", name_b="y")
    mask = _finite_mask(array_x, array_y)
    xs, ys = array_x[mask], array_y[mask]
    if xs.size < 3:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), int(xs.size), n_resamples, alpha, seed, "spearman")
    point = spearman_correlation(xs, ys)
    rng = np.random.default_rng(seed)
    positions = rng.integers(0, xs.size, size=(n_resamples, xs.size))
    rx = _rank_rows(xs[positions])
    ry = _rank_rows(ys[positions])
    distribution = _pearson_rows(rx, ry)
    lo, hi = _percentile_interval(distribution, alpha)
    return BootstrapResult(point, lo, hi, int(xs.size), n_resamples, alpha, seed, "spearman")


def partial_spearman(x: Any, y: Any, controls: Any, *, dropna: bool = True) -> float:
    """Partial Spearman: rank-transform, residualise on controls, correlate.

    Doc 2b §S10 requires partial Spearman controlling for anchor class, anchor
    degree and period pair; the implementation is the standard "partial
    correlation of ranks" (rank transform every variable, then partial Pearson
    correlation of x and y given the controls).
    """
    array_x = _as_1d(x, "x")
    array_y = _as_1d(y, "y")
    control_matrix = _as_controls(controls)
    if control_matrix.size == 0:
        control_matrix = np.empty((array_x.size, 0), dtype=float)
    if control_matrix.shape[0] != array_x.size:
        raise StatisticsError(
            f"controls have {control_matrix.shape[0]} rows but x has {array_x.size} entries"
        )
    mask = _finite_mask(array_x, array_y, control_matrix)
    if not dropna and not np.all(mask):
        raise StatisticsError("partial_spearman: NaN/inf present; pass dropna=True (default) to drop rows")
    xs, ys = array_x[mask], array_y[mask]
    controls_kept = control_matrix[mask]
    if xs.size < 4:
        return float("nan")
    if np.ptp(xs) == 0.0 or np.ptp(ys) == 0.0:
        return float("nan")
    if controls_kept.size == 0:
        return float(_scipy_stats.spearmanr(xs, ys).statistic)
    rank_x = _scipy_stats.rankdata(xs)
    rank_y = _scipy_stats.rankdata(ys)
    if controls_kept.shape[1] >= xs.size - 1:
        return float("nan")
    rank_controls = np.column_stack([_scipy_stats.rankdata(controls_kept[:, column]) for column in range(controls_kept.shape[1])])
    resid_x = _residualise(rank_x, rank_controls)
    resid_y = _residualise(rank_y, rank_controls)
    if np.ptp(resid_x) == 0.0 or np.ptp(resid_y) == 0.0:
        return float("nan")
    return float(np.corrcoef(resid_x, resid_y)[0, 1])


def partial_spearman_ci(
    x: Any,
    y: Any,
    controls: Any,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> BootstrapResult:
    """Bootstrap CI of the partial Spearman correlation (resample rows)."""
    array_x = _as_1d(x, "x")
    array_y = _as_1d(y, "y")
    control_matrix = _as_controls(controls)
    if control_matrix.size == 0:
        control_matrix = np.empty((array_x.size, 0), dtype=float)
    mask = _finite_mask(array_x, array_y, control_matrix)
    xs, ys = array_x[mask], array_y[mask]
    controls_kept = control_matrix[mask]
    if xs.size < 4:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), int(xs.size), n_resamples, alpha, seed, "partial_spearman")
    point = partial_spearman(xs, ys, controls_kept, dropna=False)
    rng = np.random.default_rng(seed)
    positions = rng.integers(0, xs.size, size=(n_resamples, xs.size))
    rx = _rank_rows(xs[positions])
    ry = _rank_rows(ys[positions])
    if controls_kept.size == 0:
        distribution = _pearson_rows(rx, ry)
    else:
        n_controls = controls_kept.shape[1]
        ranked_controls = np.stack(
            [_rank_rows(controls_kept[:, column][positions]) for column in range(n_controls)], axis=2
        )
        design = np.concatenate([np.ones((n_resamples, xs.size, 1)), ranked_controls], axis=2)
        pseudo_inverse = np.linalg.pinv(design)  # (B, k+1, n)
        beta_x = np.einsum("bij,bj->bi", pseudo_inverse, rx)
        beta_y = np.einsum("bij,bj->bi", pseudo_inverse, ry)
        residual_x = rx - np.einsum("bij,bj->bi", design, beta_x)
        residual_y = ry - np.einsum("bij,bj->bi", design, beta_y)
        distribution = _pearson_rows(residual_x, residual_y)
    lo, hi = _percentile_interval(distribution, alpha)
    return BootstrapResult(point, lo, hi, int(xs.size), n_resamples, alpha, seed, "partial_spearman")


def _as_controls(controls: Any) -> np.ndarray:
    if controls is None:
        return np.empty((0, 0), dtype=float)
    if isinstance(controls, np.ndarray):
        array = np.asarray(controls, dtype=float)
    else:  # DataFrame or nested sequence
        try:
            array = np.asarray(controls, dtype=float)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise StatisticsError(f"controls must be numeric: {exc}") from exc
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise StatisticsError(f"controls must be 2-D (n, k), got shape {array.shape}")
    return array


def _residualise(target: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(controls.shape[0]), controls])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ coefficients


def permutation_test(
    x: Any,
    y: Any,
    *,
    statistic: str | Callable[[np.ndarray, np.ndarray], float] | None = None,
    alternative: str = "greater",
    n_permutations: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> PermutationResult:
    """Permutation test against a shuffled-change null (Doc 2b §S10).

    ``statistic`` takes ``(x, y)`` and defaults to :func:`spearman_correlation`.
    ``alternative='greater'`` is the H2 direction (positive association between
    structural change and drift); ``'less'`` and ``'two-sided'`` are supported.
    """
    if statistic is None:
        statistic_func: Callable[[np.ndarray, np.ndarray], float] = spearman_correlation
        name = "spearman"
    elif callable(statistic):
        statistic_func = statistic
        name = getattr(statistic, "__name__", "callable")
    else:
        raise ValueError(f"unknown statistic {statistic!r}")
    array_x = _as_1d(x, "x")
    array_y = _as_1d(y, "y")
    _check_same_length(array_x, array_y, name_a="x", name_b="y")
    mask = _finite_mask(array_x, array_y)
    xs, ys = array_x[mask], array_y[mask]
    if xs.size < 3:
        return PermutationResult(float("nan"), float("nan"), alternative, n_permutations, seed, int(xs.size))
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError(f"alternative must be 'greater', 'less' or 'two-sided', got {alternative!r}")
    observed = float(statistic_func(xs, ys))
    if np.isnan(observed):
        return PermutationResult(float("nan"), float("nan"), alternative, n_permutations, seed, int(xs.size))
    rng = np.random.default_rng(seed)
    if statistic is None:
        # fast path: rank correlation under label permutation == Spearman permutation
        rank_x = _scipy_stats.rankdata(xs)
        permutation_index = np.argsort(rng.random((n_permutations, ys.size)), axis=1)
        permuted_y = ys[permutation_index]
        null_values = _pearson_rows(np.broadcast_to(rank_x, permuted_y.shape), _rank_rows(permuted_y))
        null_values = null_values[np.isfinite(null_values)]
    else:  # custom statistic: honest loop (rare path)
        null_values = np.empty(n_permutations, dtype=float)
        for index in range(n_permutations):
            shuffled = ys[rng.permutation(ys.size)]
            null_values[index] = float(statistic_func(xs, shuffled))
        null_values = null_values[np.isfinite(null_values)]
    if alternative == "greater":
        extreme = int(np.sum(null_values >= observed))
    elif alternative == "less":
        extreme = int(np.sum(null_values <= observed))
    else:
        extreme = int(np.sum(np.abs(null_values) >= abs(observed)))
    p_value = (1 + extreme) / (1 + n_permutations)
    return PermutationResult(observed, float(p_value), alternative, n_permutations, seed, int(xs.size))


# ---------------------------------------------------------------------------
# Multiple comparisons and effect sizes
# ---------------------------------------------------------------------------
def holm_correction(pvalues: Any) -> np.ndarray:
    """Holm step-down adjusted p-values, preserving input order (docs/05 §13).

    NaN p-values are excluded from the family and stay NaN in the output.
    Adjusted values are monotone non-decreasing in the sorted order (standard
    Holm) and capped at 1.
    """
    array = np.asarray(pvalues, dtype=float)
    if array.ndim != 1:
        raise StatisticsError("holm_correction expects a 1-D sequence of p-values")
    adjusted = np.full_like(array, np.nan)
    valid = np.isfinite(array)
    if not np.any(valid):
        return adjusted
    positions = np.where(valid)[0]
    p = array[positions]
    if np.any((p < 0.0) | (p > 1.0)):
        raise StatisticsError("p-values must be in [0, 1]")
    m = p.size
    order = positions[np.argsort(p, kind="stable")]
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, float(array[index]) * (m - rank))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def holm_correction_labelled(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm correction for a labelled family (e.g. strategy×model×explainer)."""
    labels = list(pvalues)
    raw = np.array([pvalues[label] for label in labels], dtype=float)
    adjusted = holm_correction(raw)
    return {label: float(adjusted[index]) for index, label in enumerate(labels)}


def cohens_d_paired(a: Any, b: Any) -> float:
    """Cohen's d for paired samples: ``mean(a-b) / sd(a-b)``."""
    array_a = _as_1d(a, "a")
    array_b = _as_1d(b, "b")
    _check_same_length(array_a, array_b)
    mask = _finite_mask(array_a, array_b)
    differences = array_a[mask] - array_b[mask]
    if differences.size < 2:
        return float("nan")
    sd = float(np.std(differences, ddof=1))
    mean = float(np.mean(differences))
    if sd == 0.0:
        return 0.0 if mean == 0.0 else math.copysign(float("inf"), mean)
    return mean / sd


def cohens_d(a: Any, b: Any) -> float:
    """Cohen's d for independent samples (pooled standard deviation)."""
    array_a = _as_1d(a, "a")
    array_b = _as_1d(b, "b")
    array_a = array_a[np.isfinite(array_a)]
    array_b = array_b[np.isfinite(array_b)]
    if array_a.size < 2 or array_b.size < 2:
        return float("nan")
    n_a, n_b = array_a.size, array_b.size
    pooled_variance = ((n_a - 1) * np.var(array_a, ddof=1) + (n_b - 1) * np.var(array_b, ddof=1)) / (n_a + n_b - 2)
    pooled_sd = float(np.sqrt(pooled_variance))
    if pooled_sd == 0.0:
        return 0.0
    return float((np.mean(array_a) - np.mean(array_b)) / pooled_sd)


def rank_biserial_from_wilcoxon(a: Any, b: Any) -> float:
    """Matched-pairs rank-biserial correlation ``r = (W+ - W-) / (W+ + W-)``."""
    array_a = _as_1d(a, "a")
    array_b = _as_1d(b, "b")
    _check_same_length(array_a, array_b)
    mask = _finite_mask(array_a, array_b)
    differences = array_a[mask] - array_b[mask]
    differences = differences[differences != 0.0]
    if differences.size == 0:
        return 0.0
    ranks = _scipy_stats.rankdata(np.abs(differences))
    w_plus = float(ranks[differences > 0].sum())
    w_minus = float(ranks[differences < 0].sum())
    total = w_plus + w_minus
    if total == 0.0:
        return 0.0
    return (w_plus - w_minus) / total


def wilcoxon_signed_rank(
    a: Any,
    b: Any,
    *,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank on ``a - b``; degenerate all-zero input -> p = 1."""
    array_a = _as_1d(a, "a")
    array_b = _as_1d(b, "b")
    _check_same_length(array_a, array_b)
    mask = _finite_mask(array_a, array_b)
    differences = array_a[mask] - array_b[mask]
    if differences.size == 0 or np.all(differences == 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "n": float(differences.size)}
    result = _scipy_stats.wilcoxon(differences, alternative=alternative, zero_method="wilcox")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": float(differences.size)}


def effect_size_summary(a: Any, b: Any, *, paired: bool = True) -> dict[str, float]:
    """Effect sizes required next to every comparison (docs/05 §13)."""
    array_a = _as_1d(a, "a")
    array_b = _as_1d(b, "b")
    _check_same_length(array_a, array_b)
    mask = _finite_mask(array_a, array_b)
    differences = array_a[mask] - array_b[mask]
    summary = {
        "n": float(differences.size),
        "mean_diff": float(np.mean(differences)) if differences.size else float("nan"),
        "median_diff": float(np.median(differences)) if differences.size else float("nan"),
        "cohens_d": cohens_d_paired(array_a, array_b) if paired else cohens_d(array_a, array_b),
        "rank_biserial": rank_biserial_from_wilcoxon(array_a, array_b),
    }
    return summary


def binomial_test_proportion(
    successes: int,
    n: int,
    *,
    p0: float = 0.05,
    alternative: str = "greater",
) -> dict[str, float]:
    """Exact binomial test and normal approximation for a proportion.

    Used by H1 (docs/05 §9): fraction of anchors above the null q95 versus the
    5 % expected by construction. Both p-values are reported as required.
    """
    if n <= 0:
        raise StatisticsError("binomial_test_proportion requires n > 0")
    if not 0 <= successes <= n:
        raise StatisticsError(f"successes={successes} outside [0, n={n}]")
    exact = _scipy_stats.binomtest(int(successes), int(n), p=p0, alternative=alternative)
    p_hat = successes / n
    standard_error = math.sqrt(p0 * (1.0 - p0) / n)
    z = (p_hat - p0) / standard_error if standard_error > 0 else float("nan")
    if alternative == "greater":
        normal_p = float(_scipy_stats.norm.sf(z)) if np.isfinite(z) else float("nan")
    elif alternative == "less":
        normal_p = float(_scipy_stats.norm.cdf(z)) if np.isfinite(z) else float("nan")
    else:
        normal_p = float(2.0 * _scipy_stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    return {
        "successes": float(successes),
        "n": float(n),
        "fraction": float(p_hat),
        "expected": float(p0),
        "p_binomial": float(exact.pvalue),
        "z_normal": float(z),
        "p_normal": float(normal_p),
    }


# ---------------------------------------------------------------------------
# Logistic regression + AUC (RQ3, docs/05 §11)
# ---------------------------------------------------------------------------
def standardize_features(X: Any, *, mean: Any | None = None, std: Any | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score features; when ``mean``/``std`` are omitted they are fit on ``X``.

    Callers fitting on a training fold must pass the training-fold statistics to
    transform the held-out fold (no test statistics in model selection,
    docs/05 §11).
    """
    array = np.asarray(X, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise StatisticsError(f"X must be 2-D (n, p), got shape {array.shape}")
    if mean is None:
        mean = np.mean(array, axis=0)
    if std is None:
        std = np.std(array, axis=0, ddof=0)
    mean = np.asarray(mean, dtype=float).reshape(1, -1)
    std = np.asarray(std, dtype=float).reshape(1, -1)
    safe_std = np.where(std == 0.0, 1.0, std)
    return (array - mean) / safe_std, mean.ravel(), std.ravel()


def _make_logistic_regression(*, C: float, max_iter: int, random_state: int):
    """Version-safe L2 ``LogisticRegression`` factory (sklearn 1.8 deprecated ``penalty``).

    Protocol (docs/05 §11) fixes an L2-penalised logistic regression; newer
    scikit-learn expresses L2 through the default ``l1_ratio=0`` instead of the
    deprecated ``penalty='l2'`` string, so the factory adapts instead of pinning
    a warning/behaviour that changes between lanes (torch/sklearn pins differ).
    """
    import inspect

    from sklearn.linear_model import LogisticRegression

    params = inspect.signature(LogisticRegression.__init__).parameters
    kwargs: dict[str, Any] = {
        "C": float(C),
        "solver": "lbfgs",
        "max_iter": int(max_iter),
        "random_state": int(random_state),
    }
    penalty_parameter = params.get("penalty")
    if penalty_parameter is not None and penalty_parameter.default == "deprecated":
        kwargs["l1_ratio"] = 0.0  # sklearn >= 1.8: l1_ratio=0 == L2 penalty
    elif penalty_parameter is not None:
        kwargs["penalty"] = "l2"
    return LogisticRegression(**kwargs)


def fit_logistic_l2(
    X: Any,
    y: Any,
    *,
    C: float = 1.0,
    standardize: bool = True,
    max_iter: int = 1000,
    random_state: int = 0,
) -> dict[str, Any]:
    """L2-regularised logistic regression with standardised coefficients.

    Returns coefficients in **standardised-feature units** so they are
    comparable across features (Doc 2b §S11: "standardise features; report
    coefficients with CIs"). Fitting is delegated to scikit-learn's
    ``LogisticRegression`` (lbfgs, L2); no tuning happens here.
    """
    X_array = _as_feature_matrix(X)
    y_array = _as_binary(y, X_array.shape[0])
    if np.unique(y_array).size < 2:
        raise StatisticsError("fit_logistic_l2 needs both classes present")
    if standardize:
        X_scaled, mean, std = standardize_features(X_array)
    else:
        X_scaled = X_array
        mean = np.zeros(X_array.shape[1])
        std = np.ones(X_array.shape[1])
    model = _make_logistic_regression(C=C, max_iter=max_iter, random_state=random_state)
    model.fit(X_scaled, y_array)
    return {
        "model": model,
        "intercept": float(model.intercept_[0]),
        "coef_standardized": np.asarray(model.coef_[0], dtype=float),
        "feature_mean": mean,
        "feature_std": std,
        "C": float(C),
        "n": int(y_array.size),
        "n_pos": int(np.sum(y_array == 1)),
        "n_neg": int(np.sum(y_array == 0)),
    }


def logistic_lopo_auc(
    X: Any,
    y: Any,
    groups: Any,
    *,
    C: float = 1.0,
    standardize: bool = True,
    max_iter: int = 1000,
    random_state: int = 0,
) -> AucResult:
    """Pooled out-of-fold AUC of an L2 logistic model under leave-one-period-out CV.

    For each held-out period/group the model is fit on the remaining groups and
    scored on the held-out rows; the reported AUC is computed on the pooled
    out-of-fold scores (docs/05 §11). Standardisation statistics are fit on the
    training folds only. Degenerate folds (single class in training) fall back to
    a constant 0.5 score and are listed in ``notes`` rather than silently
    dropped.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X_array = _as_feature_matrix(X)
    y_array = _as_binary(y, X_array.shape[0])
    group_array = np.asarray(groups).reshape(-1)
    if group_array.size != X_array.shape[0]:
        raise StatisticsError(f"groups length {group_array.size} != number of rows {X_array.shape[0]}")
    if np.unique(y_array).size < 2:
        raise StatisticsError("logistic_lopo_auc needs both outcome classes in y")
    unique_groups = list(dict.fromkeys(group_array.tolist()))
    if len(unique_groups) < 2:
        raise StatisticsError("leave-one-period-out CV needs at least two period groups")
    oof = np.full(X_array.shape[0], np.nan, dtype=float)
    per_fold: dict[Any, float] = {}
    notes: list[str] = []
    for group in unique_groups:
        test_mask = group_array == group
        train_mask = ~test_mask
        if not np.any(test_mask):
            continue
        y_train = y_array[train_mask]
        if np.unique(y_train).size < 2:
            oof[test_mask] = 0.5
            per_fold[group] = float("nan")
            notes.append(f"group {group!r}: single class in training fold -> constant 0.5 scores")
            continue
        if standardize:
            X_train_scaled, mean, std = standardize_features(X_array[train_mask])
            X_test_scaled = standardize_features(X_array[test_mask], mean=mean, std=std)[0]
        else:
            X_train_scaled = X_array[train_mask]
            X_test_scaled = X_array[test_mask]
        model = _make_logistic_regression(C=C, max_iter=max_iter, random_state=random_state)
        model.fit(X_train_scaled, y_train)
        scores = model.predict_proba(X_test_scaled)[:, 1]
        oof[test_mask] = scores
        y_test = y_array[test_mask]
        if np.unique(y_test).size < 2:
            per_fold[group] = float("nan")
            notes.append(f"group {group!r}: single class in held-out fold -> fold AUC undefined")
        else:
            per_fold[group] = float(roc_auc_score(y_test, scores))
    if np.any(~np.isfinite(oof)):
        raise StatisticsError("some rows received no out-of-fold score")
    pooled_auc = float(roc_auc_score(y_array, oof))
    return AucResult(
        auc=pooled_auc,
        oof_scores=oof,
        y=y_array,
        groups=group_array,
        per_fold_auc=per_fold,
        n=int(y_array.size),
        n_pos=int(np.sum(y_array == 1)),
        n_neg=int(np.sum(y_array == 0)),
        n_folds=len(per_fold),
        notes=tuple(notes),
    )


def bootstrap_auc(
    y: Any,
    scores: Any,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> BootstrapResult:
    """Case-control (label-stratified) bootstrap CI for an AUC.

    Stratifying by label guarantees both classes are present in every resample;
    this is the docs/05 §13 "stratified by class" rule applied to AUC.
    """
    from sklearn.metrics import roc_auc_score

    y_array = _as_binary(y)
    score_array = _as_1d(scores, "scores")
    _check_same_length(y_array, score_array, name_a="y", name_b="scores")
    if np.unique(y_array).size < 2:
        raise StatisticsError("bootstrap_auc needs both classes in y")
    point = float(roc_auc_score(y_array, score_array))
    rng = np.random.default_rng(seed)
    groups = [np.where(y_array == value)[0] for value in np.unique(y_array)]
    positions = _resample_rows(groups, n_resamples, rng)
    label_pattern = np.concatenate([np.full(group.size, value) for group, value in zip(groups, np.unique(y_array))])
    distribution = _auc_rows(label_pattern, score_array[positions])
    lo, hi = _percentile_interval(distribution, alpha)
    return BootstrapResult(point, lo, hi, int(y_array.size), n_resamples, alpha, seed, "auc")


def paired_auc_delta_bootstrap(
    y: Any,
    scores_a: Any,
    scores_b: Any,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> AucDeltaResult:
    """Paired bootstrap of ``AUC(a) - AUC(b)`` on identical units (docs/05 §11).

    The same resample indices are used for both score vectors, so the delta is
    properly paired; resampling is stratified by label. The p-value is the
    two-sided bootstrap sign test on the delta distribution.
    """
    from sklearn.metrics import roc_auc_score

    y_array = _as_binary(y)
    score_a = _as_1d(scores_a, "scores_a")
    score_b = _as_1d(scores_b, "scores_b")
    _check_same_length(y_array, score_a, name_a="y", name_b="scores_a")
    _check_same_length(y_array, score_b, name_a="y", name_b="scores_b")
    if np.unique(y_array).size < 2:
        raise StatisticsError("paired_auc_delta_bootstrap needs both classes in y")
    auc_a = float(roc_auc_score(y_array, score_a))
    auc_b = float(roc_auc_score(y_array, score_b))
    delta = auc_a - auc_b
    rng = np.random.default_rng(seed)
    groups = [np.where(y_array == value)[0] for value in np.unique(y_array)]
    positions = _resample_rows(groups, n_resamples, rng)
    label_pattern = np.concatenate([np.full(group.size, value) for group, value in zip(groups, np.unique(y_array))])
    distribution = _auc_rows(label_pattern, score_a[positions]) - _auc_rows(label_pattern, score_b[positions])
    lo, hi = _percentile_interval(distribution, alpha)
    return AucDeltaResult(
        auc_a=auc_a,
        auc_b=auc_b,
        delta=delta,
        lo=lo,
        hi=hi,
        p_value=_bootstrap_sign_p(distribution),
        n=int(y_array.size),
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
    )


def majority_rate(y: Any) -> float:
    """Largest class frequency in ``y`` (the accuracy-oracle baseline rate)."""
    values = np.asarray(y)
    if values.size == 0:
        return float("nan")
    _, counts = np.unique(values, return_counts=True)
    return float(counts.max() / values.size)


# ---------------------------------------------------------------------------
# Feature/outcome coercion helpers
# ---------------------------------------------------------------------------
def _as_feature_matrix(X: Any) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise StatisticsError(f"X must be 2-D (n, p), got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise StatisticsError("X contains NaN/inf; impute or drop rows before modelling (no silent filling)")
    return array


def _as_binary(y: Any, n: int | None = None) -> np.ndarray:
    array = np.asarray(y)
    if array.ndim != 1:
        raise StatisticsError(f"y must be 1-D, got shape {array.shape}")
    if n is not None and array.size != n:
        raise StatisticsError(f"y has {array.size} entries but X has {n} rows")
    if array.dtype == bool:
        return array.astype(int)
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise StatisticsError(f"y must be numeric/binary: {exc}") from exc
    unique = np.unique(numeric)
    if not set(unique.tolist()).issubset({0.0, 1.0}):
        raise StatisticsError(f"y must be binary {{0,1}}, found {unique.tolist()[:10]}")
    return numeric.astype(int)


__all__ = [
    "AucDeltaResult",
    "AucResult",
    "BootstrapResult",
    "DEFAULT_ALPHA",
    "DEFAULT_N_RESAMPLES",
    "DEFAULT_SEED",
    "PairedBootstrapResult",
    "PILOT_N_INNER",
    "PILOT_N_OUTER",
    "PermutationResult",
    "StatisticsError",
    "binomial_test_proportion",
    "bootstrap_auc",
    "bootstrap_ci",
    "cohens_d",
    "cohens_d_paired",
    "effect_size_summary",
    "fit_logistic_l2",
    "hierarchical_bootstrap_ci",
    "holm_correction",
    "holm_correction_labelled",
    "logistic_lopo_auc",
    "majority_rate",
    "paired_auc_delta_bootstrap",
    "paired_bootstrap_diff",
    "partial_spearman",
    "partial_spearman_ci",
    "permutation_test",
    "rank_biserial_from_wilcoxon",
    "spearman_correlation",
    "spearman_correlation_ci",
    "standardize_features",
    "stratified_bootstrap_ci",
    "stratify_indices",
    "wilcoxon_signed_rank",
]
