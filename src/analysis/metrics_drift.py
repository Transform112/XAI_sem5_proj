"""Drift metrics for explanation stability (stage S8).

Authority
---------
* ``docs/source_documents/Doc 2b`` §S8 "Drift metrics and noise floor": the similarity
  family and ``drift = 1 - similarity`` for every metric; "report the drift
  vector, never only the primary"; primary metric is fixed in advance.
* ``docs/05_EXPERIMENT_PROTOCOL.md`` §8 "Drift metrics (S8)":
  feature-attribution cosine, top-k Jaccard (k = 3, 5, 10), Spearman over the
  full vector, edge-mask Jaccard of the top-m edges (m = 5) and cosine over the
  mask vector; **primary metric (fixed)** ``drift = 1 - Jaccard(top-5 feature
  sets)`` for IG feature attributions, with cosine and Spearman secondary, and
  edge-mask Jaccard reported for GNNExplainer.
* ``docs/05`` §6 / Doc 2b §S7 fix the explanation objects: feature attribution
  vector ``a in R^d`` and edge-importance vector over the anchor's **stored**
  period-j 2-hop subgraph (never re-sampled, so vector order/keys are constant
  across checkpoints).

Definitions (and the conventions the protocol leaves open)
----------------------------------------------------------
* ``similarity`` is computed on the *signed* attribution vector for cosine and
  Spearman; top-k sets and top-m edge sets rank by **absolute** importance
  (a feature that flips sign but keeps magnitude is not a set-membership
  change), ties broken by ascending feature/edge id so results are
  deterministic. Both conventions are recorded in ``CONVENTIONS`` because
  Doc 2b §S8 does not spell them out.
* cosine with a zero-norm vector is mathematically undefined: the convention
  here is ``cos(0, 0) = 1`` (two identical "no attribution" explanations) and
  ``cos(0, x) = 0`` for ``x != 0``. Documented in ``CONVENTIONS`` and flagged
  to the Lead (protocol ambiguity A1).
* Spearman drift is ``1 - rho`` and therefore in ``[0, 2]`` (negative rank
  correlation gives drift > 1). It is reported raw, never clipped, so a sign
  reversal cannot be hidden (protocol ambiguity A2).
* Jaccard of two empty sets is 1.0 (identical explanations); with ``k > d`` the
  top-k set is the full feature set.
* Inputs must be finite; NaN/inf in an attribution vector raises ``ValueError``
  instead of silently contaminating the noise-floor distribution.

All functions are pure: they never touch the filesystem or real results, which
keeps them usable by both the analysis toolbox and the tests.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

try:  # optional local-analysis fallback; Kaggle/production environments have scipy
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - exercised only on minimal installs
    _scipy_stats = None

#: Feature-attribution drift metrics fixed by docs/05 §8.
FEATURE_METRICS: tuple[str, ...] = (
    "cosine",
    "jaccard_top3",
    "jaccard_top5",
    "jaccard_top10",
    "spearman",
)

#: Primary metric fixed by docs/05 §8: ``1 - Jaccard(top-5 feature sets)``.
PRIMARY_FEATURE_METRIC: str = "jaccard_top5"

#: Edge-importance drift metrics fixed by docs/05 §8 (GNNExplainer).
EDGE_METRICS: tuple[str, ...] = ("edge_jaccard_top5", "edge_cosine")

#: Primary edge metric per docs/05 §8 ("edge-mask Jaccard reported for GNNExplainer").
PRIMARY_EDGE_METRIC: str = "edge_jaccard_top5"

#: k values fixed by docs/05 §8.
DEFAULT_TOP_K: tuple[int, ...] = (3, 5, 10)

#: m fixed by docs/05 §8 (top-5 edges).
DEFAULT_EDGE_M: int = 5

#: Conventions the frozen protocol does not state explicitly; reported to the
#: Lead instead of being applied silently.
CONVENTIONS: dict[str, str] = {
    "topk_ranking": "rank by |attribution| descending, ties by ascending feature/edge id",
    "cosine_zero_norm": "cos(0,0)=1; cos(0,x!=0)=0",
    "spearman_drift_range": "drift = 1 - rho in [0, 2]; never clipped",
    "jaccard_empty": "J(empty, empty) = 1",
    "edge_keys": "top-m edge ids are the vector positions; explicit keys supported",
}


class MetricConventionsError(ValueError):
    """Raised when inputs violate an input contract of this module (e.g. NaN)."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def as_vector(values: Any, *, name: str = "vector", allow_empty: bool = False) -> np.ndarray:
    """Convert *values* to a finite 1-D float array (S8 input contract)."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    if array.size == 0 and not allow_empty:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values (NaN/inf); drift metrics require finite attributions")
    return array


# ---------------------------------------------------------------------------
# Set / rank primitives
# ---------------------------------------------------------------------------
def topk_indices(values: Any, k: int) -> np.ndarray:
    """Indices of the top-*k* entries by absolute magnitude (ascending ids).

    Documented convention (Doc 2b §S8 leaves ties open): rank by ``|value|``
    descending, break ties by ascending index, return the selected ids sorted
    ascending so set operations are canonical. ``k`` is capped at ``len(values)``.
    """
    array = as_vector(values, name="attribution vector", allow_empty=True)
    n = array.size
    if k is None:
        raise TypeError("k must be an int")
    if int(k) != k:
        raise TypeError(f"k must be an int, got {k!r}")
    k = int(k)
    if k <= 0 or n == 0:
        return np.empty(0, dtype=int)
    k = min(k, n)
    order = np.lexsort((np.arange(n), -np.abs(array)))
    return np.sort(order[:k].astype(int))


def jaccard_index(set_a: Iterable[int], set_b: Iterable[int]) -> float:
    """Jaccard similarity of two id sets; ``J(empty, empty) = 1`` by convention."""
    a = set(set_a)  # ints, numpy ints or explicit edge keys are all hashable
    b = set(set_b)
    if not a and not b:
        return 1.0
    union = a | b
    if not union:  # pragma: no cover - defensive
        return 1.0
    return len(a & b) / len(union)


def cosine_similarity(a: Any, b: Any) -> float:
    """Cosine similarity with the documented zero-norm convention.

    ``cos(0, 0) = 1`` (identical empty explanations), ``cos(0, x != 0) = 0``.
    """
    va = as_vector(a, name="vector a")
    vb = as_vector(b, name="vector b")
    if va.size != vb.size:
        raise ValueError(f"vectors must have equal length (got {va.size} vs {vb.size})")
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def spearman_similarity(a: Any, b: Any) -> float:
    """Spearman rank correlation over the full attribution vector (docs/05 §8).

    Identical orderings give 1.0; a constant vector makes rank correlation
    undefined and returns ``nan`` (callers must handle it explicitly rather than
    treat it as a similarity).
    """
    va = as_vector(a, name="vector a")
    vb = as_vector(b, name="vector b")
    if va.size != vb.size:
        raise ValueError(f"vectors must have equal length (got {va.size} vs {vb.size})")
    if va.size < 2:
        return float("nan")
    if np.ptp(va) == 0.0 or np.ptp(vb) == 0.0:
        return float("nan")
    if _scipy_stats is None:
        # Exact Spearman rho = Pearson correlation of average ranks.  This
        # fallback keeps the metric usable on minimal local installs; the
        # production requirements still include scipy.
        import pandas as pd

        rank_a = pd.Series(va).rank(method="average").to_numpy(dtype=float)
        rank_b = pd.Series(vb).rank(method="average").to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = float(np.corrcoef(rank_a, rank_b)[0, 1])
        return rho
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = float(_scipy_stats.spearmanr(va, vb).statistic)
    return rho


def _drift(similarity: float) -> float:
    """``drift = 1 - similarity`` (Doc 2b §S8) — NaN stays NaN."""
    return float("nan") if similarity is None or (isinstance(similarity, float) and math.isnan(similarity)) else 1.0 - float(similarity)


# ---------------------------------------------------------------------------
# Feature-attribution drift
# ---------------------------------------------------------------------------
def attribution_similarity(a: Any, b: Any, ks: Sequence[int] = DEFAULT_TOP_K) -> dict[str, float]:
    """Full S8 similarity family for one pair of feature-attribution vectors.

    Returns ``{"cosine", "jaccard_top3", "jaccard_top5", "jaccard_top10",
    "spearman"}`` per docs/05 §8 / Doc 2b §S8.
    """
    va = as_vector(a, name="vector a")
    vb = as_vector(b, name="vector b")
    if va.size != vb.size:
        raise ValueError(f"vectors must have equal length (got {va.size} vs {vb.size})")
    result: dict[str, float] = {"cosine": cosine_similarity(va, vb), "spearman": spearman_similarity(va, vb)}
    for k in ks:
        result[f"jaccard_top{int(k)}"] = jaccard_index(topk_indices(va, k), topk_indices(vb, k))
    # stable ordering regardless of the ks input order
    ordered = {name: result[name] for name in FEATURE_METRICS if name in result}
    ordered.update({name: value for name, value in result.items() if name not in ordered})
    return ordered


def attribution_drift(a: Any, b: Any, ks: Sequence[int] = DEFAULT_TOP_K) -> dict[str, float]:
    """``1 - similarity`` for every feature metric; enables "drift vector" reporting."""
    return {name: _drift(value) for name, value in attribution_similarity(a, b, ks).items()}


def primary_feature_drift(a: Any, b: Any) -> float:
    """The pre-registered primary metric: ``1 - Jaccard(top-5 feature sets)``."""
    return _drift(jaccard_index(topk_indices(a, 5), topk_indices(b, 5)))


# ---------------------------------------------------------------------------
# Edge-mask drift
# ---------------------------------------------------------------------------
def edge_topm_indices(mask: Any, m: int = DEFAULT_EDGE_M) -> np.ndarray:
    """Indices of the top-*m* important edges (absolute mask value, ties by id)."""
    return topk_indices(mask, m)


def _remap_ids(indices: np.ndarray, keys: Sequence[Any] | None) -> tuple[Any, ...]:
    if keys is None:
        return tuple(int(i) for i in indices)
    keys_list = list(keys)
    return tuple(keys_list[int(i)] for i in indices)


def edge_mask_similarity(
    mask_a: Any,
    mask_b: Any,
    *,
    m: int = DEFAULT_EDGE_M,
    keys_a: Sequence[Any] | None = None,
    keys_b: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Edge-importance similarity: top-m Jaccard + cosine over the mask vector.

    The anchor's stored 2-hop subgraph is fixed across checkpoints (Doc 2b §S7),
    so ``mask_a`` and ``mask_b`` must be aligned (same edge order). Explicit
    ``keys_a``/``keys_b`` let callers map vector positions to global edge/host-pair
    ids; when both are given the top-m sets are compared by key instead of
    position.

    Returns ``{"edge_jaccard_top5", "edge_cosine"}`` (docs/05 §8).
    """
    va = as_vector(mask_a, name="edge mask a")
    vb = as_vector(mask_b, name="edge mask b")
    if va.size != vb.size:
        raise ValueError(f"edge masks must have equal length (got {va.size} vs {vb.size})")
    idx_a = edge_topm_indices(va, m)
    idx_b = edge_topm_indices(vb, m)
    set_a = _remap_ids(idx_a, keys_a) if keys_a is not None else tuple(int(i) for i in idx_a)
    set_b = _remap_ids(idx_b, keys_b) if keys_b is not None else tuple(int(i) for i in idx_b)
    return {
        "edge_jaccard_top5": jaccard_index(set_a, set_b),
        "edge_cosine": cosine_similarity(va, vb),
    }


def edge_mask_drift(
    mask_a: Any,
    mask_b: Any,
    *,
    m: int = DEFAULT_EDGE_M,
    keys_a: Sequence[Any] | None = None,
    keys_b: Sequence[Any] | None = None,
) -> dict[str, float]:
    """``1 - similarity`` for every edge metric (docs/05 §8)."""
    return {
        name: _drift(value)
        for name, value in edge_mask_similarity(mask_a, mask_b, m=m, keys_a=keys_a, keys_b=keys_b).items()
    }


def primary_edge_drift(mask_a: Any, mask_b: Any, m: int = DEFAULT_EDGE_M) -> float:
    """The pre-registered primary edge metric: ``1 - Jaccard(top-5 edges)``."""
    return _drift(jaccard_index(edge_topm_indices(mask_a, m), edge_topm_indices(mask_b, m)))


# ---------------------------------------------------------------------------
# Batch helpers (anchor-level, used by S9 noise floor and the S8/S9 tables)
# ---------------------------------------------------------------------------
def pairwise_feature_drift(
    explanations_a: dict[Any, np.ndarray],
    explanations_b: dict[Any, np.ndarray],
    *,
    ks: Sequence[int] = DEFAULT_TOP_K,
    anchor_order: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Per-anchor drift rows for a matched set of explanations (S8/S9).

    ``explanations_a``/``explanations_b`` map ``anchor_id -> attribution vector``;
    both must contain the same anchors (the protocol's fixed anchor set, Doc 2
    §S6). Returns one dict per anchor with ``anchor_id`` plus every metric's
    similarity and drift value. Raises on anchor-set mismatch instead of
    silently intersecting (that would bias the paired comparison).
    """
    keys = list(anchor_order) if anchor_order is not None else sorted(explanations_a, key=str)
    missing_a = [key for key in keys if key not in explanations_a]
    missing_b = [key for key in keys if key not in explanations_b]
    if missing_a or missing_b:
        raise KeyError(f"anchor sets differ: missing from a={missing_a[:5]}, missing from b={missing_b[:5]}")
    rows: list[dict[str, Any]] = []
    for anchor_id in keys:
        similarities = attribution_similarity(explanations_a[anchor_id], explanations_b[anchor_id], ks=ks)
        row: dict[str, Any] = {"anchor_id": anchor_id}
        for name, value in similarities.items():
            row[f"sim_{name}"] = value
            row[f"drift_{name}"] = _drift(value)
        row["drift_primary"] = row[f"drift_{PRIMARY_FEATURE_METRIC}"]
        rows.append(row)
    return rows


def pairwise_edge_drift(
    explanations_a: dict[Any, np.ndarray],
    explanations_b: dict[Any, np.ndarray],
    *,
    m: int = DEFAULT_EDGE_M,
    anchor_order: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Per-anchor edge-mask drift rows (GNNExplainer, docs/05 §8)."""
    keys = list(anchor_order) if anchor_order is not None else sorted(explanations_a, key=str)
    rows: list[dict[str, Any]] = []
    for anchor_id in keys:
        if anchor_id not in explanations_a or anchor_id not in explanations_b:
            raise KeyError(f"anchor {anchor_id!r} missing from one explanation set")
        similarities = edge_mask_similarity(explanations_a[anchor_id], explanations_b[anchor_id], m=m)
        row: dict[str, Any] = {"anchor_id": anchor_id}
        for name, value in similarities.items():
            row[f"sim_{name}"] = value
            row[f"drift_{name}"] = _drift(value)
        row["drift_primary"] = row[f"drift_{PRIMARY_EDGE_METRIC}"]
        rows.append(row)
    return rows


__all__ = [
    "CONVENTIONS",
    "DEFAULT_EDGE_M",
    "DEFAULT_TOP_K",
    "EDGE_METRICS",
    "FEATURE_METRICS",
    "MetricConventionsError",
    "PRIMARY_EDGE_METRIC",
    "PRIMARY_FEATURE_METRIC",
    "as_vector",
    "attribution_drift",
    "attribution_similarity",
    "cosine_similarity",
    "edge_mask_drift",
    "edge_mask_similarity",
    "edge_topm_indices",
    "jaccard_index",
    "pairwise_edge_drift",
    "pairwise_feature_drift",
    "primary_edge_drift",
    "primary_feature_drift",
    "spearman_similarity",
    "topk_indices",
]
