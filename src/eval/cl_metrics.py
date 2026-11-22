"""S5 continual-learning evaluation metrics (Doc 2 S5) + D-033/D-047 dual macro-F1.

Definitions are **imported from the authoritative analysis toolkit** (D-044:
``analysis.metrics_cl``); this module is a thin writer-facing wrapper so the
engineer's S5 artifacts and the analyst's crosscheck cannot diverge:

* ``R[i, j]`` = test macro-F1 on period ``j`` after training through period ``i``
  (0-based; Doc 2 writes it 1-based). Only the lower-triangular cells
  ``i >= j`` are legitimate evaluations; upper-triangle cells are future
  evaluations recorded for completeness and never used by best-so-far.
* final average F1 = mean over periods of the last row,
* forgetting per period ``F_j = max_{i >= j} R[i, j] - R[T-1, j]`` (A11; D-047),
* backward transfer ``BWT = mean_{j < T-1} (R[T-1, j] - R[j, j])`` (same
  triangular rule; forward cells excluded),
* per-class F1 tensor for RQ3,
* D-033 dual macro-F1: global 10-class macro-F1 (absent classes count as 0) and
  supported-set macro-F1 over **7 classes**: Benign + DoS/Exploits/Fuzzers/
  Generic/Reconnaissance/Shellcode (D-047 correction).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from src.analysis.metrics_cl import RMetricTensor, macro_f1_from_per_class

#: D-033/D-047 supported set: the 6 attack classes **plus Benign** (7 classes).
DEFAULT_SUPPORTED_SET: tuple[str, ...] = (
    "Benign",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
)

#: A11/D-047 convention recorded in every metrics block.
FORGETTING_CONVENTION = (
    "analysis.metrics_cl/A11: F_j = max_{i>=j} R[i,j] - R[T-1,j]; "
    "forward-evaluation cells (i<j) are never used; BWT follows the triangular rule"
)


def matrix_from_scores(scores: Mapping[tuple[int, int], float], n_periods: int) -> np.ndarray:
    """Build the R matrix from ``(checkpoint_period, evaluated_period) -> score``."""
    matrix = np.full((n_periods, n_periods), np.nan, dtype="float64")
    for (row, column), value in scores.items():
        if not (0 <= row < n_periods and 0 <= column < n_periods):
            raise IndexError(f"score key ({row}, {column}) outside a {n_periods}x{n_periods} matrix")
        matrix[row, column] = float(value)
    return matrix


def cl_metrics(R: np.ndarray) -> dict[str, Any]:
    """Compute final average F1, forgetting and BWT via the authoritative reference.

    The reference (``analysis.metrics_cl.RMetricTensor``) implements the A11/D-047
    triangular best-so-far rule, so forward-evaluation cells can never inflate
    forgetting. A ``T=1`` matrix is outside the frozen protocol; it returns a
    degenerate block instead of raising.
    """
    matrix = np.asarray(R, dtype="float64")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"R must be a square matrix, got shape {matrix.shape}")
    n_periods = matrix.shape[0]
    if n_periods == 0:
        raise ValueError("R matrix is empty")
    last = matrix[n_periods - 1]
    if np.isnan(last).any():
        # T34 (bug-report #18): incremental evaluation is legitimate — the final
        # checkpoint may only have some (or none) of its cells evaluated yet. Return a
        # partial block with no final metrics instead of raising; callers that need the
        # final numbers must wait for a complete last row (run_eval.py already does).
        import warnings

        finite = np.isfinite(last)
        warnings.warn(
            f"last checkpoint row has {n_periods - int(finite.sum())} NaN cell(s); "
            "returning partial metrics without final_average_macro_f1/forgetting/bwt",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "n_periods": int(n_periods),
            "partial": True,
            "evaluated_final_periods": int(finite.sum()),
            "final_average_macro_f1": None,
            "forgetting": None,
            "forgetting_per_period": None,
            "bwt": None,
            "bwt_per_period": None,
            "last_row": [None if not np.isfinite(value) else float(value) for value in last.tolist()],
            "diagonal": [
                None if not np.isfinite(matrix[index, index]) else float(matrix[index, index])
                for index in range(n_periods)
            ],
            "convention": FORGETTING_CONVENTION
            + "; partial last row: no final metrics computed (incremental evaluation)",
        }
    if n_periods == 1:
        return {
            "n_periods": 1,
            "final_average_macro_f1": float(last[0]),
            "forgetting": 0.0,
            "forgetting_per_period": {"1": None},
            "bwt": 0.0,
            "bwt_per_period": {"1": None},
            "last_row": [float(value) for value in last.tolist()],
            "diagonal": [float(matrix[0, 0])],
            "convention": FORGETTING_CONVENTION + "; T=1 degenerate case returns 0.0 (outside frozen T=5 protocol)",
        }
    tensor = RMetricTensor(matrix, n_periods=n_periods)
    tensor.validate()
    forgetting_per_period_raw = np.asarray(tensor.forgetting_per_period(), dtype="float64")
    bwt_per_period_raw = np.asarray(tensor.bwt_per_period(), dtype="float64")
    forgetting_per_period = {
        str(index + 1): (None if math.isnan(value) else float(value))
        for index, value in enumerate(forgetting_per_period_raw)
    }
    bwt_per_period = {
        str(index + 1): (None if math.isnan(value) else float(value))
        for index, value in enumerate(bwt_per_period_raw)
    }
    return {
        "n_periods": int(n_periods),
        "final_average_macro_f1": float(tensor.final_macro_f1()),
        "forgetting": float(tensor.forgetting()),
        "forgetting_per_period": forgetting_per_period,
        "bwt": float(tensor.bwt()),
        "bwt_per_period": bwt_per_period,
        "last_row": [float(value) for value in last.tolist()],
        "diagonal": [float(matrix[index, index]) for index in range(n_periods)],
        "convention": FORGETTING_CONVENTION,
    }


def per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: Sequence[int],
) -> dict[str, float]:
    from sklearn.metrics import f1_score

    scores = f1_score(y_true, y_pred, labels=list(class_ids), average=None, zero_division=0)
    return {str(int(label)): float(score) for label, score in zip(class_ids, scores)}


def dual_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_ids: Sequence[int],
    supported_ids: Sequence[int],
) -> dict[str, Any]:
    """D-033 dual macro-F1 (all classes + supported set) plus class counts.

    Macro averaging uses the authoritative ``analysis.metrics_cl.macro_f1_from_per_class``
    so the supported-set definition cannot silently drift (D-047: includes Benign).
    """
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    all_classes = list(int(value) for value in class_ids)
    supported = list(int(value) for value in supported_ids)
    if not all_classes:
        raise ValueError("class_ids must not be empty")
    missing = [value for value in supported if value not in all_classes]
    if missing:
        raise ValueError(f"supported class ids {missing} are not in the global class space")
    all_scores = f1_score(y_true, y_pred, labels=all_classes, average=None, zero_division=0)
    supported_scores = f1_score(y_true, y_pred, labels=supported, average=None, zero_division=0)
    macro_all = float(
        macro_f1_from_per_class(
            {str(label): float(score) for label, score in zip(all_classes, all_scores)},
            nan_policy="omit",
        )
    )
    macro_supported = float(
        macro_f1_from_per_class(
            {str(label): float(score) for label, score in zip(supported, supported_scores)},
            nan_policy="omit",
        )
    )
    counts = Counter(int(value) for value in y_true.tolist())
    return {
        "macro_f1_all_classes": macro_all,
        "macro_f1_supported_set": macro_supported,
        "n_classes_all": len(all_classes),
        "n_classes_supported": len(supported),
        "supported_set_labels": supported,
        "class_counts": {str(label): int(counts.get(label, 0)) for label in all_classes},
        "n_edges": int(len(y_true)),
        "supported_set_definition": "D-033/D-047: 6 attack classes + Benign (7)",
    }


def supported_ids_from_names(
    names: Sequence[str],
    class_order: Sequence[str] | None,
) -> list[int]:
    """Map D-033 supported-class names to frozen class ids."""
    if not class_order:
        raise ValueError("dataset spec has no class_order; cannot map supported set names")
    lookup = {name: index for index, name in enumerate(class_order)}
    unknown = [name for name in names if name not in lookup]
    if unknown:
        raise ValueError(f"supported class names not in class_order: {unknown}")
    return [lookup[name] for name in names]


def supported_set_from_config(config: Mapping[str, Any]) -> list[str]:
    """Read the frozen supported-set names from ``configs/eval.yaml`` style config."""
    s5 = config.get("s5") or {}
    dual = (s5.get("dual_macro_f1") or {}) if isinstance(s5, Mapping) else {}
    names = dual.get("supported_set") if isinstance(dual, Mapping) else None
    if names:
        return [str(name) for name in names]
    return list(DEFAULT_SUPPORTED_SET)


__all__ = [
    "DEFAULT_SUPPORTED_SET",
    "FORGETTING_CONVENTION",
    "cl_metrics",
    "dual_macro_f1",
    "matrix_from_scores",
    "per_class_f1",
    "supported_ids_from_names",
    "supported_set_from_config",
]
