"""Tests for S5 continual-learning metrics (D-033/D-047) and analysis convergence."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.cl_metrics import (
    DEFAULT_SUPPORTED_SET,
    FORGETTING_CONVENTION,
    cl_metrics,
    dual_macro_f1,
    matrix_from_scores,
    per_class_f1,
    supported_ids_from_names,
    supported_set_from_config,
)


# ---------------------------------------------------------------------------
# A11 / D-047: triangular forgetting and BWT
# ---------------------------------------------------------------------------
def test_final_forgetting_bwt_hand_computed():
    # R[i, j] = F1 on period j after training through period i (0-based rows).
    R = np.array(
        [
            [0.80, 0.20, 0.10],
            [0.60, 0.70, 0.30],
            [0.50, 0.40, 0.90],
        ]
    )
    metrics = cl_metrics(R)
    assert metrics["final_average_macro_f1"] == pytest.approx(float(np.mean([0.50, 0.40, 0.90])))
    # A11: F_j = max_{i>=j} R[i,j] - R[T-1,j]
    expected_forgetting = float(np.mean([0.80 - 0.50, 0.70 - 0.40]))
    expected_bwt = float(np.mean([0.50 - 0.80, 0.40 - 0.70]))
    assert metrics["forgetting"] == pytest.approx(expected_forgetting)
    assert metrics["bwt"] == pytest.approx(expected_bwt)
    assert metrics["forgetting_per_period"]["3"] is None  # last period has no later forgetting
    assert "A11" in metrics["convention"]


def test_forgetting_ignores_forward_evaluation_cells():
    """D-047 regression: upper-triangle (future) evaluations must never inflate F_j."""
    without_forward = np.array(
        [
            [0.10, np.nan, np.nan],
            [0.90, 0.90, np.nan],
            [0.50, 0.50, 0.00],
        ]
    )
    inflated_forward = np.array(
        [
            [0.10, 5.0, 5.0],  # checkpoint 0 evaluated on FUTURE periods -> must be ignored
            [0.90, 0.90, 5.0],  # checkpoint 1 evaluated on future period -> must be ignored
            [0.50, 0.50, 0.00],
        ]
    )
    clean = cl_metrics(without_forward)
    inflated = cl_metrics(inflated_forward)
    assert inflated["forgetting"] == pytest.approx(clean["forgetting"])
    assert inflated["bwt"] == pytest.approx(clean["bwt"])
    assert inflated["forgetting"] == pytest.approx(0.40)
    assert inflated["bwt"] == pytest.approx(0.0)
    # Reference crosscheck (D-044): analysis.metrics_cl must agree exactly.
    from analysis.metrics_cl import RMetricTensor

    reference = RMetricTensor(without_forward)
    assert clean["forgetting"] == pytest.approx(reference.forgetting())
    assert clean["bwt"] == pytest.approx(reference.bwt())


def test_cl_metrics_matches_analysis_reference_on_lower_triangular_matrix():
    from analysis.metrics_cl import RMetricTensor

    rng = np.random.default_rng(7)
    n = 5
    R = np.full((n, n), np.nan)
    for row in range(n):
        for column in range(row + 1):
            R[row, column] = float(rng.uniform(0.2, 0.9))
    ours = cl_metrics(R)
    reference = RMetricTensor(R)
    assert ours["final_average_macro_f1"] == pytest.approx(reference.final_macro_f1())
    assert ours["forgetting"] == pytest.approx(reference.forgetting())
    assert ours["bwt"] == pytest.approx(reference.bwt())
    assert [value for value in ours["forgetting_per_period"].values() if value is not None] == pytest.approx(
        [value for value in reference.forgetting_per_period() if not np.isnan(value)]
    )


def test_matrix_from_scores():
    matrix = matrix_from_scores({(0, 0): 0.5, (1, 0): 0.6, (1, 1): 0.7}, n_periods=2)
    assert matrix.shape == (2, 2)
    assert np.isnan(matrix[0, 1])
    assert matrix[1, 0] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# D-033 + D-047: dual macro-F1 supported set includes Benign (7 classes)
# ---------------------------------------------------------------------------
def test_dual_macro_f1_reports_both_class_spaces():
    from analysis.metrics_cl import macro_f1_from_per_class
    from sklearn.metrics import f1_score

    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 3, 3, 3])
    result = dual_macro_f1(y_true, y_pred, class_ids=[0, 1, 2, 3], supported_ids=[2, 3])
    all_scores = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    supported_scores = f1_score(y_true, y_pred, labels=[2, 3], average=None, zero_division=0)
    expected_all = macro_f1_from_per_class({str(i): float(v) for i, v in enumerate(all_scores)})
    expected_supported = macro_f1_from_per_class({str(i): float(v) for i, v in enumerate(supported_scores)})
    assert result["macro_f1_all_classes"] == pytest.approx(float(expected_all))
    assert result["macro_f1_supported_set"] == pytest.approx(float(expected_supported))
    assert result["class_counts"]["0"] == 2 and result["class_counts"]["1"] == 2
    assert result["n_edges"] == 8
    assert result["n_classes_supported"] == 2


def test_dual_macro_f1_rejects_supported_outside_global():
    with pytest.raises(ValueError):
        dual_macro_f1(np.array([0]), np.array([0]), class_ids=[0, 1], supported_ids=[5])


def test_default_supported_set_is_seven_classes_including_benign():
    assert len(DEFAULT_SUPPORTED_SET) == 7
    assert "Benign" in DEFAULT_SUPPORTED_SET
    assert set(DEFAULT_SUPPORTED_SET) == {
        "Benign", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Shellcode"
    }


def test_config_supported_set_includes_benign_and_maps_to_ids(repo_root):
    from src.data.registry import load_dataset_spec
    from src.utils.config import load_config

    config = load_config("configs/eval.yaml", project_root=repo_root)
    names = supported_set_from_config(config.data)
    assert names == list(DEFAULT_SUPPORTED_SET)
    spec = load_dataset_spec("NF-UNSW-NB15-v3", project_root=repo_root)
    ids = supported_ids_from_names(names, spec.class_order)
    # frozen label space: Benign + 6 attack classes (D-033/D-047)
    assert len(ids) == 7 and 0 in ids


def test_supported_ids_from_names():
    order = ["Benign", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Shellcode"]
    ids = supported_ids_from_names(["DoS", "Exploits", "Shellcode"], order)
    assert ids == [1, 2, 6]
    with pytest.raises(ValueError):
        supported_ids_from_names(["Nope"], order)


def test_per_class_f1_keys_are_string_ids():
    scores = per_class_f1(np.array([0, 1, 1]), np.array([0, 0, 1]), [0, 1])
    assert set(scores) == {"0", "1"}


def test_forgetting_convention_string_documented():
    assert "forward-evaluation" in FORGETTING_CONVENTION
    assert "analysis.metrics_cl" in FORGETTING_CONVENTION


# ---------------------------------------------------------------------------
# T34 (bug-report #18): incremental evaluation must not raise on a partial last row
# ---------------------------------------------------------------------------
def test_partial_last_row_returns_partial_metrics_with_warning():
    R = np.array(
        [
            [0.80, np.nan, np.nan],
            [0.60, 0.70, np.nan],
            [0.50, np.nan, np.nan],
        ]
    )
    with pytest.warns(RuntimeWarning, match="partial metrics"):
        metrics = cl_metrics(R)
    assert metrics["partial"] is True
    assert metrics["evaluated_final_periods"] == 1
    assert metrics["final_average_macro_f1"] is None
    assert metrics["forgetting"] is None
    assert metrics["bwt"] is None
    assert metrics["last_row"][0] == pytest.approx(0.50)
    assert metrics["last_row"][1] is None
    assert "incremental evaluation" in metrics["convention"]


def test_entirely_unevaluated_last_row_is_partial_not_an_exception():
    R = np.array(
        [
            [0.80, np.nan, np.nan],
            [0.60, 0.70, np.nan],
            [np.nan, np.nan, np.nan],
        ]
    )
    with pytest.warns(RuntimeWarning):
        metrics = cl_metrics(R)
    assert metrics["partial"] is True
    assert metrics["evaluated_final_periods"] == 0
    assert metrics["final_average_macro_f1"] is None
