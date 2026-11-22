"""S5 CL metrics — hand-computed R matrices, forgetting/BWT, per-class tensor, labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis import metrics_cl as mc


def _hand_tensor() -> mc.RMetricTensor:
    R = np.array(
        [
            [0.80, np.nan, np.nan],
            [0.60, 0.90, np.nan],
            [0.50, 0.70, 0.60],
        ]
    )
    return mc.RMetricTensor(R, strategy="ft", model="m", seed=42, n_periods=3)


def test_final_forgetting_bwt_hand_computed() -> None:
    tensor = _hand_tensor()
    # final = mean(0.5, 0.7, 0.6) = 0.6
    assert tensor.final_macro_f1() == pytest.approx(0.6)
    # F_1 = max(0.8, 0.6) - 0.5 = 0.3 ; F_2 = max(0.9) - 0.7 = 0.2 ; mean = 0.25
    assert tensor.forgetting_per_period()[:2].tolist() == pytest.approx([0.3, 0.2])
    assert np.isnan(tensor.forgetting_per_period()[2])
    assert tensor.forgetting() == pytest.approx(0.25)
    # BWT_1 = 0.5-0.8 = -0.3 ; BWT_2 = 0.7-0.9 = -0.2 ; mean = -0.25
    assert tensor.bwt() == pytest.approx(-0.25)


def test_future_evaluation_cells_allowed_but_excluded_from_forgetting() -> None:
    """Writer evaluates every checkpoint on every period (Doc 2 S5 "for all i and j").

    Forward cells (j > i) are recorded; the forgetting best-so-far term may only
    use past checkpoints for that period (i >= j), per the Doc 2 S5 check.
    """
    R = np.array(
        [
            [0.80, 0.90, 0.85],  # M_1 evaluated on P1 (past), P2/P3 (forward)
            [0.60, 0.70, np.nan],
            [0.50, 0.90, 0.60],
        ]
    )
    tensor = mc.RMetricTensor(R, strategy="ft", model="m", seed=42)
    assert tensor.R[0, 1] == pytest.approx(0.90)  # future evaluation preserved
    assert tensor.final_macro_f1() == pytest.approx((0.50 + 0.90 + 0.60) / 3)
    # j=1: the only past checkpoint for P2 is M_2 (0.70). Using the forward cell
    # R[0,1]=0.90 would make forgetting 0.0 instead of -0.2.
    assert tensor.forgetting_per_period()[1] == pytest.approx(0.70 - 0.90)
    assert tensor.forgetting_per_period()[0] == pytest.approx(max(0.80, 0.60) - 0.50)
    # BWT is unaffected by forward cells (uses R[j,j] and R[T,j] only)
    assert tensor.bwt() == pytest.approx(((0.50 - 0.80) + (0.90 - 0.70)) / 2)


def test_nan_cells_are_treated_as_not_evaluated() -> None:
    tensor = mc.RMetricTensor(np.array([[0.8, np.nan], [0.6, 0.9]]))
    assert np.isnan(tensor.R[0, 1])


def test_reference_doc2_snippet_equivalence() -> None:
    """Cross-check against the Doc 2 §S5 reference implementation."""
    R = _hand_tensor().R
    T = R.shape[0]
    final, forgetting, bwt = mc.RMetricTensor(R).final_macro_f1(), None, None
    reference_forgetting = float(
        np.mean([np.nanmax(R[: T - 1, j]) - R[T - 1, j] for j in range(T - 1)])
    )
    reference_bwt = float(np.mean([R[T - 1, j] - R[j, j] for j in range(T - 1)]))
    assert final == pytest.approx(float(R[T - 1].mean()))
    assert _hand_tensor().forgetting() == pytest.approx(reference_forgetting)
    assert _hand_tensor().bwt() == pytest.approx(reference_bwt)


def test_per_class_tensor_and_dual_macro() -> None:
    R_class = np.full((2, 2, 2), np.nan)
    R_class[0, 0] = [0.9, 0.7]  # macro 0.8
    R_class[1, 0] = [0.8, 0.6]
    R_class[1, 1] = [0.95, 0.85]
    tensor = mc.RMetricTensor(
        np.array([[0.8, np.nan], [0.7, 0.9]]),
        R_class=R_class,
        classes=("A", "B"),
        strategy="ft",
        model="m",
        seed=42,
    )
    assert tensor.validate() == []
    assert tensor.class_macro_matrix().tolist()[0][0] == pytest.approx(0.8)
    # D-033: supported-set macro excludes B
    assert tensor.class_macro_matrix(exclude_classes=["B"]).tolist()[0][0] == pytest.approx(0.9)
    assert tensor.final_macro_f1_by_class(exclude_classes=["B"]) == pytest.approx(float(np.mean([0.8, 0.95])))
    with pytest.raises(mc.CLMetricsError):
        tensor.class_macro_matrix(exclude_classes=["A", "B"], nan_policy="raise")


def test_validate_catches_macro_inconsistency() -> None:
    R_class = np.full((2, 2, 2), np.nan)
    R_class[0, 0] = [0.9, 0.7]
    R_class[1, 0] = [0.8, 0.6]
    R_class[1, 1] = [0.95, 0.85]
    tensor = mc.RMetricTensor(
        np.array([[0.5, np.nan], [0.7, 0.9]]),  # deliberately wrong R[0,0]
        R_class=R_class,
        classes=("A", "B"),
    )
    issues = tensor.validate()
    assert any("R[0,0]" in issue for issue in issues)


def test_later_drop_labels_hand_computed() -> None:
    R_class = np.full((3, 3, 1), np.nan)
    R_class[0, 0, 0] = 0.80
    R_class[1, 0, 0] = 0.70
    R_class[1, 1, 0] = 0.90
    R_class[2, 0, 0] = 0.74
    R_class[2, 1, 0] = 0.80
    R_class[2, 2, 0] = 0.60
    tensor = mc.RMetricTensor(
        np.full((3, 3), np.nan),
        R_class=R_class,
        classes=("A",),
        strategy="ft",
        model="m",
        seed=42,
    )
    labels = mc.later_drop_label_tensor(R_class, delta=0.05)
    assert labels[0, 0, 0] == 1.0  # next update drop 0.10 >= 0.05
    assert labels[1, 0, 0] == 0.0  # 0.70 -> 0.74 is an increase
    assert labels[1, 1, 0] == 1.0  # final 0.90 -> 0.80 drop 0.10
    assert np.isnan(labels[2, 0, 0])  # last checkpoint has no future observation
    rows = mc.later_drop_rows(tensor, delta=0.05)
    first = [row for row in rows if row["checkpoint"] == 1 and row["eval_period"] == 1][0]
    assert first["label"] == 1
    assert first["future_checkpoint"] == 2
    assert first["future_f1"] == pytest.approx(0.70)
    assert first["drop"] == pytest.approx(0.10)


def test_later_drop_any_later_variant_and_delta_boundary() -> None:
    R_class = np.full((3, 3, 1), np.nan)
    R_class[0, 0, 0] = 0.80
    R_class[1, 0, 0] = 0.81
    R_class[1, 1, 0] = 0.90
    R_class[2, 0, 0] = 0.75
    R_class[2, 1, 0] = 0.80
    R_class[2, 2, 0] = 0.60
    any_later = mc.later_drop_label_tensor(R_class, delta=0.05, horizon="any_later")
    assert any_later[0, 0, 0] == 1.0  # final drop 0.05 exactly -> labelled (epsilon tolerance)
    next_or_final = mc.later_drop_label_tensor(R_class, delta=0.05)
    assert next_or_final[0, 0, 0] == 1.0  # final is also a candidate in next_or_final
    with pytest.raises(ValueError):
        mc.later_drop_label_tensor(R_class, horizon="post_hoc_variant")


def test_best_so_far_and_forgetting_fj() -> None:
    R_class = np.full((3, 3, 1), np.nan)
    R_class[0, 0, 0] = 0.8
    R_class[1, 0, 0] = 0.6
    R_class[2, 0, 0] = 0.7
    best = mc.best_so_far_f1(R_class)
    assert best[0, 0, 0] == pytest.approx(0.8)
    assert best[1, 0, 0] == pytest.approx(0.8)
    assert best[2, 0, 0] == pytest.approx(0.8)
    forgetting = mc.forgetting_fj(R_class)
    assert forgetting[1, 0, 0] == pytest.approx(0.2)
    assert forgetting[2, 0, 0] == pytest.approx(0.1)


def test_record_parsing_and_matrix_construction() -> None:
    payload = {
        "strategy": "ft",
        "model": "egraphsage",
        "seed": 42,
        "checkpoint": 2,
        "lane": "canonical",
        "per_period": {
            "1": {"macro_f1": 0.6, "per_class_f1": {"A": 0.7, "B": 0.5}, "n_test": 20},
            "2": {"macro_f1": 0.9, "per_class_f1": {"A": 0.95, "B": 0.85}},
        },
    }
    records = mc.records_from_payload(payload)
    assert len(records) == 2
    assert records[0].checkpoint == 2 and records[0].eval_period == 1
    tensor = mc.RMetricTensor.from_records(records, classes=("A", "B"), n_periods=2)
    assert tensor.R[1, 0] == pytest.approx(0.6)
    assert tensor.R[1, 1] == pytest.approx(0.9)
    assert np.isnan(tensor.R[0, 1])


def test_conflicting_duplicate_cells_raise() -> None:
    base = dict(strategy="ft", model="m", seed=42, checkpoint=1, eval_period=1)
    records = [
        mc.CLEvalRecord(**base, macro_f1=0.5),
        mc.CLEvalRecord(**base, macro_f1=0.7),
    ]
    with pytest.raises(mc.DuplicateCellError):
        mc.RMetricTensor.from_records(records, n_periods=1)
    tensor = mc.RMetricTensor.from_records(records, n_periods=1, on_conflict="mean")
    assert tensor.R[0, 0] == pytest.approx(0.6)


def test_future_evaluation_record_is_accepted() -> None:
    # Doc 2 S5 defines R[i, j] for all i and j: M_1 evaluated on P_2 is a
    # legitimate forward-evaluation cell (recorded, excluded from forgetting).
    record = mc.CLEvalRecord(strategy="ft", model="m", seed=0, checkpoint=1, eval_period=2, macro_f1=0.5)
    assert record.eval_period == 2
    tensor = mc.RMetricTensor.from_records([record], n_periods=2)
    assert tensor.R[0, 1] == pytest.approx(0.5)


def test_collect_records_from_json_file(tmp_path: Path) -> None:
    payload = {
        "strategy": "replay",
        "model": "egraphsage",
        "seed": 42,
        "checkpoint": 1,
        "eval_period": 1,
        "macro_f1": 0.55,
        "per_class_f1": {"A": 0.6, "B": 0.5},
    }
    path = tmp_path / "eval_M1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    records = mc.collect_records(path)
    assert len(records) == 1 and records[0].strategy == "replay"
    assert len(mc.collect_records(tmp_path)) == 1


def test_per_class_f1_roundtrip(tmp_path: Path) -> None:
    R_class = np.full((2, 2, 2), np.nan)
    R_class[0, 0] = [0.9, 0.7]
    R_class[1, 0] = [0.8, 0.6]
    R_class[1, 1] = [0.95, 0.85]
    tensor = mc.RMetricTensor(
        np.array([[0.8, np.nan], [0.7, 0.9]]),
        R_class=R_class,
        classes=("A", "B"),
        strategy="ft",
        model="m",
        seed=42,
    )
    npz_path = tmp_path / "per_class_f1.npz"
    csv_path = tmp_path / "per_class_f1.csv"
    written = mc.save_per_class_f1(tensor, npz_path, csv_path)
    assert written == [npz_path, csv_path]
    with np.load(npz_path) as data:
        assert set(data.files) == set(mc.PER_CLASS_F1_NPZ_KEYS)
        assert data["R_class"].shape == (2, 2, 2)
    loaded = mc.load_per_class_f1(npz_path)
    assert loaded.classes == ("A", "B")
    assert loaded.R[1, 1] == pytest.approx(0.9)
    assert loaded.R_class[0, 0, 0] == pytest.approx(0.9)
    csv_text = csv_path.read_text(encoding="utf-8").splitlines()
    assert csv_text[0].split(",") == list(mc.PER_CLASS_F1_CSV_COLUMNS)
    assert len(csv_text) == 1 + 6  # 3 defined cells x 2 classes


def test_load_cl_matrix_json_from_s5_writer(tmp_path: Path) -> None:
    payload = {
        "strategy": "ft",
        "model": "egraphsage",
        "seed": 42,
        "class_order": ["A", "B"],
        # M_1 evaluated on P1 (past) and P2 (forward, Doc 2 S5 "for all i and j")
        "matrix": {"all_classes": [[0.8, 0.5], [0.7, 0.9]]},
        "dual_macro_f1_per_cell": {
            "1": {
                "1": {"per_class_f1": {"0": 0.9, "1": 0.7}},
                "2": {"per_class_f1": {"0": 0.6, "1": 0.4}},
            },
            "2": {
                "1": {"per_class_f1": {"0": 0.8, "1": 0.6}},
                "2": {"per_class_f1": {"0": 0.95, "1": 0.85}},
            },
        },
    }
    path = tmp_path / "cl_matrix.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    tensor = mc.load_cl_matrix_json(path)
    assert (tensor.strategy, tensor.model, tensor.seed) == ("ft", "egraphsage", 42)
    assert tensor.classes == ("A", "B")
    assert tensor.R[0, 1] == pytest.approx(0.5)  # forward evaluation preserved
    assert tensor.R[1, 0] == pytest.approx(0.7)
    assert tensor.R_class[1, 1, 0] == pytest.approx(0.95)
    assert tensor.validate() == []
    assert tensor.final_macro_f1() == pytest.approx(0.8)  # mean(0.7, 0.9)
    # j=0: only past checkpoint M_1 counts; max(0.8) - 0.7 = 0.1
    assert tensor.forgetting() == pytest.approx(0.1)


def test_load_cl_metrics_csv_from_s5_writer(tmp_path: Path) -> None:
    path = tmp_path / "cl_metrics.csv"
    path.write_text(
        "strategy,model,seed,metric,final_average_macro_f1,forgetting,bwt,note\n"
        "ft,egraphsage,42,all_classes,0.6,0.2,-0.1,\n",
        encoding="utf-8",
    )
    frame = mc.load_cl_metrics_csv(path)
    assert frame.iloc[0]["metric"] == "all_classes"
    assert frame.iloc[0]["forgetting"] == pytest.approx(0.2)
    bad_path = tmp_path / "wrong.csv"
    bad_path.write_text("strategy,model\nft,egraphsage\n", encoding="utf-8")
    with pytest.raises(mc.CLMetricsError):
        mc.load_cl_metrics_csv(bad_path)


def test_aggregate_seeds() -> None:
    first = mc.RMetricTensor(np.array([[0.8, np.nan], [0.6, 0.9]]), strategy="ft", model="m", seed=42)
    second = mc.RMetricTensor(np.array([[0.7, np.nan], [0.5, 0.8]]), strategy="ft", model="m", seed=43)
    aggregated = mc.aggregate_seeds([first, second])
    assert aggregated.R[0, 0] == pytest.approx(0.75)
    assert aggregated.R[1, 1] == pytest.approx(0.85)
    assert aggregated.seed is None


def test_macro_f1_from_per_class_exclusions() -> None:
    values = {"A": 0.9, "B": 0.7, "Worms": 0.0}
    assert mc.macro_f1_from_per_class(values) == pytest.approx((0.9 + 0.7 + 0.0) / 3)
    assert mc.macro_f1_from_per_class(values, exclude_classes=["Worms"]) == pytest.approx(0.8)
    with pytest.raises(mc.CLMetricsError):
        mc.macro_f1_from_per_class({"A": 0.9, "B": float("nan")}, nan_policy="raise")
