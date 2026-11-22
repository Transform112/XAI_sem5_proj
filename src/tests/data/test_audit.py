"""Tests for the data-audit suite: structure, quality, scorecard orchestration.

The end-to-end test runs the full one-command audit on a tiny synthetic NF-v3
CSV and asserts (a) the required artifacts exist, (b) every pass/fail threshold
is read from configuration (flipping a threshold flips the check), and (c) the
D-020 structure facts are recorded per period.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.audit import run_audit
from src.data.clean import clean_frame, load_raw_frame
from src.data.periods import assign_periods
from src.data.preprocess import Preprocessor
from src.data.quality import quality_audit
from src.data.registry import load_dataset_spec
from src.data.splits import build_splits
from src.data.structure import structure_report
from src.tests.data.helpers import make_nf_row, write_nf_csv


@pytest.fixture(scope="module")
def spec():
    return load_dataset_spec("NF-UNSW-NB15-v3")


def _rows(n: int = 240) -> list[dict[str, str]]:
    base_ts = 1424242193000
    rows: list[dict[str, str]] = []
    labels = ["Benign", "Benign", "Benign", "Exploits", "Fuzzers", "Worms"]
    for index in range(n):
        label = labels[index % len(labels)]
        binary = "0" if label == "Benign" else "1"
        rows.append(
            make_nf_row(
                FLOW_START_MILLISECONDS=str(base_ts + index * 1000),
                FLOW_END_MILLISECONDS=str(base_ts + index * 1000 + 10),
                IPV4_SRC_ADDR=f"10.0.0.{1 + index % 6}",
                IPV4_DST_ADDR=f"10.0.1.{1 + index % 4}",
                L4_SRC_PORT=str(40000 + index % 7),
                L4_DST_PORT=str(80 + index % 3),
                PROTOCOL="6" if index % 2 == 0 else "17",
                IN_BYTES=str(100 + index % 50),
                OUT_BYTES=str(200 + index % 40),
                IN_PKTS=str(1 + index % 5),
                Label=binary,
                Attack=label,
            )
        )
    # self-loop and repeated host pair
    rows[0]["IPV4_DST_ADDR"] = rows[0]["IPV4_SRC_ADDR"]
    return rows


@pytest.fixture()
def cleaned(spec, tmp_path: Path):
    csv_path = write_nf_csv(tmp_path / spec.name / f"{spec.name}.csv", _rows())
    raw = load_raw_frame(spec, csv_path)
    clean = clean_frame(raw, spec)
    periods = assign_periods(clean.frame, n_periods=2)
    return {
        "csv_path": csv_path,
        "frame": periods.frame,
        "clean": clean,
        "periods": periods,
    }


@pytest.fixture()
def splits(cleaned):
    return build_splits(
        cleaned["frame"],
        fractions={"train": 0.7, "val": 0.1, "test": 0.2},
        seed=42,
        min_groups_per_split=1,
        min_share=0.0,
    )


def _audit_config() -> dict:
    """Tiny-data audit config: same schema as configs/audit.yaml, relaxed values."""
    return {
        "seed": 42,
        "dataset": "NF-UNSW-NB15-v3",
        "periods": {"count": 2, "strategy": "equal_counts", "min_flows_per_period": 10},
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
            "host_pair_disjoint": {"enabled": True, "min_groups_per_split": 1, "min_share": 0.0},
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
            "evaluate_per_period": True,
            "union_diagnostics": True,
            "small_host_flag_threshold": 100,
            "giant_component_undirected": True,
            "degree_mode": "undirected_pair",
            "repeated_host_pairs": True,
            "self_loops": True,
        },
        "quality": {
            "ks_sample_rows": 200,
            "ks_minimum_rows": 10,
            "mutual_information": True,
            "mi_sample_rows": 200,
            "single_feature_auc": True,
            "auc_sample_rows": 200,
            "adversarial_sample_rows": 200,
            "adversarial_model": "hist_gradient_boosting",
            "adversarial_max_iter": 10,
            "adversarial_cv": 2,
            "constant_feature_unique_max": 1,
            "label_conflict_key": "near_duplicate",
        },
        "graph_lift": {"enabled": False},
        "thresholds": {
            "nan_inf_fraction_max": 0.0,
            "near_duplicate_cross_split_max": 0.0,
            "duplicate_fraction_max": 0.001,
            "label_conflict_fraction_max": 1.0,
            "constant_feature_count_max": 1000,
            "adversarial_auc_within_period_max": 2.0,
            "median_host_degree_min": 0.0,
            "giant_component_fraction_min": 0.0,
            "host_recurrence_min": 0.0,
            "repeated_host_pair_fraction_min": 0.0,
            "max_self_loop_fraction_max": 1.0,
            "graph_lift_macro_f1_min": 1.0,
            "graph_lift_vs_randomised_min": 1.0,
        },
    }


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
def test_structure_report_global_facts(cleaned, spec):
    report = structure_report(
        cleaned["frame"], class_map=cleaned["clean"].class_map
    )
    assert report["global"]["n_edges"] == len(cleaned["frame"])
    assert report["global"]["n_hosts"] == 10  # 6 source + 4 destination addresses
    assert report["global"]["components"]["n_components"] >= 1
    assert 0.0 < report["global"]["components"]["giant_fraction"] <= 1.0
    assert report["global"]["self_loops"]["count"] == 1
    assert report["global"]["repeated_host_pairs"]["fraction_pairs"] > 0.0
    assert report["global"]["degree"]["median"] > 0


def test_structure_report_per_period_turnover_and_recurrence(cleaned):
    report = structure_report(cleaned["frame"], class_map=cleaned["clean"].class_map)
    # JSON-ready report: period keys are 1-based-month strings in the JSON sense
    assert set(report["periods"]) == {"0", "1"}
    for period in ("0", "1"):
        block = report["periods"][period]
        assert block["hosts"] > 0
        assert 0.0 <= block["repeated_pair_fraction"] <= 1.0
        assert block["empty"] is False
        assert "hosts_union_with_previous" in block
    # period 1 has no previous period -> recurrence undefined/None
    assert report["periods"]["0"]["host_recurrence"] is None
    assert report["periods"]["1"]["host_recurrence"] is not None
    assert report["host_recurrence"]["min"] is not None


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
def test_quality_audit_metrics(cleaned, splits, spec):
    frame = cleaned["frame"]
    train = frame.iloc[splits.masks[0]["train"]]
    pre = Preprocessor(spec, config={}).fit(train)
    config = _audit_config()["quality"]
    quality = quality_audit(frame, splits, pre, spec, config=config, seed=42)
    assert quality["nan_inf"]["fraction"] == 0.0
    assert quality["nan_inf"]["total"] == 0
    assert len(quality["per_feature"]) == len(pre.feature_names)
    within_auc = quality["adversarial"]["within_period_auc"]
    assert within_auc is None or 0.0 <= within_auc <= 1.0
    assert 0.0 <= quality["label_conflict"]["fraction"] <= 1.0
    assert quality["label_conflict"]["n_groups"] > 0
    assert quality["constant_features"]["count"] >= 0
    within = quality["ks"]["within_period"]
    assert within["n_features"] == len(pre.feature_names)
    assert quality["single_feature_auc"]["top"]


# ---------------------------------------------------------------------------
# End-to-end one-command audit
# ---------------------------------------------------------------------------
def test_run_audit_writes_required_artifacts(cleaned, spec, tmp_path):
    out_dir = tmp_path / "audit_out"
    splits_dir = tmp_path / "splits"
    scorecard = run_audit(
        spec=spec,
        csv_path=cleaned["csv_path"],
        config=_audit_config(),
        out_dir=out_dir,
        splits_dir=splits_dir,
        dataset=spec.name,
        seed=42,
        with_pilot=False,
    )
    for name in (
        "scorecard.json",
        "scorecard.md",
        "feature_manifest.csv",
        "split_manifest.json",
        "structure.json",
        "preprocessor_state.json",
        "manifest.json",
    ):
        assert (out_dir / name).is_file(), f"missing artifact {name}"
    assert list(splits_dir.glob("period_*_masks.npz"))

    manifest = json.loads((out_dir / "feature_manifest.csv").read_text(encoding="utf-8")) if False else None
    feature_manifest = pd.read_csv(out_dir / "feature_manifest.csv")
    assert len(feature_manifest) == 55
    split_manifest = json.loads((out_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert split_manifest["frozen_hash"] and len(split_manifest["frozen_hash"]) == 64
    assert split_manifest["mode"] == "grouped"
    scorecard_json = json.loads((out_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard_json["dataset"] == spec.name
    assert scorecard_json["checks"]
    assert scorecard["status"] in {"pass", "flag", "fail"}
    structure = json.loads((out_dir / "structure.json").read_text(encoding="utf-8"))
    assert structure["periods"]["1"]["host_turnover"] is not None


def test_run_audit_thresholds_are_read_from_config(cleaned, spec, tmp_path):
    config = _audit_config()
    # nan/inf fraction is 0.0 on clean data; an impossible threshold must flip
    # the check to fail, proving the value is read from config.
    config["thresholds"]["nan_inf_fraction_max"] = -0.1
    scorecard = run_audit(
        spec=spec,
        csv_path=cleaned["csv_path"],
        config=config,
        out_dir=tmp_path / "audit_out",
        splits_dir=tmp_path / "splits",
        dataset=spec.name,
        seed=42,
        with_pilot=False,
    )
    check = next(
        check for check in scorecard["checks"] if check["name"] == "nan_inf_fraction"
    )
    assert check["status"] == "fail"
    assert check["threshold"] == -0.1
    assert scorecard["status"] == "fail"
