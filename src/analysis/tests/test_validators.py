"""T19 validators — schema/invariant checks on synthetic-demo and real partial artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import artifacts, pipeline_fixtures, synthetic, tables, validators

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def demo_tables(tmp_path_factory):
    """Generate the full synthetic artifact package once (incl. S5 writer files)."""
    layout = artifacts.ArtifactLayout(tmp_path_factory.mktemp("results"))
    bundle = synthetic.make_bundle(models=("egraphsage",))
    pipeline_fixtures.write_synthetic_s6_s7(layout)
    tables.generate_all(
        cl_tensors=bundle.cl_tensors,
        real_drift=bundle.real_drift,
        model_null_noise=bundle.model_null_noise,
        explainer_repeat_noise=bundle.explainer_repeat_noise,
        retention=bundle.retention,
        neighbourhood=bundle.neighbourhood,
        ear_comparison=bundle.ear_comparison,
        ear_ablations=bundle.ear_ablations,
        layout=layout,
        supported_classes=synthetic.SUPPORTED_CLASSES,
    )
    return layout.tables_root


def test_synthetic_demo_artifacts_validate_cleanly(demo_tables: Path) -> None:
    report = validators.validate_artifact_set(demo_tables, stage="all")
    assert report.overall == "pass", [f.as_dict() for f in report.findings if f.status != "pass"]
    assert report.counts["fail"] == 0
    assert report.counts["pass"] > 50
    assert 0 == report.counts["warn"]


def test_real_partial_artifacts_validate_without_failure() -> None:
    """The M_1-only partial S5 artifact is a legitimate intermediate state."""
    real_tables = REPO_ROOT / "results" / "tables"
    if not (real_tables / "cl_matrix.json").is_file():
        pytest.skip("partial S5 artifacts not present in this checkout")
    report = validators.validate_artifact_set(real_tables, stage="s5")
    assert report.overall != "fail", [f.as_dict() for f in report.failures]
    matrix_check = [f for f in report.findings if f.check == "cl_matrix.complete"]
    assert matrix_check and matrix_check[0].status in {"pass", "warn"}
    metrics_check = [f for f in report.findings if f.check == "cl_metrics.metric_rows"]
    assert metrics_check and metrics_check[0].status == "pass"


def test_missing_required_artifact_is_a_failure(tmp_path: Path) -> None:
    report = validators.validate_artifact_set(tmp_path, stage="s5")
    assert report.overall == "fail"
    assert any(f.check == "required.present.cl_matrix.json" for f in report.failures)


def test_rq3_future_leakage_is_detected(tmp_path: Path) -> None:
    root = tmp_path
    frame = pd.DataFrame(
        {
            "strategy": ["ft"],
            "cl_model": ["m"],
            "checkpoint_t": [2],
            "class": ["A"],
            "label": [1],
            "mean_excess_drift": [0.2],
            "p90_excess_drift": [0.3],
            "future_f1": [0.9],  # leakage!
        }
    )
    frame.to_csv(root / "rq3_features.csv", index=False)
    report = validators.validate_artifact(root, "rq3_features.csv")
    check = [f for f in report.findings if f.check == "rq3_features.no_future_leakage"]
    assert check and check[0].status == "fail"


def test_noise_floor_protocol_values_are_enforced(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "channel": ["model_null"],
            "class": ["A"],
            "n_null_pairs": [10],
            "quantile": [0.90],  # wrong frozen quantile
            "q95": [0.5],
            "threshold_source": ["class"],
        }
    )
    frame.to_csv(tmp_path / "noise_floor.csv", index=False)
    report = validators.validate_artifact(tmp_path, "noise_floor.csv")
    quantile = [f for f in report.findings if f.check == "noise_floor.quantile"]
    assert quantile and quantile[0].status == "fail"
    multiples = [f for f in report.findings if f.check == "noise_floor.model_null_multiples"]
    assert multiples and multiples[0].status == "pass"  # 10 is a multiple of R=5


def test_rq4_h4_flag_is_recomputed(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "model": ["egraphsage", "egraphsage"],
            "variant": ["replay", "ear_a"],
            "seed": [42, ""],
            "drift_mean": [0.3, 0.2],
            "macro_f1": [0.62, 0.615],
            "row_kind": ["run", "h4_delta_vs_replay"],
            "h4_supported": [float("nan"), False],  # true decision should be True (lower drift, >= -1pt)
            "drift_diff_vs_replay": [float("nan"), -0.1],
            "f1_diff_vs_replay": [float("nan"), -0.005],
        }
    )
    frame.to_csv(tmp_path / "rq4_comparison.csv", index=False)
    report = validators.validate_artifact(tmp_path, "rq4_comparison.csv")
    h4 = [f for f in report.findings if f.check == "rq4_comparison.h4_rule"]
    assert h4 and h4[0].status == "fail"


def test_validator_report_is_json_serialisable(demo_tables: Path) -> None:
    report = validators.validate_artifact_set(demo_tables, stage="s8")
    payload = report.as_dict()
    json.dumps(payload)  # must not raise
    assert payload["overall"] in {"pass", "warn", "fail"}
    assert "findings" in payload and payload["findings"]


def test_unknown_artifact_and_stage_raise() -> None:
    with pytest.raises(validators.ValidatorError):
        validators.validate_artifact(".", "not_an_artifact.csv")
    with pytest.raises(validators.ValidatorError):
        validators.validate_artifact_set(".", stage="s99")
