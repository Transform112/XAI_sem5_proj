"""T20 S6/S7 validator tests — anchor sets, retention table, explanation cache index."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import artifacts, pipeline_fixtures, validators


@pytest.fixture(scope="module")
def pipeline_root(tmp_path_factory) -> artifacts.ArtifactLayout:
    layout = artifacts.ArtifactLayout(tmp_path_factory.mktemp("results"))
    pipeline_fixtures.write_synthetic_s6_s7(layout)
    return layout


def test_anchor_artifacts_validate(pipeline_root: artifacts.ArtifactLayout) -> None:
    report = validators.ValidationReport(results_tables="synthetic", stage="s6")
    validators.validate_anchor_artifacts(report, pipeline_root.results_root / "anchors")
    assert report.overall == "pass", [finding.as_dict() for finding in report.findings if finding.status != "pass"]
    checks = {finding.check for finding in report.findings}
    assert {"anchors.periods", "anchors.structure"} <= checks


def test_anchor_validation_detects_corruption(pipeline_root: artifacts.ArtifactLayout, tmp_path: Path) -> None:
    import torch

    corrupted = tmp_path / "anchors"
    corrupted.mkdir()
    payload = torch.load(pipeline_root.results_root / "anchors" / "period_1.pt", weights_only=False)
    payload["anchors"][0]["confidence"] = 0.5  # below docs/05 §7 threshold
    payload["anchors"][1]["target_edge"] = torch.tensor([99999])  # target not in subgraph
    torch.save(payload, corrupted / "period_1.pt")
    report = validators.ValidationReport(results_tables="synthetic", stage="single")
    validators.validate_anchor_artifacts(report, corrupted)
    assert report.overall == "fail"
    detail = " ".join(finding.detail for finding in report.findings)
    assert "confidence" in detail and "target edge" in detail


def test_retained_fraction_validates_and_detects_bad_rows(pipeline_root: artifacts.ArtifactLayout, tmp_path: Path) -> None:
    good = pipeline_root.results_root / "anchors" / "retained_fraction.csv"
    report = validators.ValidationReport(results_tables="synthetic", stage="single")
    validators.validate_retained_fraction_csv(report, good)
    assert report.overall == "pass"
    bad = tmp_path / "retained_fraction.csv"
    text = good.read_text(encoding="utf-8").replace("0.", "1.", 1)  # fraction > 1 / inconsistent
    bad.write_text(text, encoding="utf-8")
    bad_report = validators.ValidationReport(results_tables="synthetic", stage="single")
    validators.validate_retained_fraction_csv(bad_report, bad)
    assert bad_report.overall == "fail"


def test_explanation_index_validates(pipeline_root: artifacts.ArtifactLayout) -> None:
    index = pipeline_root.results_root / "explanations" / "index.csv"
    report = validators.ValidationReport(results_tables="synthetic", stage="s7")
    validators.validate_explain_index_csv(report, index)
    assert report.overall == "pass", [finding.as_dict() for finding in report.findings if finding.status != "pass"]
    checks = {finding.check for finding in report.findings}
    assert {"explain_index.entries", "explain_index.entry_files", "explain_index.entry_arrays", "explain_index.provenance_columns"} <= checks


def test_explanation_index_detects_missing_entry_files(pipeline_root: artifacts.ArtifactLayout, tmp_path: Path) -> None:
    import shutil

    cache = tmp_path / "explanations"
    shutil.copytree(pipeline_root.results_root / "explanations", cache)
    index = cache / "index.csv"
    rows = index.read_text(encoding="utf-8").splitlines()
    # drop one npz file referenced by the second row
    first_entry = rows[1].split(",")
    (cache / first_entry[8]).unlink()
    report = validators.ValidationReport(results_tables="synthetic", stage="single")
    validators.validate_explain_index_csv(report, index)
    assert report.overall == "fail"
    assert any("missing=" in finding.detail for finding in report.findings)


def test_explanation_index_fails_when_evidence_columns_missing(pipeline_root: artifacts.ArtifactLayout, tmp_path: Path) -> None:
    import json
    import shutil

    cache = tmp_path / "explanations"
    shutil.copytree(pipeline_root.results_root / "explanations", cache)
    index = cache / "index.csv"
    row = index.read_text(encoding="utf-8").splitlines()[1].split(",")
    frame = pd.read_csv(index)
    frame = frame.drop(columns=["fidelity_sufficiency", "fidelity_comprehensiveness", "fidelity_over_random", "runtime_s"])
    frame.to_csv(index, index=False)
    report = validators.ValidationReport(results_tables="synthetic", stage="single")
    validators.validate_explain_index_csv(report, index)
    assert report.counts["fail"] >= 1  # D-048(c): per-entry fidelity+runtime columns required
    assert any("provenance_columns" in finding.check and finding.status == "fail" for finding in report.findings)


def test_dual_index_copies_must_match(pipeline_root: artifacts.ArtifactLayout, tmp_path: Path) -> None:
    import shutil

    results_root = tmp_path / "results"
    (results_root / "tables").mkdir(parents=True)
    (results_root / "explanations").mkdir(parents=True)
    canonical = results_root / "tables" / "explain_index.csv"
    operational = results_root / "explanations" / "index.csv"
    shutil.copyfile(pipeline_root.results_root / "explanations" / "index.csv", operational)
    shutil.copyfile(operational, canonical)
    report = validators.ValidationReport(results_tables=str(results_root / "tables"), stage="s7")
    validators.validate_dual_index_consistency(report, results_root)
    assert report.overall == "pass"
    # break one key in the canonical copy -> FAIL
    lines = canonical.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace(lines[1].split(",")[0], "0" * 64)
    canonical.write_text("\n".join(lines) + "\n", encoding="utf-8")
    broken = validators.ValidationReport(results_tables=str(results_root / "tables"), stage="s7")
    validators.validate_dual_index_consistency(broken, results_root)
    assert broken.overall == "fail"


def test_stage_requirements_include_s6_s7() -> None:
    assert "s6" in validators.STAGE_REQUIREMENTS and "s7" in validators.STAGE_REQUIREMENTS
    assert validators.STAGE_ORDER == ("s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12")
    assert any(str(spec).startswith("anchors/") for spec in validators.STAGE_REQUIREMENTS["s6"])
    assert any("index.csv" in str(spec) for spec in validators.STAGE_REQUIREMENTS["s7"])


def test_canonical_anchor_retention_schema(pipeline_root: artifacts.ArtifactLayout) -> None:
    path = pipeline_root.results_root / "tables" / "anchor_retention.csv"
    assert path.is_file()
    report = validators.ValidationReport(results_tables="synthetic", stage="single")
    validators.validate_anchor_retention_csv(report, path)
    assert report.overall == "pass", [finding.as_dict() for finding in report.findings if finding.status != "pass"]


def test_noise_cache_schema_and_repeat_counts(pipeline_root: artifacts.ArtifactLayout) -> None:
    report = validators.ValidationReport(results_tables="synthetic", stage="s8")
    validators.validate_noise_cache_artifacts(report, pipeline_root.results_root)
    assert report.overall == "pass", [finding.as_dict() for finding in report.findings if finding.status != "pass"]
    checks = {finding.check for finding in report.findings}
    assert {"noise_cache.model_null.repeat_counts", "noise_cache.explainer_repeat.repeat_counts", "noise_cache.combined.schema"} <= checks


def test_noise_cache_detects_wrong_repeat_count(pipeline_root: artifacts.ArtifactLayout, tmp_path: Path) -> None:
    import shutil

    import pandas as pd

    results_root = tmp_path / "results"
    shutil.copytree(pipeline_root.results_root / "explanations_noise", results_root / "explanations_noise")
    index = results_root / "explanations_noise" / "model_null" / "index.csv"
    frame = pd.read_csv(index)
    trimmed = frame[frame.duplicated(subset=["anchor_id", "explainer"], keep="first")].head(0)  # noqa: F841
    frame.iloc[:1].to_csv(index, mode="w", index=False)  # one entry per anchor -> wrong R=5 count
    report = validators.ValidationReport(results_tables="synthetic", stage="s8")
    validators.validate_noise_cache_artifacts(report, results_root)
    assert report.overall == "fail"


def test_explain_index_relocates_remote_absolute_paths(tmp_path: Path) -> None:
    """A downloaded Kaggle index keeps /kaggle/... paths; the validator must relocate them."""
    import json

    import numpy as np

    results = tmp_path / "results"
    (results / "tables").mkdir(parents=True)
    entries = results / "explanations" / "entries"
    entries.mkdir(parents=True)
    np.savez(entries / "entry.npz", attributions=np.zeros((3, 2), dtype=float))
    (entries / "entry.json").write_text(
        json.dumps(
            {
                "explainer": "ig",
                "anchor_id": "a1",
                "checkpoint_period": 1,
                "anchor_period": 1,
                "fidelity": {"comprehensiveness": 0.5},
                "runtime_s": 0.2,
            }
        ),
        encoding="utf-8",
    )
    remote_npz = "/kaggle/working/stage/results/text/explanations/entries/entry.npz"
    remote_json = "/kaggle/working/stage/results/text/explanations/entries/entry.json"
    row = {
        "key": "k1",
        "checkpoint_sha": "sha-ckpt",
        "anchor_id": "a1",
        "explainer": "ig",
        "seed": 42,
        "k": 5,
        "config_sha": "sha-config",
        "created_at_utc": "2026-11-02T00:00:00+00:00",
        "checkpoint_j": 1,
        "checkpoint_t": 1,
        "npz_path": remote_npz,
        "json_path": remote_json,
        "fidelity_sufficiency": 0.5,
        "fidelity_comprehensiveness": 0.4,
        "fidelity_over_random": 0.1,
        "runtime_s": 0.2,
    }
    index = results / "tables" / "explain_index.csv"
    pd.DataFrame([row]).to_csv(index, index=False)
    report = validators.ValidationReport(results_tables=str(results / "tables"), stage="s7")
    validators.validate_explain_index_csv(report, index)
    checks = {finding.check: finding for finding in report.findings}
    assert checks["explain_index.entry_files"].status == "pass", checks["explain_index.entry_files"].detail
    assert report.overall == "pass", [finding.as_dict() for finding in report.findings if finding.status != "pass"]


def test_anchors_manifest_relocates_remote_absolute_paths(tmp_path: Path) -> None:
    """Downloaded Kaggle anchors manifest records /kaggle/... paths; relocate under results/."""
    results = tmp_path / "results"
    anchors = results / "anchors"
    anchors.mkdir(parents=True)
    payload_entries = {}
    for period in (1, 2):
        anchor_file = anchors / f"period_{period}.pt"
        anchor_file.write_bytes(b"fake-anchor-set")
        payload_entries[str(period)] = {"path": f"/kaggle/working/stage/results/text/anchors/period_{period}.pt", "sha256": "x"}
    (results / "tables").mkdir(parents=True)
    retention = results / "tables" / "anchor_retention.csv"
    retention.write_text("strategy,model,class,checkpoint_t,retained_fraction\nft,egraphsage,Benign,1,1.0\n", encoding="utf-8")
    manifest = anchors / "anchors_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "NF-UNSW-NB15-v3",
                "strategy": "ft",
                "model": "egraphsage",
                "seed": 42,
                "periods": payload_entries,
                "canonical_anchor_retention": "/kaggle/working/stage/results/text/tables/anchor_retention.csv",
            }
        ),
        encoding="utf-8",
    )
    report = validators.ValidationReport(results_tables=str(results / "tables"), stage="s6")
    validators.validate_anchors_manifest_json(report, manifest)
    checks = {finding.check: finding for finding in report.findings}
    assert checks["anchors_manifest.period_files"].status == "pass", checks["anchors_manifest.period_files"].detail
    assert checks["anchors_manifest.canonical_retention"].status == "pass"
    assert report.overall == "pass"
