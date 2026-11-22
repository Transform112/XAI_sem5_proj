"""Tests for dataset identity verification (T3).

Each test targets one documented identity check against a synthetic NF-v3 CSV,
so the behaviour is proven without the 2.3 GB dataset.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.data.identity import verify_dataset, write_identity_report
from src.data.registry import load_dataset_spec
from src.tests.data.helpers import benign_then_attacks, make_nf_row, write_nf_csv


@pytest.fixture()
def spec():
    return load_dataset_spec("NF-UNSW-NB15-v3")


@pytest.fixture()
def raw_dir(tmp_path: Path, spec) -> Path:
    """Extracted-raw layout for one tiny dataset: raw/<name>/<name>.csv."""
    directory = tmp_path / "raw" / spec.name
    write_nf_csv(directory / f"{spec.name}.csv", benign_then_attacks(8))
    return directory


def _report(spec, raw_dir, **kwargs):
    return verify_dataset(spec, raw_dir, **kwargs)


# ---------------------------------------------------------------------------
# Row and label counts
# ---------------------------------------------------------------------------
def test_counts_rows_and_multiclass_labels(spec, raw_dir):
    report = _report(spec, raw_dir, options={"chunksize": 3})
    assert report["rows"]["measured"] == 8
    assert report["labels"]["measured"] == {"Benign": 5, "Exploits": 2, "Worms": 1}
    assert report["labels"]["column"] == "Attack"
    assert report["duplicates"]["exact"] == 0


def test_chunk_size_does_not_change_counts(spec, raw_dir):
    small_chunks = _report(spec, raw_dir, options={"chunksize": 1})
    big_chunks = _report(spec, raw_dir, options={"chunksize": 1000})
    assert small_chunks["rows"]["measured"] == big_chunks["rows"]["measured"] == 8
    assert small_chunks["labels"]["measured"] == big_chunks["labels"]["measured"]


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------
def test_exact_duplicate_rows_are_counted(spec, raw_dir):
    rows = benign_then_attacks(8)
    rows.append(dict(rows[0]))  # byte-identical copy
    write_nf_csv(raw_dir / f"{spec.name}.csv", rows)
    report = _report(spec, raw_dir)
    assert report["rows"]["measured"] == 9
    assert report["duplicates"]["exact"] == 1
    assert report["duplicates"]["fraction"] == pytest.approx(1 / 9)


# ---------------------------------------------------------------------------
# Missing values and infinities
# ---------------------------------------------------------------------------
def test_missing_values_and_infinities_are_counted(spec, raw_dir):
    rows = benign_then_attacks(8)
    rows[0]["IN_BYTES"] = ""      # empty field -> NaN
    rows[1]["OUT_PKTS"] = "inf"   # pandas parses to +inf for float columns
    rows[2]["IN_PKTS"] = "-inf"
    write_nf_csv(raw_dir / f"{spec.name}.csv", rows)
    report = _report(spec, raw_dir)
    assert report["missing"]["total"] == 1
    assert report["missing"]["columns"]["IN_BYTES"] == 1
    assert report["inf"]["total"] == 2
    assert report["inf"]["columns"]["OUT_PKTS"] == 1
    assert report["inf"]["columns"]["IN_PKTS"] == 1


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------
def test_timestamps_parse_and_report_range(spec, raw_dir):
    report = _report(spec, raw_dir)
    timestamps = report["timestamps"]
    expected_min = datetime.fromtimestamp(1424242193, tz=timezone.utc).isoformat()
    expected_max = datetime.fromtimestamp(1424242200, tz=timezone.utc).isoformat()
    assert timestamps["column"] == "FLOW_START_MILLISECONDS"
    assert timestamps["parse_fraction"] == pytest.approx(1.0)
    assert timestamps["min_utc"] == expected_min
    assert timestamps["max_utc"] == expected_max
    assert timestamps["non_monotonic_fraction"] == pytest.approx(0.0)


def test_non_monotonic_timestamps_are_flagged(spec, raw_dir):
    rows = benign_then_attacks(8)
    rows[2]["FLOW_START_MILLISECONDS"] = "1000"  # goes backwards
    write_nf_csv(raw_dir / f"{spec.name}.csv", rows)
    report = _report(spec, raw_dir)
    assert report["timestamps"]["non_monotonic_fraction"] > 0
    assert report["timestamps"]["monotonic"] is False


# ---------------------------------------------------------------------------
# Hosts and schema
# ---------------------------------------------------------------------------
def test_host_cardinality(spec, raw_dir):
    report = _report(spec, raw_dir)
    hosts = report["hosts"]
    assert hosts["src_unique"] == 2
    assert hosts["dst_unique"] == 2
    assert hosts["union_unique"] == 4


def test_extra_column_is_flagged_as_schema_mismatch(spec, raw_dir):
    write_nf_csv(
        raw_dir / f"{spec.name}.csv",
        benign_then_attacks(8),
        extra_columns=["MYSTERY_COLUMN"],
    )
    report = _report(spec, raw_dir)
    assert report["schema"]["extra_columns"] == ["MYSTERY_COLUMN"]
    assert report["schema"]["matches_spec"] is False
    assert report["status"] == "fail"


# ---------------------------------------------------------------------------
# Published-value comparison
# ---------------------------------------------------------------------------
def test_published_row_count_mismatch_is_flagged(spec, raw_dir):
    report = _report(spec, raw_dir)
    rows = report["rows"]
    assert rows["published"] == 2365424
    assert rows["measured"] == 8
    assert rows["delta"] == 8 - 2365424
    assert rows["match"] is False
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "row_count_vs_published" in failed
    assert report["status"] == "fail"


def test_published_label_counts_are_compared_when_pinned(spec, raw_dir):
    report = _report(spec, raw_dir)
    mismatches = {item["class"] for item in report["labels"]["mismatches"]}
    # Analysis/Backdoor/Worms are pinned in the dataset yaml; only Worms exists
    # in the fixture, with a different count than the published 158.
    assert {"Analysis", "Backdoor", "Worms"} <= mismatches


def test_consistent_dataset_passes(spec, raw_dir):
    published = {"flows": 8, "classes": {"Benign": 5, "Exploits": 2, "Worms": 1}}
    matching = dataclasses.replace(spec, published=published)
    report = _report(matching, raw_dir)
    assert report["status"] == "pass", [c for c in report["checks"] if not c["passed"]]
    assert report["rows"]["match"] is True
    assert report["labels"]["mismatches"] == []


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def test_provenance_size_and_hash_mismatch_is_flagged(spec, raw_dir):
    archive = raw_dir / f"{spec.name}.zip"
    archive.write_bytes(b"not really a zip")
    provenance = {"zip_bytes": 999999, "sha256": "0" * 64, "published_flow_count": 8}
    report = _report(
        spec,
        raw_dir,
        options={"hash_archives": True},
        provenance=provenance,
    )
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "provenance_zip_bytes" in failed
    assert "provenance_zip_sha256" in failed
    assert report["status"] == "fail"


def test_class_table_sum_discrepancy_is_reported_not_forced(spec, raw_dir):
    published = {
        "flows": 8,
        "classes": {"Benign": 5, "Exploits": 2, "Worms": 1},
        "class_table_sum": 10,  # deliberately inconsistent with the row total
    }
    variant = dataclasses.replace(spec, published=published)
    report = _report(variant, raw_dir)
    check = next(
        check for check in report["checks"] if check["name"] == "label_total_vs_published_class_table"
    )
    assert check["passed"] is False
    assert check["severity"] == "warn"  # published table inconsistency is a finding, not a failure
    assert check["measured"] == 8
    assert check["expected"] == 10
    assert report["labels"]["class_table_delta"] == -2
    assert report["status"] == "warn"


def test_published_class_count_check(spec, raw_dir):
    published = {
        "flows": 8,
        "classes": {"Benign": 5, "Exploits": 2, "Worms": 1},
        "classes_count": 3,
    }
    variant = dataclasses.replace(spec, published=published)
    report = _report(variant, raw_dir)
    check = next(
        check for check in report["checks"] if check["name"] == "label_classes_count_vs_published"
    )
    assert check["passed"] is True
    assert check["measured"] == 3


def test_website_values_are_reported_but_never_enforced(spec, raw_dir):
    """D-013: UQ website values are wrong; they must be reported, not failed on."""
    published = {
        "flows": 8,
        "classes": {"Benign": 5, "Exploits": 2, "Worms": 1},
        "website_classes": {"Analysis": 2381, "Worms": 999},
    }
    variant = dataclasses.replace(spec, published=published)
    report = _report(variant, raw_dir)
    mismatch_classes = {item["class"] for item in report["labels"]["website_mismatches"]}
    assert mismatch_classes == {"Analysis", "Worms"}
    website_checks = [check for check in report["checks"] if "website" in check["name"]]
    assert website_checks == []  # report-only: no pass/fail check is created
    assert report["status"] == "pass"


def test_attack_total_is_checked_against_paper_value(spec, raw_dir):
    published = {
        "flows": 8,
        "classes": {"Benign": 5, "Exploits": 2, "Worms": 1},
        "attack_total": 3,  # 2 Exploits + 1 Worms
    }
    variant = dataclasses.replace(spec, published=published)
    report = _report(variant, raw_dir)
    check = next(check for check in report["checks"] if check["name"] == "attack_total_vs_published")
    assert check["passed"] is True
    assert report["labels"]["attack_total"] == 3


def test_alternate_flow_counts_are_reported_with_delta(spec, raw_dir):
    published = {
        "flows": 8,
        "classes": {"Benign": 5, "Exploits": 2, "Worms": 1},
        "alternate_flow_counts": {"uq_website": 10},
    }
    variant = dataclasses.replace(spec, published=published)
    report = _report(variant, raw_dir)
    alternate = report["rows"]["alternate_flow_counts"]["uq_website"]
    assert alternate["published"] == 10
    assert alternate["delta_measured_minus_published"] == -2
    assert alternate["match"] is False
    assert report["status"] == "pass"  # discrepancy is reported, not a failure


# ---------------------------------------------------------------------------
# Report artifacts
# ---------------------------------------------------------------------------
def test_write_identity_report_creates_json_and_markdown(tmp_path, spec, raw_dir):
    report = _report(spec, raw_dir)
    paths = write_identity_report(report, tmp_path / "out")
    json_path = Path(paths["json"])
    md_path = Path(paths["md"])
    assert json_path.name == f"identity_{spec.name}.json"
    assert md_path.name == f"identity_{spec.name}.md"
    assert json.loads(json_path.read_text(encoding="utf-8"))["rows"]["measured"] == 8
    markdown = md_path.read_text(encoding="utf-8")
    assert spec.name in markdown
    assert "FAIL" in markdown  # published count deliberately mismatched here


# ---------------------------------------------------------------------------
# Encoding / corrupt input handling
# ---------------------------------------------------------------------------
def test_unparseable_numeric_text_does_not_crash(spec, raw_dir):
    rows = benign_then_attacks(8)
    rows[0]["IN_BYTES"] = "not_a_number"
    write_nf_csv(raw_dir / f"{spec.name}.csv", rows)
    report = _report(spec, raw_dir)
    # Coercion failure becomes a missing value and must be reported, not fatal.
    assert report["missing"]["columns"].get("IN_BYTES", 0) == 1
