"""Tests for the S-D1 preprocessing/split pipeline (T4).

Covers: raw loading, exact dedupe, sentinel/NaN/inf policy, chronological
periods, leakage-safe grouped splits, train-only preprocessing and the feature
manifest. All tests run on synthetic NF-v3 CSVs (no dataset download needed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean_frame, load_raw_frame
from src.data.features import build_feature_manifest
from src.data.periods import assign_periods
from src.data.preprocess import Preprocessor
from src.data.registry import load_dataset_spec
from src.data.splits import build_splits, freeze_split_masks
from src.tests.data.helpers import make_nf_row, write_nf_csv


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------
@pytest.fixture()
def spec():
    return load_dataset_spec("NF-UNSW-NB15-v3")


def _synthetic_rows() -> list[dict[str, str]]:
    """60 deterministic rows with the awkward cases the cleaner must handle.

    * 6 clean rows per "second bucket" spread over 10 seconds (timestamp
      increases, so period cutting is well defined),
    * an exact duplicate pair (same all 55 columns),
    * a near-duplicate pair (same 5-tuple + same second, different bytes),
    * a negative sentinel in IN_BYTES, an infinity in OUT_BYTES, a missing
      value in IN_PKTS,
    * two classes (Benign, Exploits) plus one Worms row.
    """
    rows: list[dict[str, str]] = []
    base_ts = 1424242193000
    for index in range(60):
        seconds = index // 6
        label = "Worms" if index == 59 else ("Exploits" if index % 3 == 0 and index % 2 == 0 else "Benign")
        binary = "0" if label == "Benign" else "1"
        rows.append(
            make_nf_row(
                FLOW_START_MILLISECONDS=str(base_ts + seconds * 1000 + index % 6),
                FLOW_END_MILLISECONDS=str(base_ts + seconds * 1000 + 10 + index % 6),
                IPV4_SRC_ADDR=f"10.0.0.{1 + (index % 3)}",
                IPV4_DST_ADDR=f"10.0.0.{4 + (index % 2)}",
                L4_SRC_PORT=str(40000 + index % 5),
                L4_DST_PORT=str(80 + index % 2),
                PROTOCOL="6" if index % 2 == 0 else "17",
                IN_BYTES=str(100 + index),
                OUT_BYTES=str(200 + index),
                IN_PKTS=str(1 + index % 7),
                Label=binary,
                Attack=label,
            )
        )
    # exact duplicate of row 0
    rows.append(dict(rows[0]))
    # near-duplicate of row 1: same 5-tuple and same second, different bytes
    near = dict(rows[1])
    near["IN_BYTES"] = "999999"
    rows.append(near)
    # sentinel / infinity / missing values
    rows[2]["IN_BYTES"] = "-1"
    rows[3]["OUT_BYTES"] = "inf"
    rows[4]["IN_PKTS"] = ""
    return rows


@pytest.fixture()
def raw_frame(tmp_path: Path, spec):
    csv_path = write_nf_csv(tmp_path / spec.name / f"{spec.name}.csv", _synthetic_rows())
    return load_raw_frame(spec, csv_path)


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------
def test_load_raw_frame_keeps_all_physical_columns(raw_frame, spec):
    assert set(raw_frame.columns) >= set(spec.all_columns())
    assert len(raw_frame) == 62  # 60 generated rows + 1 exact duplicate + 1 near-duplicate


def test_clean_drops_exact_duplicates(raw_frame, spec):
    result = clean_frame(raw_frame, spec)
    assert result.stats["rows_in"] == 62
    assert result.stats["exact_duplicates_removed"] == 1
    assert len(result.frame) == 61
    assert result.stats["rows_after_dedupe"] == 61


def test_clean_applies_sentinel_policy(raw_frame, spec):
    result = clean_frame(raw_frame, spec)
    frame = result.frame
    assert result.stats["negative_to_nan"]["IN_BYTES"] == 1
    assert not (frame["IN_BYTES"] < 0).any()  # sentinel replaced, not kept
    assert np.isnan(frame["IN_BYTES"]).sum() == 1
    assert result.stats["inf_to_nan"]["OUT_BYTES"] == 1
    assert np.isinf(frame["OUT_BYTES"]).sum() == 0
    assert np.isnan(frame["OUT_BYTES"]).sum() == 1
    assert np.isnan(frame["IN_PKTS"]).sum() == 1  # pre-existing missing value


def test_clean_sorts_chronologically_and_assigns_row_ids(raw_frame, spec):
    result = clean_frame(raw_frame, spec)
    frame = result.frame
    assert frame["_ts"].is_monotonic_increasing
    assert frame["_row_id"].tolist() == list(range(len(frame)))
    assert result.stats["time_sorted"] is True


def test_clean_builds_global_maps(raw_frame, spec):
    result = clean_frame(raw_frame, spec)
    assert result.class_map["Benign"] == 0
    assert set(result.class_map) == {"Benign", "Exploits", "Worms"}
    hosts = set(result.host_map)
    assert hosts == {"10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4", "10.0.0.5"}
    # host ids are dense and deterministic
    assert sorted(result.host_map.values()) == list(range(len(hosts)))
    assert result.class_map is not None and result.host_map == dict(sorted(result.host_map.items()))


def test_clean_groups_near_duplicates(raw_frame, spec):
    result = clean_frame(raw_frame, spec)
    frame = result.frame
    # rows 1 and the near-duplicate share a group (same 5-tuple + same second)
    near_rows = frame[frame["IN_BYTES"] == 999999]
    partner = frame[(frame["_group_id"] == int(near_rows["_group_id"].iloc[0])) & (frame["IN_BYTES"] != 999999)]
    assert len(partner) >= 1
    assert result.stats["near_duplicate_groups"] < result.stats["rows_after_dedupe"]


def test_near_duplicate_key_uses_global_host_ids(tmp_path, spec):
    """D-034: the canonical key must distinguish host pairs, not collapse all IPs."""
    base = 1424242193000
    rows = []
    # Same ports/protocol/second but different host pairs -> different groups.
    for index, (src, dst) in enumerate(
        [("10.9.0.1", "10.9.0.2"), ("10.9.0.3", "10.9.0.4"), ("10.9.0.1", "10.9.0.4")]
    ):
        rows.append(
            make_nf_row(
                FLOW_START_MILLISECONDS=str(base + index),
                IPV4_SRC_ADDR=src,
                IPV4_DST_ADDR=dst,
                L4_SRC_PORT="50000",
                L4_DST_PORT="80",
                PROTOCOL="6",
                Attack="Benign",
                Label="0",
            )
        )
    # Two rows on the SAME host pair/ports/second -> one shared group.
    for index in range(2):
        rows.append(
            make_nf_row(
                FLOW_START_MILLISECONDS=str(base + index),
                IPV4_SRC_ADDR="10.9.0.7",
                IPV4_DST_ADDR="10.9.0.8",
                L4_SRC_PORT="50001",
                L4_DST_PORT="443",
                PROTOCOL="6",
                IN_BYTES=str(index + 1),
                Attack="Benign",
                Label="0",
            )
        )
    csv_path = write_nf_csv(tmp_path / spec.name / f"{spec.name}.csv", rows)
    result = clean_frame(load_raw_frame(spec, csv_path), spec)
    frame = result.frame
    same_pair_groups = frame[frame["IPV4_SRC_ADDR"] == "10.9.0.7"]["_group_id"].nunique()
    assert same_pair_groups == 1  # repeated flows on one host pair share a group
    # Source hosts: 10.9.0.1 (.2, .4 dsts), 10.9.0.3, 10.9.0.7 -> 3 unique sources.
    assert frame["_host_src_id"].nunique() == 3
    assert frame["_group_id"].nunique() >= 4


def test_label_conflict_drop_groups_rule(tmp_path, spec):
    """D-032: conflicting near-duplicate groups are reported and optionally dropped."""
    base = 1424242193000
    rows = []
    for index, label in enumerate(["Benign", "Exploits", "Benign"]):
        rows.append(
            make_nf_row(
                FLOW_START_MILLISECONDS=str(base + index),
                FLOW_END_MILLISECONDS=str(base + index + 10),
                IPV4_SRC_ADDR="10.1.0.1",
                IPV4_DST_ADDR="10.1.0.2",
                L4_SRC_PORT="50000",
                L4_DST_PORT="80",
                PROTOCOL="6",
                IN_BYTES=str(index + 1),
                Attack=label,
                Label="0" if label == "Benign" else "1",
            )
        )
    rows.append(
        make_nf_row(
            FLOW_START_MILLISECONDS=str(base + 60_000),
            IPV4_SRC_ADDR="10.1.0.3",
            IPV4_DST_ADDR="10.1.0.4",
            IN_BYTES="777",
            Attack="Benign",
            Label="0",
        )
    )
    csv_path = write_nf_csv(tmp_path / spec.name / f"{spec.name}.csv", rows)
    raw = load_raw_frame(spec, csv_path)

    report_only = clean_frame(raw, spec)
    assert report_only.stats["label_conflict"]["n_conflicted_groups"] == 1
    assert report_only.stats["label_conflict"]["rows_dropped"] == 0
    assert len(report_only.frame) == 4

    dropped = clean_frame(raw, spec, label_conflict_policy="drop_groups")
    assert dropped.stats["label_conflict"]["rows_dropped"] == 3
    assert dropped.stats["label_conflict"]["n_conflicted_groups"] == 1
    assert len(dropped.frame) == 1
    assert dropped.stats["label_conflict"]["after"]["rows"] == 1
    examples = dropped.stats["label_conflict"]["examples"]
    assert examples and set(examples[0]["labels"]) == {"Benign", "Exploits"}


def test_unknown_label_conflict_policy_raises(raw_frame, spec):
    with pytest.raises(ValueError):
        clean_frame(raw_frame, spec, label_conflict_policy="nonsense")


def test_documented_class_aliases_are_applied_in_cleaning(tmp_path):
    """ToN lower-case CSV class values map to canonical paper names on load."""
    from src.data.registry import load_dataset_spec as _load

    ton = _load("NF-ToN-IoT-v3")
    rows = [
        make_nf_row(Attack="ddos", Label="1"),
        make_nf_row(Attack="xss", Label="1"),
        make_nf_row(Attack="Benign", Label="0"),
    ]
    csv_path = write_nf_csv(tmp_path / ton.name / f"{ton.name}.csv", rows)
    result = clean_frame(load_raw_frame(ton, csv_path), ton)
    assert result.stats["class_aliases"]["applied"]["DDoS"] == "ddos"
    assert set(result.class_map) == {"Benign", "DDoS", "XSS"}
    assert result.class_map["Benign"] == 0


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------
def test_periods_are_equal_count_and_chronological(raw_frame, spec):
    result = clean_frame(raw_frame, spec)
    periods = assign_periods(result.frame, n_periods=2, strategy="equal_counts")
    frame = periods.frame
    counts = frame["_period"].value_counts().sort_index()
    assert list(counts.index) == [0, 1]
    assert counts.max() - counts.min() <= 1
    # chronological: every timestamp in period 1 >= every timestamp in period 0
    assert frame.loc[frame["_period"] == 1, "_ts"].min() >= frame.loc[frame["_period"] == 0, "_ts"].max()
    assert periods.stats["time_disjoint"] is True


def test_period_label_histogram_is_recorded(raw_frame, spec):
    result = clean_frame(raw_frame, spec)
    periods = assign_periods(result.frame, n_periods=2)
    assert set(periods.stats["label_counts"]) == {0, 1}
    total = sum(sum(counts.values()) for counts in periods.stats["label_counts"].values())
    assert total == len(result.frame)


def _two_session_rows(n_session1: int = 30, n_session2: int = 120) -> list[dict[str, str]]:
    """Synthetic capture stream with one large timestamp gap (two sessions)."""
    base = 1424242193000
    gap = 26 * 24 * 3600 * 1000
    rows: list[dict[str, str]] = []
    labels = ["Benign", "Exploits", "Fuzzers"]
    for index in range(n_session1 + n_session2):
        in_second_session = index >= n_session1
        local = index - n_session1 if in_second_session else index
        ts = base + local * 1000 + (gap if in_second_session else 0)
        label = labels[index % len(labels)]
        rows.append(
            make_nf_row(
                FLOW_START_MILLISECONDS=str(ts),
                FLOW_END_MILLISECONDS=str(ts + 10),
                IPV4_SRC_ADDR=f"10.0.0.{1 + index % 4}",
                IPV4_DST_ADDR=f"10.0.1.{1 + index % 3}",
                Label="0" if label == "Benign" else "1",
                Attack=label,
            )
        )
    return rows


def test_session_aware_1_4_detects_capture_session_gap(tmp_path, spec):
    """D-026 C1 stream: P1 = first session, P2..P5 = equal counts in session 2."""
    n_session1, n_session2 = 30, 120
    csv_path = write_nf_csv(
        tmp_path / spec.name / f"{spec.name}.csv", _two_session_rows(n_session1, n_session2)
    )
    raw = load_raw_frame(spec, csv_path)
    clean = clean_frame(raw, spec)
    periods = assign_periods(
        clean.frame,
        n_periods=5,
        strategy="session_aware_1_4",
        min_session_gap_seconds=3600,
    )
    counts = periods.frame["_period"].value_counts().sort_index()
    assert list(counts.index) == [0, 1, 2, 3, 4]
    assert int(counts[0]) == n_session1
    assert periods.stats["session"]["session_row_counts"] == [n_session1, n_session2]
    assert periods.stats["session"]["session_gap_seconds"] >= 24 * 3600
    remainder = counts.iloc[1:]
    assert int(remainder.max() - remainder.min()) <= 1
    assert periods.stats["time_disjoint"] is True
    assert periods.stats["strategy"] == "session_aware_1_4"


def test_session_aware_1_4_requires_a_session_gap(raw_frame, spec):
    clean = clean_frame(raw_frame, spec)  # single-session stream: no large gap
    with pytest.raises(ValueError) as excinfo:
        assign_periods(clean.frame, n_periods=5, strategy="session_aware_1_4", min_session_gap_seconds=3600)
    assert "session gap" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
@pytest.fixture()
def split_input(raw_frame, spec):
    clean = clean_frame(raw_frame, spec)
    periods = assign_periods(clean.frame, n_periods=2)
    return periods.frame


def test_grouped_split_never_crosses_near_duplicate_groups(split_input):
    splits = build_splits(split_input, fractions={"train": 0.7, "val": 0.1, "test": 0.2}, seed=42)
    for period in (0, 1):
        masks = splits.masks[period]
        seen: dict[int, str] = {}
        for split_name in ("train", "val", "test"):
            group_ids = split_input.iloc[masks[split_name]]["_group_id"].to_numpy()
            for group in np.unique(group_ids):
                assert group not in seen or seen[group] == split_name, f"group {group} crosses splits"
                seen[int(group)] = split_name
    assert splits.stats["groups_crossing_splits"] == 0


def test_split_is_deterministic_given_seed(split_input):
    first = build_splits(split_input, fractions={"train": 0.7, "val": 0.1, "test": 0.2}, seed=42)
    second = build_splits(split_input, fractions={"train": 0.7, "val": 0.1, "test": 0.2}, seed=42)
    third = build_splits(split_input, fractions={"train": 0.7, "val": 0.1, "test": 0.2}, seed=43)
    for period in (0, 1):
        for split_name in ("train", "val", "test"):
            np.testing.assert_array_equal(first.masks[period][split_name], second.masks[period][split_name])
    assert not np.array_equal(first.masks[0]["train"], third.masks[0]["train"])


def test_split_fractions_are_approximately_honoured(split_input):
    splits = build_splits(split_input, fractions={"train": 0.7, "val": 0.1, "test": 0.2}, seed=42)
    for period in (0, 1):
        masks = splits.masks[period]
        total = sum(len(masks[name]) for name in ("train", "val", "test"))
        assert total == int((split_input["_period"] == period).sum())
        assert len(masks["train"]) / total == pytest.approx(0.7, abs=0.15)
        assert len(masks["test"]) / total == pytest.approx(0.2, abs=0.15)


def test_freeze_split_masks_hashes_are_stable(split_input):
    splits = build_splits(split_input, fractions={"train": 0.7, "val": 0.1, "test": 0.2}, seed=42)
    manifest = freeze_split_masks(splits)
    assert manifest["frozen_hash"] and len(manifest["frozen_hash"]) == 64
    again = freeze_split_masks(build_splits(split_input, fractions={"train": 0.7, "val": 0.1, "test": 0.2}, seed=42))
    assert manifest["frozen_hash"] == again["frozen_hash"]
    other = freeze_split_masks(build_splits(split_input, fractions={"train": 0.7, "val": 0.1, "test": 0.2}, seed=7))
    assert manifest["frozen_hash"] != other["frozen_hash"]


def test_host_pair_disjoint_split_keeps_host_pairs_out_of_test(split_input):
    splits = build_splits(
        split_input,
        fractions={"train": 0.7, "val": 0.1, "test": 0.2},
        seed=42,
        mode="host_pair_disjoint",
        min_groups_per_split=1,
        min_share=0.0,
    )
    frame = split_input
    pair = list(zip(frame["_host_src_id"], frame["_host_dst_id"], frame["PROTOCOL"]))
    train_pairs = {pair[i] for i in splits.masks[0]["train"]}
    test_pairs = {pair[i] for i in splits.masks[0]["test"]}
    assert not (train_pairs & test_pairs)


# ---------------------------------------------------------------------------
# Preprocessing (train-only fitting)
# ---------------------------------------------------------------------------
@pytest.fixture()
def prepared(split_input):
    splits = build_splits(split_input, fractions={"train": 0.7, "val": 0.1, "test": 0.2}, seed=42)
    return split_input, splits


def test_preprocessor_fits_only_on_period1_train(prepared, spec):
    frame, splits = prepared
    pre = Preprocessor(spec, config={"missing_indicator_threshold": 0.0001, "clip_std": 10.0})
    train = frame.iloc[splits.masks[0]["train"]]
    pre.fit(train)
    stats = pre.state_dict()["columns"]
    # The scaler means are exactly the train means of the transformed train rows.
    column = "IN_BYTES"
    expected_median = np.nanmedian(np.log1p(np.clip(train[column].to_numpy(dtype=float), 0, None)))
    assert stats[column]["impute_value"] == pytest.approx(expected_median)

    # Transforming a different period must not change fitted statistics.
    before = pre.state_dict()
    _ = pre.transform(frame.iloc[splits.masks[0]["test"]])
    after = pre.state_dict()
    assert before == after


def test_preprocessor_applies_log1p_and_standardisation(prepared, spec):
    frame, splits = prepared
    pre = Preprocessor(spec, config={})
    train = frame.iloc[splits.masks[0]["train"]]
    pre.fit(train)
    matrix = pre.transform(train)
    names = pre.feature_names
    column = "OUT_BYTES"  # no extreme values / clipping in the synthetic fixture
    column_index = names.index(column)
    expected_median = np.nanmedian(
        np.log1p(np.clip(train[column].to_numpy(dtype=float), 0, None))
    )
    assert pre.state_dict()["columns"][column]["impute_value"] == pytest.approx(expected_median)
    assert abs(matrix[:, column_index].mean()) < 1e-6  # standardised on the fitted rows
    assert np.abs(matrix[:, column_index]).max() < 10.0  # clip_std default not hit


def test_preprocessor_encodes_unseen_categories_to_unknown(prepared, spec):
    frame, splits = prepared
    pre = Preprocessor(spec, config={})
    pre.fit(frame.iloc[splits.masks[0]["train"]])
    test = frame.iloc[splits.masks[0]["test"]].copy()
    test.loc[test.index[0], "PROTOCOL"] = 12345  # never seen in train
    matrix = pre.transform(test)
    index = pre.feature_names.index("PROTOCOL")
    assert matrix[0, index] == 0.0  # reserved unknown index is standardised to a known value
    vocab = pre.state_dict()["categorical"]["PROTOCOL"]["vocab"]
    assert vocab["<unknown>"] == 0


def test_preprocessor_output_is_finite(prepared, spec):
    frame, splits = prepared
    pre = Preprocessor(spec, config={})
    pre.fit(frame.iloc[splits.masks[0]["train"]])
    matrix = pre.transform(frame)
    assert np.isfinite(matrix).all()


def test_preprocessor_round_trips_state(prepared, spec, tmp_path):
    frame, splits = prepared
    pre = Preprocessor(spec, config={})
    train = frame.iloc[splits.masks[0]["train"]]
    pre.fit(train)
    path = tmp_path / "preprocessor_state.json"
    pre.save(path)
    loaded = Preprocessor.load(spec, path)
    np.testing.assert_allclose(pre.transform(train), loaded.transform(train))


def test_missing_indicator_added_when_train_missing_rate_is_high(prepared, spec):
    frame, splits = prepared
    train = frame.iloc[splits.masks[0]["train"]].copy()
    train.loc[train.index[0], "IN_PKTS"] = np.nan  # guarantee a missing value in the fit set
    pre = Preprocessor(spec, config={"missing_indicator_threshold": 0.0001})
    pre.fit(train)
    assert "IN_PKTS_missing" in pre.feature_names


# ---------------------------------------------------------------------------
# Feature manifest
# ---------------------------------------------------------------------------
def test_feature_manifest_classifies_every_column(spec):
    manifest = build_feature_manifest(spec)
    assert set(manifest["name"]) == set(spec.all_columns())
    assert len(manifest) == 55
    roles = manifest.set_index("name")["role"]
    assert roles["Attack"] == "target"
    assert roles["Label"] == "dropped"
    assert roles["IPV4_SRC_ADDR"] == "dropped"
    assert roles["FLOW_START_MILLISECONDS"] == "dropped"
    assert roles["IN_BYTES"] == "feature"
    # every dropped column carries a non-empty reason
    dropped = manifest[manifest["role"] == "dropped"]
    assert dropped["reason"].str.len().min() > 20
    # the 10 raw/derived temporal columns are explicitly sidecar/feature
    sidecar = manifest.set_index("name").loc[list(spec.temporal_raw), "graph_role"]
    assert (sidecar == "sidecar").all()
