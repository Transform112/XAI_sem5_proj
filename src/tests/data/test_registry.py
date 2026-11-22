"""Tests for the dataset registry (configs/datasets/*.yaml)."""

from __future__ import annotations

import pytest

from src.data.registry import DatasetSpecError, available_datasets, load_dataset_spec
from src.tests.data.helpers import NF_V3_COLUMNS, write_nf_csv, benign_then_attacks


def test_primary_dataset_spec_loads(repo_root):
    spec = load_dataset_spec("NF-UNSW-NB15-v3", project_root=repo_root)
    assert spec.name == "NF-UNSW-NB15-v3"
    assert spec.role == "primary"
    assert spec.schema == "nf_v3"
    assert spec.published["flows"] == 2365424
    assert spec.class_order is not None and len(spec.class_order) == 10
    assert spec.column("label_multiclass") == "Attack"
    assert spec.column("timestamp_start") == "FLOW_START_MILLISECONDS"


def test_common_spec_covers_exactly_the_real_columns(repo_root):
    """The YAML feature lists must partition the 55 real columns."""
    spec = load_dataset_spec("NF-UNSW-NB15-v3", project_root=repo_root)
    declared = set(spec.drop_features) | set(spec.categorical) | set(spec.log1p) | set(spec.numeric)
    assert declared == set(NF_V3_COLUMNS)
    assert len(spec.drop_features) == 7
    assert len(spec.categorical) == 5
    assert len(spec.log1p) == 34
    assert len(spec.numeric) == 9


def test_published_schema_is_43_plus_10_temporal_features(repo_root):
    """Verified T1 reference: 43 NetFlow-v2 + 10 temporal = 53 feature columns."""
    spec = load_dataset_spec("NF-UNSW-NB15-v3", project_root=repo_root)
    assert len(spec.nf_v2_features) == 43
    assert len(spec.temporal_raw) == 2
    assert len(spec.temporal_iat) == 8
    published = set(spec.nf_v2_features) | set(spec.temporal_raw) | set(spec.temporal_iat)
    assert len(published) == 53
    assert published | {"Label", "Attack"} == set(NF_V3_COLUMNS)


def test_temporal_raw_columns_are_dropped_as_features(repo_root):
    """Raw timestamps are sidecar-only; IAT columns stay as legitimate features."""
    spec = load_dataset_spec("NF-UNSW-NB15-v3", project_root=repo_root)
    for column in spec.temporal_raw:
        assert column in spec.drop_features
        assert column not in spec.feature_columns()
    for column in spec.temporal_iat:
        assert column in spec.feature_columns()
        assert column in spec.log1p


def test_every_dropped_column_has_a_documented_reason(repo_root):
    spec = load_dataset_spec("NF-UNSW-NB15-v3", project_root=repo_root)
    reasons = spec.drop_reasons
    assert set(reasons) == set(spec.drop_features)
    for column, reason in reasons.items():
        assert len(reason) > 20, f"reason for {column} is not descriptive"


def test_feature_columns_exclude_leakage_columns(repo_root):
    spec = load_dataset_spec("NF-UNSW-NB15-v3", project_root=repo_root)
    features = spec.feature_columns()
    assert len(features) == 48
    for leaked in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "FLOW_START_MILLISECONDS", "Label", "Attack"):
        assert leaked not in features


def test_available_datasets_lists_the_family_and_alternative(repo_root):
    names = available_datasets(project_root=repo_root)
    nf_v3 = [name for name in names if name.startswith("NF-")]
    assert nf_v3 == [
        "NF-BoT-IoT-v3",
        "NF-CSE-CIC-IDS2018-v3",
        "NF-ToN-IoT-v3",
        "NF-UNSW-NB15-v3",
    ]
    assert "Elliptic" in names  # alternative domain, audited separately (T10)


def test_unknown_dataset_raises_with_available_names(repo_root):
    with pytest.raises(DatasetSpecError) as excinfo:
        load_dataset_spec("does-not-exist", project_root=repo_root)
    assert "NF-UNSW-NB15-v3" in str(excinfo.value)


def test_csv_discovery_uses_spec_glob(repo_root, tmp_path):
    spec = load_dataset_spec("NF-UNSW-NB15-v3", project_root=repo_root)
    raw_dir = tmp_path / "raw" / spec.name
    write_nf_csv(raw_dir / f"{spec.name}.csv", benign_then_attacks(8))
    (raw_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
    found = spec.csv_files(raw_dir)
    assert [path.name for path in found] == [f"{spec.name}.csv"]


def test_missing_required_column_role_raises(tmp_path):
    configs = tmp_path / "configs" / "datasets"
    configs.mkdir(parents=True)
    (configs / "Broken.yaml").write_text(
        "name: Broken\nrole: test\ncolumns:\n  timestamp_start: T\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetSpecError) as excinfo:
        load_dataset_spec("Broken", configs_root=tmp_path / "configs", project_root=tmp_path)
    assert "label_multiclass" in str(excinfo.value)


def test_alternative_domain_spec_loads_without_common_schema(repo_root):
    spec = load_dataset_spec("Elliptic", project_root=repo_root)
    assert spec.name == "Elliptic"
    assert spec.schema == "elliptic"
    assert spec.class_order is None


def test_ton_iot_documented_class_aliases(repo_root):
    """D-020/D-031: ToN CSV spells 8 class names lower-case; aliases are explicit."""
    spec = load_dataset_spec("NF-ToN-IoT-v3", project_root=repo_root)
    aliases = spec.class_aliases
    assert aliases["XSS"] == "xss"
    assert aliases["DDoS"] == "ddos"
    assert len(aliases) == 8
    canonical = spec.canonical_class_map()
    assert canonical["ddos"] == "DDoS"
    # Every canonical class in class_order is covered by either direct name or alias.
    assert set(spec.class_order) == {"Benign", "Backdoor", "DoS", "DDoS", "Injection", "MITM", "Password", "Ransomware", "Scanning", "XSS"}
