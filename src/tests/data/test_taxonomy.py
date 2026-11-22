"""Tests for the documented label taxonomy (D-031 CIC 15->7 aggregation)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.taxonomy import LabelTaxonomy, TaxonomyError

CONFIG = {
    "name": "cic_ids2018_15_to_7",
    "aggregate_to": {
        "Benign": "Benign",
        "Bot": "Bot",
        "DDOS_attack-HOIC": "DDoS",
        "DDoS_attacks-LOIC-HTTP": "DDoS",
        "DDOS_attack-LOIC-UDP": "DDoS",
        "DoS_attacks-GoldenEye": "DoS",
        "DoS_attacks-Hulk": "DoS",
        "DoS_attacks-Slowhttptest": "DoS",
        "DoS_attacks-Slowloris": "DoS",
        "FTP-BruteForce": "BruteForce",
        "SSH-Bruteforce": "BruteForce",
        "Infilteration": "Infiltration",
        "Brute_Force_-Web": "Web Attacks",
        "Brute_Force_-XSS": "Web Attacks",
        "SQL_Injection": "Web Attacks",
    },
    "published_aggregated_counts": {
        "Benign": 17514626,
        "Bot": 207703,
        "BruteForce": 575194,
        "DDoS": 1324350,
        "DoS": 302966,
        "Infiltration": 188152,
        "Web Attacks": 2538,
    },
    "strict_aggregate_check": True,
}


def test_15_to_7_mapping_aggregates_to_published_classes():
    taxonomy = LabelTaxonomy.from_config(CONFIG)
    assert taxonomy is not None
    assert len(taxonomy.aggregate_to) == 15
    assert taxonomy.target_classes() == [
        "Benign",
        "Bot",
        "BruteForce",
        "DDoS",
        "DoS",
        "Infiltration",
        "Web Attacks",
    ]
    assert sum(taxonomy.published_aggregated_counts.values()) == 20115529


def test_aggregate_counts_matches_published_table():
    taxonomy = LabelTaxonomy.from_config(CONFIG)
    # Build fine counts that reproduce the published aggregated counts.
    fine = {
        "Benign": 17514626,
        "Bot": 207703,
        "DDOS_attack-HOIC": 1032311,
        "DDoS_attacks-LOIC-HTTP": 288589,
        "DDOS_attack-LOIC-UDP": 3450,
        "DoS_attacks-GoldenEye": 100000,
        "DoS_attacks-Hulk": 100000,
        "DoS_attacks-Slowhttptest": 51483,
        "DoS_attacks-Slowloris": 51483,
        "FTP-BruteForce": 386720,
        "SSH-Bruteforce": 188474,
        "Infilteration": 188152,
        "Brute_Force_-Web": 1618,
        "Brute_Force_-XSS": 480,
        "SQL_Injection": 440,
    }
    aggregated = taxonomy.aggregate_counts(fine)
    taxonomy.validate_counts(aggregated, measured_total=sum(fine.values()))
    assert aggregated["DDoS"] == 1324350
    assert aggregated["DoS"] == 302966
    assert aggregated["BruteForce"] == 575194
    assert aggregated["Web Attacks"] == 2538
    assert aggregated["Infiltration"] == 188152
    assert aggregated["Bot"] == 207703


def test_aggregated_count_mismatch_fails_loudly():
    taxonomy = LabelTaxonomy.from_config(CONFIG)
    fine = {
        "Benign": 17514626,
        "Bot": 207703,
        "DDOS_attack-HOIC": 1032311,
        "DDoS_attacks-LOIC-HTTP": 288589,
        "DDOS_attack-LOIC-UDP": 3450,
        "DoS_attacks-GoldenEye": 100000,
        "DoS_attacks-Hulk": 100000,
        "DoS_attacks-Slowhttptest": 51483,
        "DoS_attacks-Slowloris": 51483,
        "FTP-BruteForce": 386720,
        "SSH-Bruteforce": 188474,
        "Infilteration": 188152,
        "Brute_Force_-Web": 1618,
        "Brute_Force_-XSS": 480,
        "SQL_Injection": 999,  # wrong
    }
    with pytest.raises(TaxonomyError) as excinfo:
        taxonomy.aggregate_counts(fine)
    assert "Web Attacks" in str(excinfo.value)


def test_unknown_label_raises_instead_of_silent_relabel():
    taxonomy = LabelTaxonomy.from_config(CONFIG)
    series = pd.Series(["Benign", "NEW_ATTACK_KIND"])
    with pytest.raises(TaxonomyError) as excinfo:
        taxonomy.apply_series(series)
    assert "NEW_ATTACK_KIND" in str(excinfo.value)


def test_apply_series_maps_labels():
    taxonomy = LabelTaxonomy.from_config(CONFIG)
    series = pd.Series(["Benign", "BoT".replace("BoT", "Bot"), "FTP-BruteForce", None])
    mapped = taxonomy.apply_series(series)
    assert mapped.tolist()[:3] == ["Benign", "Bot", "BruteForce"]
    assert pd.isna(mapped.iloc[3])


def test_empty_config_returns_none():
    assert LabelTaxonomy.from_config(None) is None
    assert LabelTaxonomy.from_config({}) is None
    with pytest.raises(TaxonomyError):
        LabelTaxonomy.from_config({"aggregate_to": {}})
