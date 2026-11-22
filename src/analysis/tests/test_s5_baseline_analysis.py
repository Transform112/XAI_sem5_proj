"""Tests for T46 baseline CL analysis helpers (synthetic, no Kaggle inputs)."""

from __future__ import annotations

import csv
import json

import numpy as np

from src.analysis import s5_baseline_analysis as mod


def _matrix_payload(base: float, seed: int, model: str, strategy: str) -> dict:
    # monotonically ordered per-class metrics; values vary by seed/strategy/model
    strategy_offset = {"ft": 0.0, "replay": -0.03, "ewc": -0.01}[strategy]
    offset = strategy_offset * (1.0 + 0.1 * (seed - 45)) + (seed - 45) * 0.001
    final = base + offset
    return {
        "checkpoints_evaluated": [1, 2, 3, 4, 5],
        "evaluation": {"mode": "edge_batch"},
        "cl_metrics_all_classes": {
            "final_average_macro_f1": final,
            "forgetting": 0.04 + offset / 3.0,
            "bwt": -(0.04 + offset / 3.0),
            "forgetting_per_period": {"1": 0.02, "2": 0.03, "3": 0.01, "4": 0.0, "5": None},
        },
        "cl_metrics_supported_set": {
            "final_average_macro_f1": final + 0.1,
            "forgetting": 0.02 + offset / 4.0,
            "bwt": -(0.02 + offset / 4.0),
            "forgetting_per_period": {"1": 0.01, "2": 0.02, "3": 0.0, "4": 0.0, "5": None},
        },
        "matrix": {
            "all_classes": np.full((5, 5), final).tolist(),
            "supported_set": np.full((5, 5), final + 0.1).tolist(),
        },
    }


def _synthetic_groups() -> dict:
    groups = {}
    for strategy in mod.PRIMARY_STRATEGIES:
        for model in mod.MODELS:
            for seed in mod.SEEDS:
                base = 0.44 if model == "egraphsage" else 0.41
                groups[(strategy, model, seed)] = {"matrix": _matrix_payload(base, seed, model, strategy)}
    return groups


def test_mean_sd_handles_nan_and_single_value():
    mean, sd = mod.mean_sd(np.array([1.0, 2.0, 3.0]))
    assert mean == 2.0 and sd == 1.0
    mean, sd = mod.mean_sd(np.array([np.nan, 2.0]))
    assert mean == 2.0 and sd == 0.0
    mean, sd = mod.mean_sd(np.array([np.nan]))
    assert np.isnan(mean) and np.isnan(sd)


def test_comparison_rows_holm_family_and_effect_sizes():
    groups = _synthetic_groups()
    rows = mod.comparison_rows(groups, n_resamples=50, seed=7)
    assert len(rows) == len(mod.MODELS) * len(mod.METRICS) * len(mod.COMPARISONS)
    families: dict[str, list[dict]] = {}
    for row in rows:
        families.setdefault(row["family"], []).append(row)
        assert row["n"] == 5
        assert np.isfinite(row["diff"])
        assert row["ci_lo"] <= row["diff"] <= row["ci_hi"]
        assert np.isfinite(row["cohens_d"])
        assert -1.0 <= row["rank_biserial"] <= 1.0
        assert row["p_holm"] >= row["p_raw"] - 1e-12
    assert all(len(rows) == 3 for rows in families.values())
    family = families["egraphsage:final_all"]
    holm = [row["p_holm"] for row in family]
    assert min(holm) >= min(row["p_raw"] for row in family) - 1e-12


def test_comparison_uses_paired_seed_order():
    groups = _synthetic_groups()
    rows = mod.comparison_rows(groups, n_resamples=20, seed=3)
    egs = [
        row for row in rows
        if row["model"] == "egraphsage" and row["metric"] == "final_all" and row["comparison"] == "replay_minus_ft"
    ][0]
    expected = np.mean(
        [groups[("replay", "egraphsage", seed)]["matrix"]["cl_metrics_all_classes"]["final_average_macro_f1"]
         - groups[("ft", "egraphsage", seed)]["matrix"]["cl_metrics_all_classes"]["final_average_macro_f1"]
         for seed in mod.SEEDS]
    )
    assert abs(egs["diff"] - expected) < 1e-12


def test_per_period_rows_finalrow_diagonal_forgetting():
    groups = _synthetic_groups()
    finalrow, diagonal, forgetting = mod.per_period_rows(groups)
    n_expected = len(mod.MODELS) * len(mod.PRIMARY_STRATEGIES) * 2 * 5
    assert len(finalrow) == n_expected
    assert len(diagonal) == n_expected
    assert len(forgetting) == len(mod.MODELS) * len(mod.PRIMARY_STRATEGIES) * 2 * 4
    assert finalrow[0]["period"] == 1
    assert forgetting[0]["period"] == 1
    assert all(np.isfinite(row["mean"]) for row in finalrow)


def test_headline_crosscheck(tmp_path):
    means = [
        {"model": "egraphsage", "strategy": "ft", "metric": "final_all", "mean": 0.44, "sd": 0.01, "n": 5, "values": ""},
        {"model": "egraphsage", "strategy": "replay", "metric": "final_all", "mean": 0.41, "sd": 0.01, "n": 5, "values": ""},
    ]
    headline = tmp_path / "headline.csv"
    with headline.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["strategy", "model", "metric", "final_mean", "final_sd"])
        writer.writeheader()
        writer.writerow({"strategy": "ft", "model": "egraphsage", "metric": "all_classes", "final_mean": 0.44, "final_sd": 0.01})
        writer.writerow({"strategy": "replay", "model": "egraphsage", "metric": "all_classes", "final_mean": 0.41, "final_sd": 0.01})
    checks = mod.headline_crosscheck(means, headline)
    assert checks["max_abs_mean_diff"] < 1e-9
    assert len(checks["headline_rows_checked"]) == 2


def test_write_csv_roundtrip(tmp_path):
    rows = [{"a": 1, "b": 2.5}, {"a": 3, "b": 4.5}]
    path = tmp_path / "table.csv"
    mod._write_csv(path, rows)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = list(csv.DictReader(handle))
    assert len(reader) == 2 and reader[1]["a"] == "3"
