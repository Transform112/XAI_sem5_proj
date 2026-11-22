"""S8 drift metrics — hand-computed toy cases (docs/05 §8, Doc 2b §S8)."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import metrics_drift as md


def test_identical_explanations_have_zero_drift() -> None:
    vector = np.array([0.5, -1.0, 2.0, 0.0, -0.25])
    similarities = md.attribution_similarity(vector, vector)
    assert similarities["cosine"] == pytest.approx(1.0)
    assert similarities["spearman"] == pytest.approx(1.0)
    for k in (3, 5, 10):
        assert similarities[f"jaccard_top{k}"] == pytest.approx(1.0)
    drifts = md.attribution_drift(vector, vector)
    assert all(value == pytest.approx(0.0) for value in drifts.values())


def test_orthogonal_vectors_cosine_and_topk() -> None:
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 0.0])
    assert md.cosine_similarity(a, b) == pytest.approx(0.0)
    assert md.jaccard_index(md.topk_indices(a, 1), md.topk_indices(b, 1)) == pytest.approx(0.0)
    # k=5 >= d=4 -> both top-5 sets are the full set -> similarity 1 (k is capped, docs/05 §8)
    assert md.primary_feature_drift(a, b) == pytest.approx(0.0)
    # ranks: a -> [4, 1.5, 1.5, 1.5], b -> [1.5, 4, 1.5, 1.5] => rho = -1/3 => drift = 4/3
    assert md.spearman_similarity(a, b) == pytest.approx(-1.0 / 3.0)
    assert md.attribution_drift(a, b)["spearman"] == pytest.approx(1.0 + 1.0 / 3.0)


def test_spearman_drift_is_unclipped_for_reversed_rankings() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([4.0, 3.0, 2.0, 1.0])
    assert md.spearman_similarity(a, b) == pytest.approx(-1.0)
    assert md.attribution_drift(a, b)["spearman"] == pytest.approx(2.0)  # never clipped (A2/A3)


def test_topk_hand_computed_jaccard() -> None:
    a = np.array([1.0, -2.0, 3.0, 0.5, -0.25])  # |a| ranks: idx2, idx1, idx0, idx3, idx4
    b = np.array([1.0, -1.5, 3.0, 2.0, -0.25])  # |b| ranks: idx2, idx3, idx1, idx0, idx4
    assert list(md.topk_indices(a, 3)) == [0, 1, 2]
    assert list(md.topk_indices(b, 3)) == [1, 2, 3]
    # |{0,1,2} ∩ {1,2,3}| = 2, union = 4 -> J = 0.5, drift = 0.5
    assert md.jaccard_index(md.topk_indices(a, 3), md.topk_indices(b, 3)) == pytest.approx(0.5)
    assert md.attribution_drift(a, b)["jaccard_top3"] == pytest.approx(0.5)


def test_topk_tie_break_is_deterministic_and_uses_absolute_values() -> None:
    values = np.array([1.0, 1.0, 1.0, 1.0])
    assert list(md.topk_indices(values, 2)) == [0, 1]
    mixed = np.array([-5.0, 1.0, 2.0, -3.0])
    assert list(md.topk_indices(mixed, 2)) == [0, 3]  # magnitudes 5 and 3


def test_jaccard_empty_convention() -> None:
    assert md.jaccard_index([], []) == pytest.approx(1.0)
    assert md.jaccard_index([1], []) == pytest.approx(0.0)
    assert md.jaccard_index([], [1, 2]) == pytest.approx(0.0)


def test_cosine_zero_norm_convention() -> None:
    zeros = np.zeros(4)
    other = np.array([1.0, 2.0, 0.0, 0.0])
    assert md.cosine_similarity(zeros, zeros) == pytest.approx(1.0)
    assert md.cosine_similarity(zeros, other) == pytest.approx(0.0)
    assert md.attribution_similarity(zeros, zeros)["cosine"] == pytest.approx(1.0)


def test_edge_mask_hand_computed() -> None:
    # top-3 by |mask|: a -> {0,1,2}, b -> {0,1,3} -> J = 2/4 = 0.5
    mask_a = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    mask_b = np.array([0.9, 0.8, 0.1, 0.6, 0.5, 0.4])
    similarities = md.edge_mask_similarity(mask_a, mask_b, m=3)
    assert similarities["edge_jaccard_top5"] == pytest.approx(0.5)  # column name fixed by docs/05 §8
    expected_cosine = float(np.dot(mask_a, mask_b) / (np.linalg.norm(mask_a) * np.linalg.norm(mask_b)))
    assert similarities["edge_cosine"] == pytest.approx(expected_cosine)
    assert md.edge_mask_drift(mask_a, mask_b, m=3)["edge_jaccard_top5"] == pytest.approx(0.5)


def test_edge_mask_explicit_keys_compare_by_key_not_position() -> None:
    mask_a = np.array([0.9, 0.8, 0.7])
    mask_b = np.array([0.9, 0.8, 0.7])
    keys_a = ["edge-x", "edge-y", "edge-z"]
    keys_b = ["edge-x", "edge-y", "edge-z"]
    assert md.edge_mask_similarity(mask_a, mask_b, m=2, keys_a=keys_a, keys_b=keys_b)["edge_jaccard_top5"] == pytest.approx(1.0)
    keys_b = ["edge-z", "edge-y", "edge-x"]
    # top-2 of a = {x, y}; top-2 of b (same values, permuted keys) = {z, y} -> J = 1/3
    assert md.edge_mask_similarity(mask_a, mask_b, m=2, keys_a=keys_a, keys_b=keys_b)["edge_jaccard_top5"] == pytest.approx(1.0 / 3.0)


def test_invalid_inputs_raise_instead_of_contaminating_distributions() -> None:
    with pytest.raises(ValueError):
        md.attribution_similarity(np.array([1.0, np.nan]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        md.cosine_similarity(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        md.edge_mask_similarity(np.array([1.0, np.inf]), np.array([1.0, 0.0]))


def test_pairwise_feature_drift_rows() -> None:
    explanations_a = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, 1.0, 0.0])}
    explanations_b = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([1.0, 0.0, 0.0])}
    rows = md.pairwise_feature_drift(explanations_a, explanations_b)
    assert [row["anchor_id"] for row in rows] == ["a", "b"]
    assert rows[0]["drift_primary"] == pytest.approx(0.0)
    assert rows[1]["drift_primary"] == pytest.approx(
        md.primary_feature_drift(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    )
    assert rows[1]["drift_primary"] == rows[1]["drift_jaccard_top5"]
    with pytest.raises(KeyError):
        md.pairwise_feature_drift({"a": np.zeros(3)}, {"b": np.zeros(3)})


def test_module_documents_conventions() -> None:
    assert md.CONVENTIONS  # the undocumented-by-protocol choices are declared
    assert set(md.CONVENTIONS) == {"topk_ranking", "cosine_zero_norm", "spearman_drift_range", "jaccard_empty", "edge_keys"}
