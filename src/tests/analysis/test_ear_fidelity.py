"""Tests for the T57 bounded EAR fidelity diagnostic helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_select_anchor_sample_is_deterministic_and_stratified():
    from scripts.run_ear_fidelity import select_anchor_sample

    def anchor(period, label, anchor_id):
        return {"period": period, "label": label, "anchor_id": anchor_id}

    anchor_sets = {
        1: [anchor(1, 0, 5), anchor(1, 0, 1), anchor(1, 0, 3), anchor(1, 1, 2)],
        2: [anchor(2, 1, 9), anchor(2, 1, 8), anchor(2, 0, 7)],
    }
    sample = select_anchor_sample(anchor_sets, per_class=2)
    ids = [int(item["anchor_id"]) for item in sample]
    assert ids == [1, 3, 2, 7, 8, 9]
    assert select_anchor_sample(anchor_sets, per_class=2) == sample


def test_fidelity_anchor_data_preserves_target_context():
    from scripts.run_ear_fidelity import _anchor_data

    anchor = {
        "n_sub_nodes": 3,
        "sub_edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "sub_edge_attr": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "label": 1,
    }
    data = _anchor_data(anchor)
    assert data.x.shape == (3, 8)
    assert data.edge_index.shape == (2, 2)
    assert data.y.tolist() == [1]
