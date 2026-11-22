"""Tests for S10/S11 analysis helpers (T55; CPU-only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.tests.helpers_torch import requires_torch


@requires_torch
def test_build_cl_tensors_from_signed_per_class_csv(tmp_path):
    from scripts.run_s10_s11 import _build_cl_tensors

    classes = [
        "Benign", "Analysis", "Backdoor", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
    ]
    rows = []
    for i in range(1, 6):
        for j in range(1, i + 1):
            for c, name in enumerate(classes):
                rows.append(
                    {
                        "strategy": "ft",
                        "model": "egraphsage",
                        "seed": 42,
                        "checkpoint": i,
                        "eval_period": j,
                        "class": name,
                        "f1": float(i + j + c) / 20.0,
                    }
                )
    path = tmp_path / "per_class_f1.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    tensors = _build_cl_tensors(path)
    assert set(tensors) == {("ft", "egraphsage", 42)}
    tensor = tensors[("ft", "egraphsage", 42)]
    assert tensor.R_class is not None and tensor.R_class.shape == (5, 5, 10)
    assert tensor.classes == tuple(classes)
    assert np.isfinite(tensor.R_class[0, 0, 0])
    assert np.isnan(tensor.R_class[0, 1, 0])  # future evaluation cell not present


@requires_torch
def test_host_presence_conditions_on_endpoint_membership():
    from scripts.run_s10_s11 import _host_presence

    class _Graph:
        def __init__(self, ids):
            self.node_global_ids = torch.tensor(ids, dtype=torch.long)

    graphs = {1: _Graph([10, 20, 30]), 2: _Graph([10, 20])}
    neighbourhood = pd.DataFrame(
        {
            "anchor_id": [1, 1, 2],
            "checkpoint_t": [1, 2, 2],
        }
    )
    meta = {
        1: {"host_src_global": 10, "host_dst_global": 20},
        2: {"host_src_global": 20, "host_dst_global": 30},
    }
    presence = _host_presence(neighbourhood, meta, graphs)
    assert presence.tolist() == [True, True, False]
