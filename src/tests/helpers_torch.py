"""Shared torch/PyG toy fixtures for model/data tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.torch_support import torch_available, torch_unavailable_reason

requires_torch = pytest.mark.skipif(
    not torch_available(),
    reason=f"torch unavailable: {torch_unavailable_reason()}",
)


def build_toy_graph(n_nodes: int = 30, n_edges: int = 180, n_classes: int = 3, seed: int = 0, n_features: int = 6):
    """Small deterministic edge-classification graph with train/val/test masks."""
    import torch
    from torch_geometric.data import Data

    rng = np.random.default_rng(seed)
    src = rng.integers(0, n_nodes, n_edges)
    dst = rng.integers(0, n_nodes, n_edges)
    edge_index = torch.as_tensor(np.stack([src, dst]).astype("int64"))
    edge_attr = torch.as_tensor(rng.normal(size=(n_edges, n_features)).astype("float32"))
    y = torch.as_tensor((np.arange(n_edges) % n_classes).astype("int64"))
    torch.manual_seed(seed)
    shuffle = torch.randperm(n_edges)
    y = y[shuffle]
    x = torch.ones((n_nodes, 4), dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    n = int(data.num_edges)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[: int(0.7 * n)] = True
    val_mask[int(0.7 * n) : int(0.8 * n)] = True
    test_mask[int(0.8 * n) :] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.node_global_ids = torch.arange(n_nodes, dtype=torch.int64)
    data.edge_row_ids = torch.arange(n_edges, dtype=torch.int64)
    data.edge_ts = torch.arange(n_edges, dtype=torch.int64) * 1000
    data.edge_host_src_global = edge_index[0].clone()
    data.edge_host_dst_global = edge_index[1].clone()
    data.period = 0
    data.dataset = "toy"
    return data
