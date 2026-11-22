"""T30 evaluation memory-bound regression tests (head-only / chunked logits)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tests.helpers_torch import build_toy_graph, requires_torch

pytestmark = requires_torch


def _model_and_data(model_name: str = "egraphsage"):
    from src.models.edge_sage import build_model

    data = build_toy_graph(n_nodes=30, n_edges=180, n_classes=3, seed=0)
    model = build_model(
        model_name,
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        config={"model": {"hidden_dim": 16, "num_classes": 3}},
    )
    return model, data


@pytest.mark.parametrize("model_name", ["egraphsage", "gcn"])
def test_edge_batch_logits_equal_full_slice(model_name):
    from src.eval.predict import predict_logits

    model, data = _model_and_data(model_name)
    model.eval()
    indices = torch.tensor([0, 3, 17, 42, 100, 179], dtype=torch.long)
    bounded = predict_logits(model, data.x, data.edge_index, data.edge_attr, indices, device="cpu")
    with torch.inference_mode():
        full = model(data.x, data.edge_index, data.edge_attr)
    torch.testing.assert_close(bounded, full[indices], rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("model_name", ["egraphsage", "gcn"])
def test_chunked_logits_equal_full_slice(model_name):
    from src.eval.predict import predict_logits

    model, data = _model_and_data(model_name)
    model.eval()
    indices = torch.arange(0, int(data.num_edges), 3, dtype=torch.long)
    chunked = predict_logits(
        model, data.x, data.edge_index, data.edge_attr, indices,
        device="cpu", chunk_edges=7, force_mode="chunked",
    )
    with torch.inference_mode():
        full = model(data.x, data.edge_index, data.edge_attr)
    torch.testing.assert_close(chunked, full[indices], rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("model_name", ["egraphsage", "gcn"])
def test_head_never_projects_full_edge_set_for_split_indices(model_name):
    """Tight-memory regression: head input rows = split size, not all edges."""
    from src.eval.predict import predict_logits

    model, data = _model_and_data(model_name)
    model.eval()
    seen_rows: list[int] = []

    def hook(module, inputs):
        seen_rows.append(int(inputs[0].shape[0]))

    handle = model.head.register_forward_pre_hook(hook)
    try:
        indices = torch.tensor([1, 7, 31, 88], dtype=torch.long)
        out = predict_logits(model, data.x, data.edge_index, data.edge_attr, indices, device="cpu")
    finally:
        handle.remove()
    assert out.shape[0] == len(indices)
    assert seen_rows and max(seen_rows) == len(indices)
    assert max(seen_rows) < int(data.num_edges)


def test_chunked_path_head_rows_bounded_by_chunk():
    from src.eval.predict import predict_logits

    model, data = _model_and_data("egraphsage")
    model.eval()
    seen_rows: list[int] = []

    def hook(module, inputs):
        seen_rows.append(int(inputs[0].shape[0]))

    handle = model.head.register_forward_pre_hook(hook)
    try:
        indices = torch.arange(int(data.num_edges), dtype=torch.long)
        predict_logits(
            model, data.x, data.edge_index, data.edge_attr, indices,
            device="cpu", chunk_edges=11, force_mode="chunked",
        )
    finally:
        handle.remove()
    assert seen_rows and max(seen_rows) <= 11


def test_predict_mode_selection():
    from src.eval.predict import predict_mode

    model, _ = _model_and_data("egraphsage")
    assert predict_mode(model, chunk_edges=0) == "edge_batch"
    # Models with node_embeddings/edge_logits may use the chunked fallback if a
    # caller forces it; the default for our models remains the cheaper edge_batch.
    assert predict_mode(model, chunk_edges=1024) == "edge_batch"


@pytest.mark.parametrize('model_name', ['egraphsage', 'gcn'])
def test_chunked_edge_batch_equals_full_indices(model_name):
    from src.eval.predict import predict_logits

    model, data = _model_and_data(model_name)
    model.eval()
    indices = torch.tensor([0, 3, 17, 42, 100, 179, 5, 12, 60, 150], dtype=torch.long)
    unbounded = predict_logits(model, data.x, data.edge_index, data.edge_attr, indices, device='cpu')
    chunked = predict_logits(model, data.x, data.edge_index, data.edge_attr, indices, device='cpu', chunk_edges=3)
    torch.testing.assert_close(chunked, unbounded, rtol=0.0, atol=1e-6)

