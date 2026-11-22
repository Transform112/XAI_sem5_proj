# `src/models` — edge-classification GNN definitions

Two model families share one edge-prediction interface used by training,
continual learning and explanation code.

| File | Model | Role |
|---|---|---|
| `edge_sage.py` | `EdgeSAGEModel` (`build_model("egraphsage")`) | Primary E-GraphSAGE detector over host nodes and flow edges. |
| `edge_sage.py` | `EdgeGCNModel` (`build_model("gcn")`) | Simpler GCN baseline to test architecture dependence. |

The models aggregate edge features into host-node embeddings, then classify an
edge from its two endpoint embeddings plus its own flow features. Feature
dimensions, hidden size and class count are driven by `configs/` and the
dataset taxonomy; this package contains no training loop.

## Build and smoke-test

```python
from src.models.edge_sage import build_model

model = build_model("egraphsage", in_dim=..., hidden_dim=..., num_classes=...)
```

The canonical way to train is through `src/scripts/run_train.py` (single group)
or `src/scripts/run_matrix.py` (the S3–S5 matrix). See `src/scripts/README.md`.

```powershell
.\.venv\Scripts\python.exe src\scripts\run_train.py --help
```

## Notes

- The primary comparison in the paper is E-GraphSAGE; GCN is a pre-declared
  architecture control, not a tuning playground.
- Model definitions must stay import-light (no dataset loading at import time)
  so tests and analysis tooling can import them without a GPU.
- Checkpoints are written by the training package, not here.

## Tests

`src/tests/test_models*.py` (shape/forward/registry checks). Run with
`python -m pytest src\tests -q` after installing the environment.
