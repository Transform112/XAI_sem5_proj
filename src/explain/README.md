# `src/explain` — explainers, cache and fidelity

This package produces the attribution artifacts that later drift/noise analyses
consume. Explanations are expensive, so every result is cached by a stable key
and reused.

| Module | Responsibility |
|---|---|
| `ig.py` | Integrated Gradients (`integrated_gradients`, `top_k_features`). Deterministic. |
| `gnn_explainer.py` | GNNExplainer edge/feature mask. Stochastic across runs. |
| `fidelity.py` | Feature and edge-mask fidelity diagnostics. |
| `cache.py` | `ExplanationCache`, `cache_key`, batch cache helpers. |

## Canonical contract

- **IG** is deterministic: same flow + checkpoint + baseline gives the same
  attribution.
- **GNNExplainer** is stochastic: repeated runs can differ; this is why the
  noise floor has an explainer-repeat channel (R=10).
- Every cached explanation records: flow/anchor id, checkpoint, explainer,
  seed, context hash, and the stored period-j context.
- The primary drift metric is `1 − Jaccard(top-5 IG features)` before and after
  a model update; secondary metrics are reported too.

## Canonical commands

```powershell
# one explainer run on one group
.\.venv\Scripts\python.exe src\scripts\run_explain.py --help

# batch anchors + explain for a group
.\.venv\Scripts\python.exe src\scripts\run_anchors_explain.py --help
```

Outputs go under `results/text/explanations/<group>/` and the explanation index
`results/text/explanations/index.csv`. Noise caches go under
`results/text/explanations_noise/`.

## Invariants

- Do not change the top-k value or baseline after seeing drift results; the
  metric is pre-registered.
- Cache keys include the context hash so a changed graph cannot silently reuse
  an old explanation.
- GNNExplainer runs must record the seed; repeated runs are separate rows.

## Tests

`src/tests/test_ig*.py`, `src/tests/test_explain*.py`,
`src/tests/test_fidelity*.py`. Run after installing the environment:
`python -m pytest src\tests -q`.
