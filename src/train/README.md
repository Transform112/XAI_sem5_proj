# `src/train` — deterministic training loop and batching

The training package owns the optimisation loop, class weighting, early
stopping and the two execution paths used across the project.

| Module | Responsibility |
|---|---|
| `trainer.py` | `train_edge_classifier`, evaluation helpers, macro/per-class F1, class weights, early stopping. |
| `fast.py` | Dense host-plan / sampled fast path used on larger period graphs. |
| `sampler.py` | Edge-neighbour sampling, replay-entry sampling and static-aggregation helpers. |

The same trainer is used for fine-tuning, replay and EWC. Continual-learning
strategy objects only define the update/penalty step; they do not own a
separate training loop.

## Canonical commands

```powershell
# one group (model × strategy × seed)
.\.venv\Scripts\python.exe src\scripts\run_train.py --help

# the full S3–S5 matrix
.\.venv\Scripts\python.exe src\scripts\run_matrix.py `
  --dataset NF-UNSW-NB15-v3 --models egraphsage,gcn `
  --strategies ft,replay,ewc --seeds 42,43,44,45,46 --periods 5 `
  --lane cpu --device cpu
```

On Kaggle use the T4 lane and the kernel entry points under
`src/kaggle_lane/kernels/`; the CLI is the same.

## Outputs

- checkpoints and per-group run manifests: `results/text/runs/<run_id>/`;
- per-group metrics: `results/text/tables/` through the evaluation stage;
- logs: `results/text/runs/<run_id>/logs/`.

No file is written under `src/`.

## Invariants

- Seeds are explicit and recorded in the run manifest.
- Early stopping is on validation macro-F1 with the frozen patience.
- `graph_batch` mode is the canonical recipe (D-059); other batching paths are
  diagnostic and must be labelled as such.
- Do not silently change augmentation, class weighting or epoch budget between
  strategies — that would break the continual-learning control.

## Tests

`src/tests/test_train*.py`, `src/tests/test_sampler*.py`, and the fast-path
tests. Run after installing the environment:
`python -m pytest src\tests -q`.
