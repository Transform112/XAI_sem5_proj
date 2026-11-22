# `src/` — all executable code

Every executable module in the project lives here. The tree is organised by
**responsibility**, not by experiment: data → models → training → continual
learning → evaluation → explanations → anchors → analysis → evidence.

```text
src/
├─ data/              audit, preprocessing, frozen splits, period graphs
├─ models/            E-GraphSAGE and GCN definitions
├─ train/             training loop, batching, static-aggregation fast path
├─ cl/                fine-tune / corrected replay / EWC; retained EAR-A code
├─ eval/              R[i,j] matrix, dual macro-F1, forgetting, BWT, prediction
├─ anchors/           canonical anchor selection + retention
├─ explain/           Integrated Gradients, GNNExplainer, cache, fidelity
├─ analysis/          drift/noise/RQ mathematics, validators, tests
├─ analysis_wiring/   S8–S12 wiring adapters
├─ scripts/           CLI entry points (one command per stage)
├─ kaggle_lane/       Kaggle dataset build, kernel push/poll/download
│  └─ remote_rtx5060/ frozen Kaggle input contract + code snapshot helper
├─ verification/      independent verifier command-line scripts
├─ presentation/      editable 15-slide deck source and assets
├─ tests/             repository-wide unit/integration tests
├─ orchestration/     stage orchestration helpers
├─ pilot/             graph-lift pilot (setup evidence)
└─ utils/             path resolution, config loading, hashes, manifests
```

## Canonical execution

Run from the repository root. All paths resolve through `src/utils/paths.py`
(`XDRIFT_ROOT` wins; otherwise the root is found by walking up to the directory
containing `src/` and `platforms/`).

```powershell
# tests
.\.venv\Scripts\python.exe -m pytest src\tests src\analysis\tests -q

# stage entry points (see src/scripts/README.md for arguments)
.\.venv\Scripts\python.exe src\scripts\run_data_audit.py --dataset NF-UNSW-NB15-v3 --lane cpu
.\.venv\Scripts\python.exe src\scripts\build_graphs.py --dataset NF-UNSW-NB15-v3 --lane cpu
.\.venv\Scripts\python.exe src\scripts\run_matrix.py --dataset NF-UNSW-NB15-v3 --models egraphsage,gcn --strategies ft,replay,ewc --seeds 42 --periods 5 --lane cpu --device cpu --max-runs 2
.\.venv\Scripts\python.exe src\scripts\run_explain.py --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage --seed 42 --period 1 --lane cpu --device cpu
.\.venv\Scripts\python.exe src\scripts\build_noise_floor.py --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage --seed 42 --period 1 --lane cpu --device cpu
```

On Linux/Kaggle use `python` instead of `.\.venv\Scripts\python.exe`.

## Folder READMEs

| Folder | README |
|---|---|
| `src/data/` | [data/README.md](data/README.md) — audit, splits, feature contract, period graphs |
| `src/models/` | [models/README.md](models/README.md) — E-GraphSAGE and GCN definitions |
| `src/train/` | [train/README.md](train/README.md) — training loop, batching, early stopping |
| `src/cl/` | [cl/README.md](cl/README.md) — fine-tune, replay, EWC, retained EAR-A |
| `src/eval/` | [eval/README.md](eval/README.md) — R-matrix, dual macro-F1, forgetting/BWT |
| `src/anchors/` | [anchors/README.md](anchors/README.md) — canonical anchor selection |
| `src/explain/` | [explain/README.md](explain/README.md) — IG, GNNExplainer, cache, fidelity |
| `src/analysis_wiring/` | [analysis_wiring/README.md](analysis_wiring/README.md) — S8–S12 artifact adapters |
| `src/orchestration/` | [orchestration/README.md](orchestration/README.md) — matrix/explain batch planning |
| `src/scripts/` | [scripts/README.md](scripts/README.md) — every CLI entry point and argument |
| `src/analysis/` | [analysis/README.md](analysis/README.md) — analysis toolkit + validators |
| `src/pilot/` | [pilot/README.md](pilot/README.md) — graph-lift setup pilot |
| `src/utils/` | [utils/README.md](utils/README.md) — paths, config, hashes, manifests, seeds |
| `src/kaggle_lane/` | [kaggle_lane/README.md](kaggle_lane/README.md) — Kaggle dataset/kernel lane |
| `src/verification/` | [verification/README.md](verification/README.md) — verifier scripts |
| `src/tests/` | [tests/README.md](tests/README.md) — tests and fixtures |
| `src/presentation/` | [presentation/README.md](presentation/README.md) — deck build |

## Import rules

- Absolute imports only: `from src.analysis import ...`, `from src.utils.paths import ...`.
- Do not add repository-root `sys.path` hacks in new modules; use
  `python -m src.<module>` or run a `src/scripts/*.py` entry point.
- Entry points deliberately add the repository root to `sys.path` exactly once
  so they work when invoked by path from any cwd.

## Where results go

No module writes into `src/`. Outputs go to:

- `results/text/` — tables, JSON, TXT, Markdown;
- `results/images/` — chart PNGs only;
- `results/text/runs/` — run manifests and logs;
- `datasets/` — raw/processed/splits/manifests.

See the root [`README.md`](../README.md) for the full path contract.
