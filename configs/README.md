# `configs/` — frozen experiment and path configuration

All YAML here is part of the frozen protocol. Do not edit a frozen value
without a decision entry in `docs/DECISIONS.md` and a verifier-visible reason.

```text
configs/
├─ base.yaml                # shared defaults
├─ audit.yaml               # data audit
├─ train.yaml               # canonical training recipe (30 epochs, patience 5)
├─ train_ewc_lambda100.yaml # EWC λ=100 selected by pre-registered pilot
├─ eval.yaml                # S5 evaluation
├─ explain.yaml             # S7 explainers/context
├─ drift.yaml               # S8/S9 drift + noise
├─ rq3.yaml                 # RQ3 labels/features
├─ ear.yaml                 # EAR (retained; RQ4 dropped)
├─ datasets/                # per-dataset schema/labels
│  ├─ NF-UNSW-NB15-v3.yaml
│  └─ …
└─ paths/
   ├─ default.yaml          # canonical local paths
   ├─ cpu.yaml              # CPU diagnostic lane
   ├─ legacy.yaml           # GTX 1050 dev lane
   ├─ rtx5060.yaml          # historical canonical lane
   └─ kaggle.yaml           # canonical Kaggle T4 lane
```

## Path contract (after restructure)

`defaults` in `configs/paths/default.yaml`:

```yaml
data_root: datasets
runs_root: results/text/runs
cache_root: cache
results_root: results
```

Derived in `src/utils/paths.py`:

```text
tables_root       = results/text/tables
figures_root      = results/images
audit_root        = results/text/data_audit
explanations_root = results/text/explanations
```

`configs/paths/kaggle.yaml` overrides:

```yaml
data_root: /kaggle/working/datasets
runs_root: /kaggle/working/results/text/runs
cache_root: /kaggle/working/cache
results_root: /kaggle/working/results
```

## Environment overrides

`XDRIFT_ROOT`, `XDRIFT_DATA`, `XDRIFT_RUNS`, `XDRIFT_CACHE`, `XDRIFT_RESULTS`
override the YAML. Relative values resolve against the project root, never the
current working directory.

## Usage

```powershell
# lane selection
.\.venv\Scripts\python.exe src\scripts\run_data_audit.py --dataset NF-UNSW-NB15-v3 --lane cpu

# explicit configs
.\.venv\Scripts\python.exe src\scripts\run_train.py --config configs\train.yaml --dataset-config configs\datasets\NF-UNSW-NB15-v3.yaml
```

## Rules

- Frozen configs carry hashes in run manifests; never silently change a recipe.
- `configs/paths/*` are the only place writable roots are declared.
- Dataset-specific schemas live under `configs/datasets/`; the primary dataset
  is `NF-UNSW-NB15-v3.yaml`.
