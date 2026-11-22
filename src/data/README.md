# `src/data` — data contract, audit, splits, period graphs

This package turns the frozen NetFlow CSV contract into auditable period graphs.
It is the **only** stage allowed to create or validate raw/processed data.

## Responsibilities

| Module | What it does |
|---|---|
| `registry.py` | Dataset registry and dataset specs (`NF-UNSW-NB15-v3`). |
| `identity.py` | CSV identity checks and file hashes. |
| `audit.py` | Schema, duplicates, leakage, support and quality audit. |
| `clean.py` / `quality.py` | Deterministic cleaning and quality summaries. |
| `taxonomy.py` | Label taxonomy and supported-class mapping. |
| `periods.py` | Chronological period construction. |
| `splits.py` | Frozen block split masks (train/validation/test per period). |
| `features.py` | Feature selection/encoding contract. |
| `graphs.py` | Period graph construction and graph manifests. |
| `preprocess.py` | Preprocessing entry points shared by scripts. |
| `structure.py` | Host/flow structural statistics for later analyses. |

## Canonical commands

Run from the repository root. The audited dataset must already exist at
`datasets/raw/NF-UNSW-NB15-v3/NF-UNSW-NB15-v3.csv` (see the root README for hashes).

```powershell
.\.venv\Scripts\python.exe src\scripts\run_data_audit.py --dataset NF-UNSW-NB15-v3 --lane cpu
.\.venv\Scripts\python.exe src\scripts\verify_datasets.py
.\.venv\Scripts\python.exe src\scripts\build_graphs.py --dataset NF-UNSW-NB15-v3 --lane cpu
```

On Linux/Kaggle use `python` instead of the venv path. Outputs go to
`datasets/processed/NF-UNSW-NB15-v3/` and `datasets/splits/NF-UNSW-NB15-v3/`;
audit evidence goes to `results/text/data_audit/`.

## Invariants

- The raw CSV is never modified in place.
- Split masks are frozen and hashed; do not regenerate them casually.
- The period graph manifest is written next to the processed graphs.
- This package does not train models and does not write to `results/text/tables/`.

## Tests

`src/tests/test_data_*.py`, `src/tests/test_splits*.py`, and the data audit
tests under `src/tests/`. Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\tests -q
```

If the local venv is absent, install the environment first with
`platforms/requirements/*` (see root README).
