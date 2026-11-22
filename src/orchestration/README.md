# `src/orchestration` — matrix and explain batch planning

Planning helpers for the batch drivers. They answer “which jobs are complete,
which are missing, and what command would produce the missing artifacts?” — they
do not train or explain by themselves.

| Module | Responsibility |
|---|---|
| `matrix_plan.py` | `MatrixRun`, `build_plan`, completion/manifest/checkpoint checks for S3–S5 groups. |
| `explain_plan.py` | `AnchorJob`, `ExplainJob`, `build_jobs`, anchor/explain completion checks. |

## Canonical commands

```powershell
# plan/execute the S3–S5 matrix (see src/scripts/README.md)
.\.venv\Scripts\python.exe src\scripts\run_matrix.py --help

# anchors + explanations driver
.\.venv\Scripts\python.exe src\scripts\run_anchors_explain.py --help
```

Both drivers accept a lane (`cpu`, `kaggle`, `legacy`, `rtx5060`) and a device.
The plan is idempotent: completed groups are skipped only when their manifest
and checkpoint checks pass.

## Invariants

- A job is complete only when the expected metrics/manifest artifact exists and
  its hash/checkpoint validation succeeds.
- Never overwrite a completed signed group in place; write to a new run id and
  update the table through the analysis pipeline.
- The canonical matrix is 30 groups = 2 models × 3 strategies × 5 seeds over 5
  periods. Do not add unregistered groups to the signed count.

## Tests

`src/tests/test_orchestration*.py`, `src/tests/test_matrix_plan*.py`. Run after
installing the environment: `python -m pytest src\tests -q`.
