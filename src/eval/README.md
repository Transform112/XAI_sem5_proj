# `src/eval` — continual-learning metrics and prediction

The evaluation package owns the metric definitions used by the signed S3–S5
tables and RQ3. It does not train anything; it consumes checkpoints or logits
and writes metric tables.

| Module | Responsibility |
|---|---|
| `cl_metrics.py` | `R[i,j]` matrix, forgetting/BWT (A11 convention), dual macro-F1 (10-class and supported 7-class). |
| `predict.py` | Checkpoint loading and prediction (`predict_logits`, `predict_mode`, supported-set helpers). |

## Metric contract

- **Dual macro-F1** is always reported: all 10 classes and the supported set of
  7 classes. Do not report only the favourable variant.
- **R[i,j,c]** = F1 of checkpoint *i* evaluated on period *j* test edges for
  class *c*.
- **Forgetting** follows the pre-declared A11 convention; BWT is reported
  alongside it and is not interchangeable with it.
- Class support counts accompany per-class tables; rare classes use the
  documented D-033 stratification.

## Canonical commands

```powershell
# evaluate one trained group
.\.venv\Scripts\python.exe src\scripts\run_eval.py --help

# the matrix runner evaluates every group as it finishes
.\.venv\Scripts\python.exe src\scripts\run_matrix.py --help
```

Outputs go to `results/text/tables/` (for example
`results/text/tables/s5_baseline/`) and `results/text/runs/` manifests.

## Tests

`src/tests/test_eval*.py`, `src/tests/test_cl_metrics*.py`. Run after installing
the environment: `python -m pytest src\tests -q`. The signed S3–S5 numbers are
reproduced by the verifier checks documented in
`results/text/verification/` and `docs/VERIFICATION_REPORT_*`.
