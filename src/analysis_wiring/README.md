# `src/analysis_wiring` — S8–S12 adapters

This package connects real, frozen artifacts (checkpoints, anchors, explanation
caches, period graphs) to the analysis layer. It contains **no statistical
tests of its own**; the tests live in `src/analysis`.

| Module | Responsibility |
|---|---|
| `frames.py` | Build drift/noise/structural/retention frames from caches and tables. |
| `pipeline.py` | High-level wiring entry points (`run_real_wiring`, `run_synthetic_wiring`, required-input checks). |

## Why it exists

The signed S8/S9 and S10/S11 results must be reproducible from the artifact
tree, not from an exploratory notebook. The wiring layer:

1. Locates inputs through `src/utils/paths.py` and the artifact registry.
2. Enforces one row per `(group, explainer, anchor, checkpoint_j, checkpoint_t)`
   including the diagonal.
3. Attaches the per-`(explainer, class)` reference-null threshold with the
   documented pooled fallback.
4. Passes the resulting frames to `src/analysis` for statistics.

## Canonical commands

```powershell
# real wiring (S8/S9/…)
.\.venv\Scripts\python.exe src\scripts\run_s10_s11.py --help

# full analysis entry point for the project
.\.venv\Scripts\python.exe src\scripts\run_analysis.py --help
```

Outputs go to `results/text/analysis/` and `results/text/tables/`. The signed
tables are under `results/text/tables/s8_s9_signed/` and
`results/text/tables/s10_rq2/`, `results/text/tables/s11_rq3/`.

## Invariants

- Never re-index or deduplicate pairs silently; the row-count contract is part
  of the signed evidence.
- The diagonal pair `j == t` is included and handled explicitly by the drift
  definition.
- The reference null is per explainer and class; do not pool across explainers
  to increase n.

## Tests

`src/tests/test_analysis_wiring*.py`. Run after installing the environment:
`python -m pytest src\tests -q`.
