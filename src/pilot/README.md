# `src/pilot` — graph-lift structure-informativeness pilot

A small, pre-matrix pilot that asks: **does the host-graph structure add
information beyond flow features alone?** It is setup evidence, not one of the
paper's four research questions.

| Module | Responsibility |
|---|---|
| `graph_lift.py` | Build a matched E-GraphSAGE vs feature-only sample, train/evaluate both, bootstrap macro-F1 deltas, write a verdict. |

The pilot uses the same data contract and seeds as the main pipeline. It is
intended to be cheap (small sample, CPU) and to finish before the full matrix.

## Canonical command

```powershell
.\.venv\Scripts\python.exe src\scripts\run_graph_lift_pilot.py --help
```

Outputs go to `results/text/pilot/graph_lift/` (JSON + markdown verdict).
The verdict is diagnostic evidence; do not cite it as a paper result.

## Notes

- The pilot does not change the canonical model/training recipe.
- If the pilot is re-run, record the sample seed and preserve the old outputs
  rather than overwriting them.
- The main S3–S5 matrix is independent of this pilot.

## Tests

`src/tests/test_pilot*.py` (if present) and the graph-lift command's own
`--help` smoke test. Run after installing the environment:
`python -m pytest src\tests -q`.
