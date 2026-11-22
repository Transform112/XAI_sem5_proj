# `src/tests/` — repository-wide tests

Tests live next to the code they pin but are grouped here for a single
repository-wide run. Analysis-specific tests are in `src/analysis/tests/`.

## Run

```powershell
# from the repository root
.\.venv\Scripts\python.exe -m pytest src\tests src\analysis\tests -q
```

On Linux/Kaggle use `python -m pytest ...`.

## What is covered

- data audit / preprocessing / frozen split invariants;
- graph construction and fingerprint stability;
- model shapes and setup checks;
- training loop mechancs (synthetic data, CPU);
- S5 metric definitions: dual macro-F1, forgetting, BWT, R matrix;
- anchor selection schema and caps;
- explainer cache keys and fidelity contracts;
- analysis metrics (Jaccard/cosine/Spearman/Pearson), bootstrap, Holm, AUC;
- validators and artifact schema checks;
- path/config resolution (lane YAML, env overrides, root discovery).

## Fixtures

`src/tests/conftest.py` and `src/analysis/tests/conftest.py` derive the repo root
from the test file (`parents[2]` / `parents[3]` after the restructure). Tests use
deterministic synthetic fixtures where possible; real-data tests are marked
slow and skip when the payload is absent.

## Notes

- The system Python in this environment has no torch/numpy/pandas/pytest
  installed, so the restructure only ran `compileall` and path checks. Install
  `platforms/requirements/cpu.txt` into `.venv` before claiming test passes.
- Do not add tests that hard-code `data/`, `results/tables` or `paper/`; use
  `src.utils.paths.load_paths()` and `src.analysis.artifacts`.
