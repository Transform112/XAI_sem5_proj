# `src/utils` — paths, config, hashes, manifests, seeds

Shared, import-light utilities used by every other package.

| Module | Responsibility |
|---|---|
| `paths.py` | Project-root discovery, `Paths` resolution, lane path configs, env overrides. |
| `config.py` | YAML config loading, deep merge, overrides, validation. |
| `manifest.py` | Canonical JSON, file hashing, run manifests and code fingerprints. |
| `seed.py` | Deterministic seeding for Python/NumPy/torch. |
| `logging.py` | Project logging setup. |
| `memory.py` | Memory budget checks (RAM/VRAM diagnostics). |
| `gates.py` | Preflight gate checks for lane/experiment requirements. |
| `artifact_paths.py` | Helpers for resolving artifact paths from the registry. |
| `torch_support.py` | Optional torch import/device helpers. |

## Path contract

The repository root is found by walking up until a directory contains both
`src/` and `platforms/`. Environment variables override lane files:
`XDRIFT_ROOT`, `XDRIFT_DATA`, `XDRIFT_RUNS`, `XDRIFT_CACHE`, `XDRIFT_RESULTS`.

Defaults after the restructure:

```text
data_root      = datasets
runs_root      = results/text/runs
cache_root     = results/text/cache
results_root   = results
tables_root    = results/text/tables
audit_root     = results/text/data_audit
explanations   = results/text/explanations
figures_root   = results/images
```

Lane overrides live in `configs/paths/{default,cpu,kaggle,legacy,rtx5060}.yaml`.

## Quick checks

```powershell
.\.venv\Scripts\python.exe -c "from src.utils.paths import find_project_root; print(find_project_root())"
.\.venv\Scripts\python.exe -c "from src.utils.paths import load_paths; p=load_paths(); print(p.data_root); print(p.tables_root)"
```

## Invariants

- Never hardcode an absolute machine path in code; resolve it through this
  package or an entry point.
- Do not add repository-root `sys.path` hacks to library modules; entry points
  handle that once.
- Manifest hashes are canonical JSON with sorted keys; changing that format
  invalidates signed evidence.

## Tests

`src/tests/test_paths*.py`, `src/tests/test_config*.py`,
`src/tests/test_manifest*.py`. Run after installing the environment:
`python -m pytest src\tests -q`.
