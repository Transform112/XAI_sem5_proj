# XDRIFT platforms - execution lanes and setup

This folder is the **only** place with per-machine setup and launch instructions.
All research code lives in `src/`, `scripts/`, `configs/`, `tests/` and is
operating-system neutral: it never hard-codes an absolute path and every entry
point resolves its inputs through `configs/paths/<lane>.yaml` plus `XDRIFT_*`
environment variables (see `src/utils/paths.py`).

## Lane table

| Lane | Machine | Role | torch / PyG | Requirements file |
| --- | --- | --- | --- | --- |
| `rtx5060` | user's PC, RTX 5060 8 GB (Blackwell sm_120, 32 GB RAM) | **canonical** training lane for all headline results | torch 2.7.1+cu128, PyG 2.7.0 | `platforms/requirements/gpu-blackwell-cu128.txt` |
| `legacy` | current dev PC, GTX 1050 3 GB (CUDA 11.1 driver) | dev / smoke / data audit on the legacy GPU or CPU | torch 2.4.1+cu118, PyG 2.6.1 | `platforms/requirements/gpu-legacy-cu118.txt` |
| `kaggle` | Kaggle notebook, Tesla T4 (verified D-029) | parallel-seed replication only, never the canonical lane | image torch 2.11.0+cu128 / py3.13, PyG 2.8.1 | `platforms/requirements/kaggle.txt` |
| `cpu` | any CPU-only box (CI, laptop) | data audit, tests, analysis | torch 2.4.1 CPU, PyG 2.6.1 | `platforms/requirements/cpu.txt` |

Shared dependencies (numpy/pandas/scipy/scikit-learn/networkx/pytest/...) are in
`platforms/requirements/common.txt`; the canonical lanes additionally install a
GPU-specific torch/PyG wheel.

## Path contract (all lanes)

The four writable roots are configurable and every stage respects them:

| Variable | Default (repo-relative) | Contents |
| --- | --- | --- |
| `XDRIFT_DATA` | `data` | `raw/` archives + extracted CSVs, `processed/`, `splits/` frozen masks |
| `XDRIFT_RUNS` | `runs` | checkpoints, `manifests/` run records, `logs/` |
| `XDRIFT_CACHE` | `cache` | intermediate tensors / explanation caches |
| `XDRIFT_RESULTS` | `results` | small trackable tables, `data_audit/` scorecards, figures |
| `XDRIFT_ROOT` | auto-detected repo root | overrides project-root discovery |
| `XDRIFT_GIT` | `git` on PATH | git executable used for the manifest commit SHA |

Lane files: `configs/paths/{default,rtx5060,legacy,kaggle,cpu}.yaml`. A lane file
only overrides what differs; the loader merges it over `default.yaml`. Derived
locations (`datasets/raw`, `datasets/splits`, `results/text/runs/manifests`, `results/text/data_audit`, ...)
are computed in `src/utils/paths.py` - never in stage code.

Verified remote-kit contract: running the entry points with `XDRIFT_DATA` /
`XDRIFT_RUNS` / `XDRIFT_CACHE` / `XDRIFT_RESULTS` pointed at any directory
resolves the raw CSV from `<XDRIFT_DATA>/raw/<dataset>/`, writes scorecards to
`<XDRIFT_RESULTS>/data_audit/`, frozen masks to `<XDRIFT_DATA>/splits/<dataset>/`
and manifests to `<XDRIFT_RUNS>/manifests/`. Checked with:

```powershell
$env:XDRIFT_DATA='<tmp>\data'; $env:XDRIFT_RESULTS='<tmp>\results'
.venv\Scripts\python.exe scripts\run_data_audit.py --dataset NF-UNSW-NB15-v3 --config configs\audit.yaml --dry-run
.venv\Scripts\python.exe scripts\run_graph_lift_pilot.py --dataset NF-UNSW-NB15-v3 --config configs\audit.yaml --dry-run
```

## Canonical lane - RTX 5060 (user's PC)

Run in **Windows PowerShell** (works with 5.1; `pwsh` 7 optional):

```powershell
# 1) one-time environment
powershell -NoProfile -ExecutionPolicy Bypass -File platforms\setup\setup_windows_rtx5060.ps1

# 2) environment check (CUDA required on this lane)
.venv\Scripts\python.exe src\scripts\check_env.py --require-cuda --json-out results\env_check_rtx5060.json

# 3) data identity + data audit (data copied to remote_rtx5060\data\raw with the same layout)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_lane.ps1 -Lane rtx5060 -Config configs\audit.yaml -Entry scripts\verify_datasets.py -- --dataset all
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_lane.ps1 -Lane rtx5060 -Config configs\audit.yaml -Entry scripts\run_data_audit.py -- --dataset NF-UNSW-NB15-v3 --with-pilot
```

`scripts/run_lane.ps1` validates the lane, exports `XDRIFT_*`, runs the entry
script and writes `runs\manifests\<run-name>.json` (git SHA, config hash,
versions, GPU, seed, timestamps; marks the run failed on a non-zero exit).

## Legacy / CPU dev lane (this PC)

```powershell
# one-time: venv + torch 2.4.1+cu118 + PyG 2.6.1 + common deps
powershell -NoProfile -ExecutionPolicy Bypass -File platforms\setup\setup_windows_legacy.ps1

# CUDA is optional here (driver may reject cu118); record which happened
.venv\Scripts\python.exe src\scripts\check_env.py --json-out results\env_check_legacy.json

# small graph-lift pilot on the 3 GB GTX 1050 (sample-scale, ~2 min)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_lane.ps1 -Lane legacy -Config configs\audit.yaml -Entry scripts\run_graph_lift_pilot.py -- --dataset NF-UNSW-NB15-v3
```

Known limitation (fixed in D-018): some Windows hosts lack the MSVC runtime, so a
cu118 torch wheel imports the DLLs but raises `OSError: [WinError 126] c10.dll`.
Fix used here: `pip install msvc-runtime` and copy its DLLs beside
`.venv\Lib\site-packages\torch\lib\` (also applied by the setup script and
`common.txt` for the RTX 5060 lane). If a host still cannot load torch, data
stages run fine without it (pandas/numpy only) and torch-dependent tests skip.

CPU-only analysis (no GPU needed):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r platforms\requirements\cpu.txt
.venv\Scripts\python.exe -m pip install -r platforms\requirements\common.txt
.venv\Scripts\python.exe -m pytest tests
```

## Kaggle lane (parallel-seed replication)

The token is a credential; it is stored outside the repository at
`%USERPROFILE%\.kaggle\kgat.txt` (Bearer `KGAT_...` token, D-029). Smoke-tested
image: Tesla T4, Python 3.13.15, torch 2.11.0+cu128, capability [7.5].

```bash
# inside the Kaggle notebook, after cloning/extracting the repo to /kaggle/working/<repo>
export REPO_DIR=/kaggle/working/xai_proj
bash "$REPO_DIR/platforms/setup/setup_kaggle.sh"     # PyG 2.8.1 + common deps, keeps image torch
```

Push a run kernel from the dev PC with the supported helper (see
`kaggle_lane/README.md`; slug must be `me/<slug>`, metadata id `<owner>/<slug>`):

```powershell
python kaggle_lane\run_kernel.py --code <script.py> --slug <slug> --gpu
```

Kaggle GPU hours are quota-limited: run the reduced seed replication only, tag
every run `lane=kaggle`, and report it as replication, not as an independent
headline result.

## Moving code and artifacts between lanes

```powershell
# full history as a git bundle (small, transfers to Windows or Linux)
python scripts\sync_lane.py bundle --out ..\xai_proj.bundle
# working-tree ZIP (no .git, no data/runs/venv)
python scripts\sync_lane.py archive --out ..\xai_proj.zip
# verify + clone the bundle on the target machine
python scripts\sync_lane.py verify --bundle ..\xai_proj.bundle
python scripts\sync_lane.py clone  --bundle ..\xai_proj.bundle --dest ..\xai_proj_copy
# pack small results (scorecards, split manifests, identity reports) for transfer back
python scripts\sync_lane.py collect --out results_transfer.zip
```

## Environment smoke test

`src/scripts/check_env.py` checks: torch import, CUDA (optional), core
libraries, a toy edge-aware `MessagePassing` forward pass, that edge features
actually change the output through `propagate` (explainer mask contract), the PyG
`Explainer` API at `task_level='edge'` (falls back to node with a recorded
reason), CPU determinism and YAML round-trip. It prints JSON and exits non-zero
on a required-check failure:

```powershell
.venv\Scripts\python.exe src\scripts\check_env.py --json-out results\env_check_<lane>.json
```

## Remote RTX 5060 kit (`remote_rtx5060/`)

The kit is a self-contained snapshot (`repro/`) plus a `datasets/raw/` directory. On
the target PC the user sets the four `XDRIFT_*` variables to the kit directories
and runs the same commands as above. The dry-run contract check above is the
acceptance test that the kit's paths resolve correctly on the target machine.
