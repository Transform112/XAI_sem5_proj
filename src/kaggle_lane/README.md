# `src/kaggle_lane/` — Kaggle GPU lane (canonical, D-065)

Kaggle T4 is the **canonical headline lane**. The laptop is CPU/diagnostic only.
This package prepares the private input dataset, pushes kernels, polls them,
downloads outputs and validates artifacts. It uses the Kaggle API with a token
stored **outside** the repo.

## Status

| Item | Value |
|---|---|
| Input dataset | `mysteriousavailable/xdrift-nf-unsw-c1-inputs` (private, immutable carrier) |
| Run-control dataset | `mysteriousavailable/xdrift-runctl` |
| Resume dataset | `mysteriousavailable/xdrift-resume` |
| Matrix kernel | `mysteriousavailable/xdrift-matrix` |
| Explain kernel | `mysteriousavailable/xdrift-explain` |
| Noise kernel | `mysteriousavailable/xdrift-noise` |
| Bring-up kernel | `mysteriousavailable/xdrift-t4-bring-up-canonical-lane-d-065` |
| Canonical environment | Python 3.13.15 · torch 2.11.0+cu128 · CUDA 12.8 · PyG 2.8.1 · Tesla T4 · 15,360 MiB |
| Preserved run reports | `results/text/kaggle_runs/` |
| GPU budget at sign-off | ≈10–11 GPU-h used; no GPU jobs left running |

## Token handling

Store the KGAT token at `%USERPROFILE%\.kaggle\kgat.txt` (Windows) or
`~/.kaggle/kgat.txt`; never commit it.

```powershell
$env:KAGGLE_API_TOKEN = (Get-Content "$env:USERPROFILE\.kaggle\kgat.txt" -Raw).Trim()
```

## Files

| File | Purpose |
|---|---|
| `kaggle_client.py` | Minimal Bearer-auth client (token check, kernel push/poll/output) |
| `run_kernel.py` | Build kernel dir, `kaggle kernels push`, poll, download output |
| `bringup.py` | T4 bring-up driver: push → wait → download → verify R matrix → report |
| `run_full.py` | Long-run driver: matrix / explain / noise stages, chunks, resume |
| `build_dataset.py` | Build/check the private input dataset staging folder |
| `push_smoke.py` | Original GPU smoke test (superseded by `bringup.py`) |
| `kernels/` | Kernel source (`script.py` + `kernel-metadata.json`) |
| `remote_rtx5060/make_code_snapshot.ps1` | Snapshot the committed HEAD for the Kaggle code payload |
| `remote_rtx5060/frozen/` | Frozen Kaggle input contract (hashes/config) |
| `runs/`, `outputs/` | Local-only generated kernel/output dirs (gitignored) |

## Dataset build and refresh

The input dataset contains the raw NF-UNSW-NB15-v3 CSV, frozen C1 artifacts,
a code snapshot of the committed HEAD, and hash manifests.

```powershell
$env:KAGGLE_API_TOKEN = (Get-Content "$env:USERPROFILE\.kaggle\kgat.txt" -Raw).Trim()

# 1) snapshot committed HEAD (dirty files excluded)
powershell -NoProfile -ExecutionPolicy Bypass -File src\kaggle_lane\remote_rtx5060\make_code_snapshot.ps1

# 2) build/verify the staging folder
.\.venv\Scripts\python.exe src\kaggle_lane\build_dataset.py
.\.venv\Scripts\python.exe src\kaggle_lane\build_dataset.py --check

# 3) create first version (once) or refresh
.\.venv\Scripts\python.exe -m kaggle datasets create `
  -p src\kaggle_lane\outputs\dataset_stage -t -r skip

.\.venv\Scripts\python.exe -m kaggle datasets version `
  -p src\kaggle_lane\outputs\dataset_stage -t -r skip `
  -m "refresh: <reason>" --dir-mode zip

.\.venv\Scripts\python.exe -m kaggle datasets status mysteriousavailable/xdrift-nf-unsw-c1-inputs
```

Staging dir: `incoming_rtx5060/kaggle_bringup/dataset_stage/` (local-only,
≈582 MB). Kaggle auto-extracts uploaded zips; the kernel handles both zip and
extracted layouts.

## Kernel execution

```powershell
$env:KAGGLE_API_TOKEN = (Get-Content "$env:USERPROFILE\.kaggle\kgat.txt" -Raw).Trim()

# bring-up
.\.venv\Scripts\python.exe src\kaggle_lane\bringup.py --timeout-s 14400 --poll-s 30

# 30-group canonical matrix
.\.venv\Scripts\python.exe src\kaggle_lane\run_full.py --stage matrix `
  --run-id t37-matrix-full-90e7635 --commit 90e7635 --dataset-version 3

# explanation smoke / full corpus
.\.venv\Scripts\python.exe src\kaggle_lane\run_full.py --stage explain --smoke `
  --run-id t37-explain-smoke-90e7635 --commit 90e7635 --dataset-version 3
.\.venv\Scripts\python.exe src\kaggle_lane\run_full.py --stage explain `
  --max-anchors 1000 --max-jobs 30 --max-pushes 6 `
  --run-id t37-explain-full-90e7635 --commit 90e7635 --dataset-version 3

# noise floor
.\.venv\Scripts\python.exe src\kaggle_lane\run_full.py --stage noise `
  --run-id t42-noise-v7-3f3dbad --commit 3f3dbad --dataset-version 7
```

Outputs land under `src/kaggle_lane/outputs/kaggle_runs/<run-id>/` locally and in
`results/text/kaggle_runs/` once preserved. The kernels write to
`/kaggle/working/` using `configs/paths/kaggle.yaml`:

```text
/kaggle/input/xdrift-nf-unsw-c1-inputs   # read-only
/kaggle/working/datasets                 # writable data root
/kaggle/working/results/text/runs        # writable runs
/kaggle/working/results/text/tables      # tables
/kaggle/working/results/images            # figures
```

## In-kernel stage commands

The kernel payload runs modules from the snapshot, not from this local tree:

```bash
python src/scripts/build_graphs.py --dataset NF-UNSW-NB15-v3 --lane kaggle
python src/scripts/build_anchors.py --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage --seed 42 --lane kaggle
python src/scripts/run_explain.py --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage --seed 42 --period 1 --lane kaggle
python src/scripts/build_noise_floor.py --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage --seed 42 --period 1 --lane kaggle
python src/scripts/run_analysis.py --mode real --lane kaggle --out-root . \
  --anchors-dir results/text/anchors \
  --cache-root results/text/explanations \
  --noise-cache-root results/text/explanations_noise \
  --cl-matrix results/text/tables/cl_matrix.json \
  --graphs-dir datasets/processed/NF-UNSW-NB15-v3
```

## T4 timing reference

| Step | Duration |
|---|---|
| PyG install (`torch-geometric==2.8.1`) | 5.7 s |
| `check_env --require-cuda` | 11.4 s |
| `build_graphs.py` (5 periods) | 64.2 s |
| Canary matrix group `ft/egraphsage/seed42` | 73.7 s |
| `run_eval.py` | 10.3 s |
| Full 30-group matrix (train + eval) | ≈1.23 GPU-h (4598 s inner wall) |
| Explanation corpus (signed) | staged; see `results/text/kaggle_runs/` |

Measured T4 epoch: **2.402 s** at period-1 size, **0.59–0.62 s** at periods 2–5.

## Notes

- `kaggle_lane/README.md` older copies may mention `results/tables` or
  `data/processed`; after the restructure the authoritative paths are
  `results/text/tables` and `datasets/processed`.
- Never write to `/kaggle/input`; set `XDRIFT_DATA` to `/kaggle/working/datasets`.
- Kernel slugs are derived from titles on first creation; always use the exact
  refs above.
