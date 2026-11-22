# RTX 5060 Kit — Prerequisites and Troubleshooting

## Hardware / OS expectations

| Item | Expected | Notes |
| --- | --- | --- |
| GPU | NVIDIA RTX 5060, 8 GB (Blackwell, compute capability 12.0 / sm_120) | CUDA 12.8+ wheels required; the kit pins torch 2.7.1+cu128 |
| Driver | NVIDIA driver supporting CUDA 12.8 or newer | If `nvidia-smi` reports an older CUDA version, update the driver before step 02 |
| RAM | 32 GB recommended | Data preprocessing peaks well below that; PyG explainers are GPU-bound |
| Disk | ≥ 40 GB free on the drive holding the kit | data ~15 GB extracted + .venv ~6 GB + runs/caches |
| OS | Windows 10/11 x64 | PowerShell 5.1 is sufficient; no admin rights required |
| Python | 3.10–3.12 x64, on PATH as `python` | Python 3.12 recommended (matches the main workstation) |
| Git | NOT required | `03_get_code.ps1` unpacks a zip snapshot; git bundle is also provided if present |

## What step 02 installs

- `.venv` at the kit root (never touches the system Python).
- torch 2.7.1+cu128 from `download.pytorch.org`, PyTorch Geometric 2.7.0, and the common
  research stack (numpy/pandas/scipy/scikit-learn/matplotlib/seaborn/pyyaml/tqdm/networkx/pytest/
  requests/httpx/arxiv/semanticscholar/bibtexparser).
- `msvc-runtime` (Windows only) and copies its DLLs beside `torch/lib` — required on PCs
  without the Visual C++ Redistributable (this exact failure was observed on the main PC).
- Writes `output/env/env_check_rtx5060.json` and `output/env/pip-freeze-rtx5060.txt`.

## First-run checklist

1. `nvidia-smi` shows the RTX 5060 and a CUDA version ≥ 12.8.
2. `python -V` prints a supported 64-bit Python.
3. Step 01 reports enough free disk (≥ 40 GB).
4. Step 02 finishes with `"ok": true` in `output/env/env_check_rtx5060.json`; in particular:
   `cuda_available: true`, `device_name` contains `RTX 5060`, `pyg_explainer.ok: true`,
   `toy_message_passing.mask_effect: ok`.
5. Step 05 `-Stage env` re-runs the check from the extracted code snapshot.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `torch.cuda.is_available()` False | old driver / CPU-only wheel installed | update NVIDIA driver; re-run step 02 (it installs cu128 explicitly) |
| `WinError 126 ... c10.dll` | missing MSVC runtime | step 02 installs `msvc-runtime` and copies DLLs; if it still fails, install the VC++ redistributable manually |
| `nvidia-smi` not found | driver not installed | install the NVIDIA driver for RTX 5060 |
| OOM during explanations | 8 GB VRAM with large subgraphs | step 05 has `-Stage explain` defaults with small batches; reduce `batch_size` in the config it prints |
| pip download very slow | first install is ~5 GB | let it run; the script is idempotent and resumes on rerun |
| `Unauthorized` from Kaggle | token not set (only for `04_get_data.ps1 -Mode download`) | set `$env:KAGGLE_API_TOKEN` (see below) |

## Kaggle (optional, for data download only)

The kit's Kaggle use is limited to dataset download if you did not copy `data/raw` from the
main workstation. Set the token in the session where you run step 04:

```powershell
$env:KAGGLE_API_TOKEN = '<token from the main workstation>'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\04_get_data.ps1 -Mode download
```

Do **not** write the token into any file inside `repro/` or `output/`. Parallel Kaggle GPU runs
are launched from the main workstation, not from this PC.
