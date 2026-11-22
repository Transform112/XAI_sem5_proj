# `src/kaggle_lane/kernels/` — Kaggle kernel sources

Notebook/kernel entry points used by the T4 lane:

- `xdrift_bringup/` — environment/bring-up checks.
- `xdrift_matrix/` — S3–S5 continual-learning matrix.
- `xdrift_explain/` — S6/S7 explanation corpus.
- `xdrift_noise/` — F4 noise-floor runs.

## How to execute

Do not run these directly on the laptop. Build/push/poll them through
`src/kaggle_lane/run_full.py` (see `src/kaggle_lane/README.md`); each kernel
expects the staged dataset and writes a run report/zip that the lane downloads
into `src/kaggle_lane/outputs/kaggle_runs/`.
