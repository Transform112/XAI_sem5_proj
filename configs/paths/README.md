# `configs/paths/` — lane path contracts

YAML path configurations loaded by `src/utils/paths.py`:

- `default.yaml` — canonical relative paths for laptops/CPU.
- `kaggle.yaml` — canonical T4 lane (`/kaggle/working/...`).
- `cpu.yaml`, `legacy.yaml`, `rtx5060.yaml` — lane-specific overrides.

## How to use

Entry points receive `--lane`/`--config`; do not hard-code paths. Relative
defaults resolve against the project root (the directory containing `src/`
and `platforms/`). Environment variables (`XDRIFT_DATA`, `XDRIFT_RESULTS`,
`XDRIFT_RUNS`, `XDRIFT_CACHE`) override the defaults.
