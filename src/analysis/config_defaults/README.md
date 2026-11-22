# `src/analysis/config_defaults/` — analysis config defaults

Fallback YAML defaults used by the analysis toolkit when a project config does
not override a key (`drift.yaml`, `ear.yaml`, `rq3.yaml`). Paths use the
post-restructure layout: `results/text/tables`, `results/text/anchors`,
`results/images`.

## How to use

Keep defaults aligned with `configs/`; the effective config is loaded by
`src/utils/config.py`. Change a default only with a matching update in
`configs/` and a decision note in `docs/DECISIONS.md`.
