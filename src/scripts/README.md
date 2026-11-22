# `src/scripts/` — stage entry points

One CLI per pipeline stage. Run every command **from the repository root** so
`src.utils.paths.find_project_root()` resolves correctly. On Windows use
`.\.venv\Scripts\python.exe`; on Linux/Kaggle use `python`.

All stages accept `--lane {cpu,legacy,kaggle,rtx5060}` (default `cpu`) and most
accept `--device {auto,cpu,cuda}`. `--dataset NF-UNSW-NB15-v3` selects
`configs/datasets/NF-UNSW-NB15-v3.yaml`.

## Pipeline order

```text
download_data.ps1
  → run_data_audit.py
  → build_graphs.py
  → run_train.py / run_matrix.py
  → run_eval.py
  → build_anchors.py
  → run_explain.py
  → build_noise_floor.py
  → run_analysis.py (S8/S9 drift + H1)
  → run_s10_s11.py (RQ2 + RQ3)
  → validators / verifier scripts
```

## Commands

### Data

| Script | Purpose | Example |
|---|---|---|
| `download_data.ps1` | Optional Windows downloader for the public NetFlow-v3 family | `powershell -ExecutionPolicy Bypass -File src\scripts\download_data.ps1` |
| `check_env.py` | Environment/torch/CUDA/PyG checks | `python src/scripts/check_env.py --require-cuda --json-out results/text/env_checks/env_check.json` |
| `run_data_audit.py` | Identity/schema/duplicates/leakage audit, frozen split masks | `python src/scripts/run_data_audit.py --dataset NF-UNSW-NB15-v3 --lane cpu` |
| `build_graphs.py` | Build one PyG graph per period | `python src/scripts/build_graphs.py --dataset NF-UNSW-NB15-v3 --lane cpu` |
| `run_graph_lift_pilot.py` | Setup-evidence graph-lift pilot | `python src/scripts/run_graph_lift_pilot.py --dataset NF-UNSW-NB15-v3 --lane cpu` |

**Outputs:** `results/text/data_audit/`, `datasets/splits/NF-UNSW-NB15-v3/`,
`datasets/processed/NF-UNSW-NB15-v3/period_*.pt`.

### Training / evaluation

| Script | Purpose | Example |
|---|---|---|
| `run_train.py` | One model × strategy × seed | `python src/scripts/run_train.py --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage --seed 42 --lane cpu --device cpu` |
| `run_matrix.py` | Full strategy matrix, resumable | `python src/scripts/run_matrix.py --dataset NF-UNSW-NB15-v3 --models egraphsage,gcn --strategies ft,replay,ewc --seeds 42 --periods 5 --lane cpu --device cpu --max-runs 2` |
| `run_eval.py` | S5 R matrix, dual macro-F1, per-class F1, forgetting/BWT | `python src/scripts/run_eval.py --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage --seed 42 --lane cpu --device cpu` |
| `ewc_lambda_pilot.py` | Pre-registered EWC λ pilot | `python src/scripts/ewc_lambda_pilot.py --dataset NF-UNSW-NB15-v3 --lane cpu --device cpu` |

**Checkpoints:** `results/text/runs/<strategy>/<model>/seed<seed>/M_*.pt`.
**S5 tables:** `results/text/tables/cl_metrics.csv`, `cl_matrix.json`,
`cl_per_class_f1.json`.

### Explanations

| Script | Purpose | Example |
|---|---|---|
| `build_anchors.py` | Canonical anchor sets | `python src/scripts/build_anchors.py --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage --seed 42 --lane cpu --device cpu` |
| `run_explain.py` | IG + GNNExplainer corpus (one group/period) | `python src/scripts/run_explain.py --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage --seed 42 --period 1 --lane cpu --device cpu` |
| `run_anchors_explain.py` | Combined anchor+explanation helper | `python src/scripts/run_anchors_explain.py --help` |
| `build_noise_floor.py` | F4 explainer repeat (R=10) + model reseed (R=5) | `python src/scripts/build_noise_floor.py --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage --seed 42 --period 1 --lane cpu --device cpu` |

**Outputs:** `results/text/anchors/`, `results/text/explanations/`,
`results/text/tables/explain_index.csv`, `results/text/tables/noise_floor.csv`.

### Analysis

| Script | Purpose | Example |
|---|---|---|
| `run_analysis.py` | S8/S9 drift, excess, above-floor, H1 tests | `python src/scripts/run_analysis.py --mode real --lane cpu --out-root results/text/analysis --anchors-dir results/text/anchors --cache-root results/text/explanations --noise-cache-root results/text/explanations_noise --cl-matrix results/text/tables/cl_matrix.json --graphs-dir datasets/processed/NF-UNSW-NB15-v3` |
| `run_s10_s11.py` | RQ2 correlations + RQ3 AUC/ΔAUC | `python src/scripts/run_s10_s11.py --s8-tables results/text/tables/s8_s9_signed --full-root results --graphs-dir datasets/processed/NF-UNSW-NB15-v3 --runs-root results/text/runs` |

**Outputs:** `results/text/tables/rq1_*.csv`, `results/text/tables/s10_rq2/`,
`results/text/tables/s11_rq3/`.

### EAR (retained, dropped from paper)

| Script | Purpose |
|---|---|
| `run_ear_comparison.py` | EAR-A vs replay comparison (no signed output; RQ4 dropped) |
| `run_ear_fidelity.py` | EAR fidelity/augmentation diagnostic |

Do **not** quote EAR numbers; RQ4 was formally dropped under Doc 1 §7.4.

### Lane / diagnostics

| Script | Purpose |
|---|---|
| `run_lane.py` | Lane-aware stage launcher |
| `run_lane.ps1` / `run_lane.sh` | Windows / POSIX wrappers |
| `sync_lane.py` | Copy/validate lane artifacts |
| `verify_datasets.py` | Dataset identity/hash checks |
| `annotate_scorecard_decision_inputs.py` | Add decision provenance to scorecard |

## Environment variables

- `XDRIFT_ROOT` — project root (optional; auto-discovered otherwise).
- `XDRIFT_DATA`, `XDRIFT_RUNS`, `XDRIFT_CACHE`, `XDRIFT_RESULTS` — override the
  four roots.
- `KAGGLE_API_TOKEN` — required only for Kaggle lane scripts.

## Notes

- All scripts write outside `src/`; the source tree is read-only at runtime.
- `--max-runs` and `--max-*` flags exist for smoke tests; production runs should
  not cap the matrix.
- Prefer `python -m src.analysis.<module>` for the analysis/validator tools;
  they are documented in [`src/analysis/README.md`](../analysis/README.md).
