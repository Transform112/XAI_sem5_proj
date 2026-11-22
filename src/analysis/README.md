# `src/analysis/` — XDRIFT analysis toolkit

Standalone, tested implementation of every quantitative analysis in the frozen
protocol (S5, S8–S13). **No training, no data mutation:** it never touches raw
data, run checkpoints, splits or masks.

## Position in the repo

```text
src/analysis/          code + tests + config proposals
analysis/reports/      human-readable reports written by the validators
results/text/tables/   signed output tables consumed here
results/text/...       audit / explanation / verification evidence
```

The analysis layer reads frozen inputs and writes only:
`results/text/tables/`, `results/text/data_audit/`, `results/images/`, and
`analysis/reports/`.

## Authority (read before changing a definition)

| Source | What it fixes here |
|---|---|
| `docs/05_EXPERIMENT_PROTOCOL.md` (FROZEN v1, D-042) | §8 drift metrics, §9 noise floor/excess/H1, §10 RQ2, §11 RQ3, §12 RQ4, §13 statistics |
| `docs/source_documents/Doc 2b – Implementation Guide Part 2 (S7–S13).md` | S8–S13 stage methods, outputs, checks |
| `docs/source_documents/Doc 2 – Implementation Guide … .md` §S5 | `R[i,j]`, final F1, forgetting, BWT, per-class tensor |
| `paper_writing/literature/claims_evidence_skeleton.md` | binding artifact paths (C1–C6) + `paper_writing/manuscript/numbers.json` schema |
| `docs/DECISIONS.md` | D-033 dual macro-F1, D-030 hierarchical pilot bootstrap, D-019 RQ3 baselines |

If a definition changes, change `docs/05` first (Lead + decision log), then here.
Tests pin the numeric definitions with hand-computed toy cases.

## Quick start

```powershell
# tests (synthetic fixtures; ~30 s once deps are installed)
.\.venv\Scripts\python.exe -m pytest src\analysis -q

# end-to-end synthetic demo: every table/figure + schema-valid numbers file
.\.venv\Scripts\python.exe -m src.analysis.run_synthetic_demo

# S5 writer cross-check (writes analysis/reports/s5_crosscheck.md)
.\.venv\Scripts\python.exe -m src.analysis.s5_crosscheck

# validate real artifacts as each stage lands
.\.venv\Scripts\python.exe -m src.analysis.validators --stage s8
.\.venv\Scripts\python.exe -m src.analysis.validators --stage all --json

# readiness map + independent S8/S9 H1 check
.\.venv\Scripts\python.exe -m src.analysis.pipeline_readiness
.\.venv\Scripts\python.exe -m src.analysis.s8_validation
```

## Modules

| Module | Stage | Responsibility |
|---|---|---|
| `artifacts.py` | S13 | exact artifact path registry + explainer-name normalisation |
| `metrics_drift.py` | S8 | attribution cosine, top-k Jaccard k=3/5/10, Spearman, edge-mask Jaccard@m=5 + cosine; `drift = 1 − similarity` |
| `metrics_cl.py` | S5 | `R[i,j]` tensor, final mean macro-F1, forgetting `F_j`, BWT, per-class F1 tensor (npz+csv), D-033 dual macro, RQ3 later-drop labels |
| `stats.py` | §13 | percentile bootstrap, stratified/paired/hierarchical bootstrap, Spearman + partial Spearman + permutation, Holm, effect sizes, L2-logistic LOPO AUC, paired AUC-Δ bootstrap |
| `noise_floor.py` | S8/S9 | null-update + explainer-repeat + combined distributions, per-(explainer,class) q95, `excess = drift − null mean`, `above_floor`, H1 tests with Holm |
| `rq3.py` | S11 | forgetting labels, feature assembly, competing-baseline ΔAUC (drift vs confidence/margin/retention/F1) |
| `synthetic.py` | — | deterministic protocol-shaped fixtures (tests + demo only) |
| `tables.py` | S13 | writes the skeleton CSV artifacts |
| `figures.py` | S13 | writes `results/images/s8_* … s12_*` |
| `numbers.py` | S13 | `paper_writing/manuscript/numbers.json` builder / validator / table extractor |
| `run_synthetic_demo.py` | — | one-command synthetic proof of the artifact contract + validators |
| `s5_crosscheck.py` | T19 | engineer S5 writer vs this reference; writes `analysis/reports/s5_crosscheck.md` |
| `validators.py` | T19/T20 | schema/column/invariant validators for real artifacts (`--stage s5…s12/all`) |
| `pipeline_fixtures.py` | T20 | schema-faithful synthetic S6 anchor sets / S7 cache fixtures |
| `pipeline_readiness.py` | T20 | artifact → producer CLI → validator readiness report |
| `s8_validation.py` | T20 | independent S8/S9 validation + H1 arithmetic cross-check |
| `config_defaults/` | T19 | proposed `configs/` defaults (analyst does not write `configs/`) |
| `tests/` | — | unit tests for every module |

## Artifact contract

Written to `results/text/tables/` by default:

`cl_metrics.csv`, `cl_per_class_f1.{npz,csv}`, `noise_floor.csv`,
`drift_metrics_ig.csv`, `drift_metrics_gnnexplainer.csv`,
`rq1_{anchor_level,aggregated,h1_tests,strategy_comparison}.csv`,
`rq2_{anchor_level,correlations}.csv`,
`rq3_{features,forgetting_labels,auc}.csv` (+ `rq3_coefficients.csv`),
`rq4_{comparison,ablations}.csv`.

Figures: `results/images/s8_drift_real_vs_null.png`,
`results/images/s8_noise_floor_q95.png`, etc.
Numbers: `paper_writing/manuscript/numbers.json`.

Every table row carries statistics provenance columns (`n`, `ci_lo`, `ci_hi`,
bootstrap resamples, seed, caveat) so the verifier can recompute it.

## Input schemas expected upstream

1. **S5 evaluation result JSON** (`results/text/runs/{strategy}/{model}/seed{s}/…`),
   one per checkpoint either flat or with a `per_period` block:
   ```json
   {"strategy": "ft", "model": "egraphsage", "seed": 42, "checkpoint": 3,
    "lane": "canonical",
    "per_period": {"1": {"macro_f1": 0.61, "per_class_f1": {"Benign": 0.82, "Worms": 0.0}}}}
   ```
2. **Explanation arrays for S8**: `dict[anchor_id -> np.ndarray]` for IG and
   GNNExplainer edge masks over the stored period-j subgraph. Noise floors need
   `dict[anchor_id -> {null_seed -> array}]` with exactly R=5 reseeds and
   `dict[anchor_id -> [repeat_0 … repeat_9]]` with exactly R=10 repeats.
3. **Drift frames**: long tables with at least
   `strategy, model, explainer, seed, anchor_id, class, checkpoint_j,
   checkpoint_t, anchor_period, drift_primary` (+ metric columns).
4. **RQ4 table** (retained, not used for the paper): `model, variant, seed,
   drift_mean, macro_f1, forgetting, bwt, runtime_s`.

## Statistical conventions (docs/05 §13)

- 1000 resamples, percentile 95% CI, stratified by class (and period where
  applicable); paired bootstrap resamples the pair.
- Paired-bootstrap p-value = two-sided bootstrap sign test with +1 correction,
  always reported with CI and effect size (Cohen's d / rank-biserial).
- Holm correction within each RQ family; raw and adjusted p reported.
- RQ3: L2 logistic regression, leave-one-period-out CV, AUC bootstrap CI,
  ΔAUC paired bootstrap; exploratory caveat in every row.
- Pilot only: hierarchical bootstrap (D-030) is the accepted deviation.

## Write scope

`src/analysis/**`, `analysis/reports/**`, `results/text/tables/**`,
`results/text/data_audit/**`, `results/images/**`. The synthetic demo writes
only under `src/analysis/examples/**`.

## Path note

After the repository restructure, the authoritative paths are:
`results/text/tables`, `results/text/data_audit`, `results/text/explanations`,
`results/images`, `paper_writing/manuscript/numbers.json`. Older docs that say
`results/tables`, `results/figures` or `paper/numbers.json` are historical.
