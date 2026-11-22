# Analysis config defaults — proposals for `configs/{drift,rq3,ear}.yaml`

*Owner: analyst (T19). Status: **PROPOSAL** — the engineer materialises these into `configs/`; the analyst does not write `configs/` (T19 scope). Authority: `docs/05_EXPERIMENT_PROTOCOL.md` (FROZEN v1) §5, §8–§13; `documents/Doc 2b` §S8/S9/S11/S12; `docs/DECISIONS.md` D-019/D-043/D-044. Changing a numeric value after the first real run requires a `docs/DECISIONS.md` entry.*

## Files

| File | Target config | Stage | Sources |
| --- | --- | --- | --- |
| `drift.yaml` | `configs/drift.yaml` | S8/S9 | docs/05 §8 (drift metrics), §9 (noise floor / excess / H1), §13 (stats); Doc 2b S8–S9; A1–A7 |
| `rq3.yaml` | `configs/rq3.yaml` | S11 | docs/05 §11 (RQ3), §13; Doc 2b S11; D-019; A8–A10 |
| `ear.yaml` | `configs/ear.yaml` | S12 | docs/05 §5 (strategies), §12 (EAR/H4); Doc 2b S12; D-019 |

## Frozen values encoded (must not drift)

**Drift (`drift.yaml`)**
- Metric family: feature cosine, top-k Jaccard k = 3/5/10, Spearman; edge top-m Jaccard m = 5 + cosine; **all reported**, primary fixed = `1 − Jaccard(top-5)` (feature, IG) and edge-mask Jaccard for GNNExplainer (docs/05 §8).
- Noise floor: explainer-repeat R = 10 seeds on a stratified subset (~100 anchors/period); null-update R = 5 same-period reseeds (FT on period j, same architecture, same explainer seed); combined channel optional/robustness. Primary floor channel = `model_null` (A6).
- q95 pooled **per (explainer, class)**, with global-pool fallback below **20** pairs and `threshold_source` recorded per row (A7).
- Excess = `drift_real − drift_null_mean` per anchor; `above_floor = drift_real > q95` (strict); report raw/null/excess/flags side by side.
- IG-repetition drift canary must be exactly 0; model-null drift must be positive (Doc 2b S8 checks).
- H1: expected above-floor fraction 0.05, stratified bootstrap CI, exact binomial + normal approximation, paired Wilcoxon on excess, Holm within (strategy, model, explainer); rank-biserial effect size; retained-anchor fraction reported alongside.
- Aggregation: anchor-level bootstrap unit, 1000 resamples, percentile 95 % (numpy linear, A5), stratified by class, macro over classes.

**RQ3 (`rq3.yaml`)**
- Pre-registered label: δ = 0.05; horizon = `next_or_final` (A10), `any_later` declared robustness only; label from `F_{c,j}(i) = max_{k≤i} R[k,j,c] − R[i,j,c]`; no post-hoc variants.
- Unit `(checkpoint_t, eval_period, class)` with class-level features at checkpoint `i` (A8); no future-F1 features (T19 validator enforces).
- Drift features: mean and 90th-percentile excess drift; structural: mean neighbourhood change; controls: anchor count + class F1 at `i`.
- Competing baselines (D-019 mandatory): confidence mean, margin mean, retained fraction, class F1; majority reference; headline = ΔAUC(drift+baselines − baselines only) with paired bootstrap CI; `h3_supported` iff CI lower bound > 0.
- Model: L2 logistic regression, standardised, C = 1.0; leave-one-period-out CV with primary group = `checkpoint_t` (A9), `eval_period` robustness variant; 1000 resamples for AUC/ΔAUC, 200 for supplementary coefficient CIs; exploratory T = 5 caveat in every row.

**EAR (`ear.yaml`)**
- Replay base identical for all variants: α = 1, 2000-edge class-balanced buffer (ablation 500/2000/5000), 2-hop subgraphs with ≤10 neighbours/hop, identical epochs/lr/early stopping.
- EAR-A: `L_attr = mean(1 − cos(a_t, a_stored))` with differentiable grad×input/1-step-IG surrogate (documented as approximation).
- EAR-B: differentiable soft top-k/rank consistency, k = 5.
- β grid {0.1, 0.5, 1.0} (0.01 only via DECISIONS); selection on validation macro-F1 of old periods + validation drift, never test; same budget for all variants.
- Attribution targets: default = insertion-time checkpoint; ablations = frozen early checkpoint and current model (upper bound on self-reference); stored source IG default, GNNExplainer ablation; hash + provenance per buffer entry; anchor fidelity reported alongside drift (C³L self-reference diagnostic).
- H4: lower drift than plain replay AND ≤ 1 macro-F1 point lost (Δmacro-F1 ≥ −0.01), paired bootstrap CI, failures reported honestly.
- Ablations: buffer size, β grid, attribution source, with/without consistency loss, runtime overhead.

## Cross-references

- Artifact filenames are frozen by **D-043** (see `analysis/artifacts.py`).
- Implementation definitions live in `analysis/{metrics_drift,noise_floor,stats,rq3,metrics_cl}.py` and are authoritative per **D-044**; the pipeline must import them rather than re-implement.
- The S5 writer contract (cl_metrics.csv / cl_matrix.json / cl_per_class_f1.json) is confirmed by **D-043**; the cross-check report is `analysis/reports/s5_crosscheck.md`.
- Real-artifact schema validators: `python -m analysis.validators --stage s8` etc. (see `analysis/validators.py`).
