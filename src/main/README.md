# Main pipeline scripts

This folder is the new, numbered entry-point layer. Each script is one complete
stage and can be run from the repository root. The old `src/scripts/` tree is
kept only while these replacements are being written; it will be removed once
the new chain is complete.

The design goals are:

- one file per major stage, not a web of small launchers;
- notebook-style stdout (numbered cells, `[ OK ]` / `[WARN]` / `[FAIL]`) so a
  Kaggle run log reads like an executed notebook;
- machine-readable JSON written next to the human output;
- each stage checks its inputs and stops before doing expensive work if they
  are missing.

## Stage order

| Step | Script | What it does |
|---|---|---|
| 01 | `01_setup_environment.py` | Checks Python/Kaggle runtime, scientific packages, GPU/CUDA, repo paths, **dataset presence**, and writable output folders. Writes `results/text/env_checks/env_check.json`. Does **not** download data. |
| 02 | `02_clean_preprocess.py` | Cleans all four NF-v3 CSVs (exact dedupe, sentinel handling, chronological sort, host/class ids, near-duplicate groups, five periods), fits the period-1 **train-only** preprocessor, and records a balance snapshot in the JSON summary. Writes the cleaned table + preprocessor state per dataset, plus ten diagnostic figures under `results/images/preprocessing/<dataset>/`. Does **not** split, resample, or train. |
| 03 | `03_eda.py` | Explains the cleaned data before any split/SMOTE: class support, period mix, attack rate, frozen split support, missingness, feature distributions, correlation, temporal shift, graph-side host/pair stats, class profiles, and an **illustrative** balance preview. Writes 15 figures and a markdown report per dataset. Does **not** resample or train. |
| 04 | `04_split_balance.py` | Builds the deterministic grouped 70/10/20 split for every period, verifies it against the frozen masks (period 1 must match the signed mask), saves split masks + manifest, and writes per-period training indices plus **class weights** and an optional benign cap. No SMOTE, no synthetic rows, no val/test resampling. Writes four split/balance figures. |
| 05 | `05_build_graphs.py` | Builds one PyG graph per period from the frozen cleaned table, preprocessor, split masks and balance indices. Supports `constant`, `reputation`, and `constant+reputation` node features, writes a graph manifest plus five structural figures. Does **not** train. |
| 06 | `06_train.py` | Trains the graph variants sequentially over periods with the frozen `graph_batch` trainer, builds the R matrix, and writes run JSON/CSV, depth/reputation figures and a Markdown report. Smoke/pilot runs are engineering checks; 5-seed Kaggle runs are the paper evidence. |
| 07 | `07_explain.py` | Loads a step-06 checkpoint, selects canonical anchors with a fixed graph context, and writes IG / GNNExplainer results for every checkpoint `M_t (t >= j)` to the explanation cache + index. No retraining; context variants get a separate anchor/cache namespace. |
| 08 | `08_drift.py` | Turns the step-07 corpus into an **uncalibrated** raw drift table: IG 1−Jaccard(top-k features), GNNExplainer 1−Jaccard(top-k edge mask), cross and diagonal rows, by-class/by-period summaries and four figures. No null floor and no H1 claim. |
| 08B | `08b_null.py` | Trains fresh model-null checkpoints for each period and explains the same stored anchors, producing the matching IG null rows and q95 floor. No test-time resampling, no graph rebuild. |
| 08C | `08c_calibrate.py` | Attaches the 08B floor to the 08A cross rows and reports above-floor fraction, excess, class/period breakdowns and cluster-bootstrap CIs. Diagnostic only; not signed S8. |
| 09 | `09_rq2_structural.py` | Exploratory RQ2 on the new corpus: joins calibrated IG drift to per-anchor 2-hop host-neighbourhood change, reports Spearman/partial-Spearman/permutation by period pair and writes figures + report. One seed; not signed S10. |

Dataset acquisition/download is a separate task and is intentionally not part of
the numbered modelling chain. Step 02 expects the raw CSVs to already exist
under `datasets/raw/<dataset>/` (or the configured `XDRIFT_DATA` root).

## Diagnostic tools

`src/main/tools/profile_raw_dataset.py` is not a numbered stage. It profiles the raw CSV
before the cleaning script is written, using only the standard library, and
writes `results/text/data_audit/raw_dataset_profile.json`. The resulting
recommendations are in `analysis/raw_dataset_profile.md`. See
`src/main/tools/README.md` for usage.

## Running step 01
From the repository root:

```bash
python src/main/01_setup_environment.py
```

Kaggle GPU lane (CUDA required by default on Kaggle):

```bash
python src/main/01_setup_environment.py --lane kaggle
```

Kaggle CPU notebook (explicitly allow no GPU):

```bash
python src/main/01_setup_environment.py --lane kaggle --allow-cpu
```

Explicit dataset location, if it is not in the default `datasets/raw/...` or a
standard `/kaggle/input/...` location:

```bash
python src/main/01_setup_environment.py --csv /kaggle/input/nf-unsw-nb15-v3/NF-UNSW-NB15-v3.csv
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--dataset NAME` | dataset name, default `NF-UNSW-NB15-v3` |
| `--data-root PATH` | override the datasets root |
| `--csv PATH` | explicit raw CSV path |
| `--lane NAME` | path lane; defaults to `kaggle` on Kaggle, `cpu` elsewhere |
| `--json-out PATH` | report path, relative to the repo root |
| `--require-cuda` | fail if CUDA is unavailable (default on Kaggle) |
| `--allow-cpu` | allow CPU-only Kaggle runs |
| `--hash-dataset` | also record the raw CSV SHA-256 |

Exit codes: `0` ready (possibly with warnings), `1` environment/configuration
failure, `2` environment okay but the raw CSV is missing.


## Running step 02
From the repository root (all four NF-v3 datasets are attempted; datasets with
no raw CSV are reported as `SKIP`):

```bash
python src/main/02_clean_preprocess.py
python src/main/02_clean_preprocess.py --dataset NF-UNSW-NB15-v3
python src/main/02_clean_preprocess.py --dataset all --strict   # missing raw data fails
```

Outputs per dataset:

```text
datasets/processed/<dataset>/clean/cleaned.parquet   # or .pkl fallback
datasets/processed/<dataset>/clean/preprocessor_state.json
results/text/data_audit/clean/<dataset>.json          # counts, paths, hashes, warnings
results/images/preprocessing/<dataset>/fig01..fig10   # data/preprocessing diagnostics
```

Ten figures are written for each cleaned dataset:

1. raw vs cleaned row count (exact deduplication);
2. attack-class distribution (log scale);
3. rows per chronological period;
4. class mix by period;
5. sentinel values converted to NaN by column;
6. preprocessing-recipe composition (categorical / log1p / standardise / missing indicators);
7. raw vs log1p values for a heavy-tailed flow feature;
8. skew before/after log1p (top eight columns);
9. categorical cardinality;
10. class counts in the full data vs the period-1 training split.

The preprocessor is fitted only on period-1 **training** rows. If frozen split
masks already exist under `datasets/splits/<dataset>/<period strategy>/`, they
are reused; otherwise the same deterministic split is built in memory for
fitting only — the masks are not written here. The next stage owns the frozen
split and the class-balance metadata. Applying SMOTE before splitting would leak
synthetic copies across train/validation/test, so the order is deliberately
**split → balance training only → transform**; this project uses class weights
instead of SMOTE.


## Running step 03 (EDA)
Step 03 reads the cleaned table and preprocessor state written by step 02, plus
the frozen period-1 masks. It is read-only: no data is written except figures,
one JSON summary, and one markdown report per dataset.

```bash
python src/main/03_eda.py
python src/main/03_eda.py --dataset NF-UNSW-NB15-v3
python src/main/03_eda.py --dataset all --strict
```

Outputs per dataset:

```text
results/images/eda/<dataset>/fig01..fig15          # 15 diagnostic figures
results/text/data_audit/eda/<dataset>.json         # machine-readable summary
analysis/eda/<dataset>.md                          # human-readable report
```

The 15 figures cover: cleaned rows; class counts (full vs period-1 train);
class mix by period; attack rate by period; frozen split support; missingness
and sentinels; key feature distributions; feature spread by class; correlation
heatmap; temporal feature shift; host/pair summary; class mean profiles; the
balance preview; feature shift caused by the preview sampler; benign share
before/after the preview.

The balance preview caps benign rows (`--max-benign`, default 30,000) and
repeats minority rows with replacement (`--min-class`, default 2,000). It is
illustrative only: it does not write a balanced dataset, does not touch
validation/test, and is not SMOTE. It exists to show what balancing would do
before the real split/balance stage runs.


## Running step 04 (split + class balance)
Step 04 reads the cleaned table and rebuilds the **same** leakage-safe split
used by the frozen protocol. It refuses to replace the signed period-1 mask if
the rebuilt indices differ, and it never resamples validation or test rows.

```bash
python src/main/04_split_balance.py
python src/main/04_split_balance.py --dataset NF-UNSW-NB15-v3
python src/main/04_split_balance.py --benign-cap 100000   # optional undersampling
python src/main/04_split_balance.py --balance-method none  # control run
```

Outputs per dataset:

```text
datasets/splits/<dataset>/<strategy>/period_{1..5}_masks.npz
datasets/splits/<dataset>/<strategy>/split_manifest.json
datasets/processed/<dataset>/balance/period_{1..5}_train_indices.npz
datasets/processed/<dataset>/balance/balance_summary.json
results/text/data_audit/split/<dataset>.json
results/images/split/<dataset>/fig01..fig04.png
```

What the balance means:

- **class weights** are `N_train / (K * n_c)`, normalized so the mean training
  sample weight is one; the trainer can pass them directly to its loss;
- `--benign-cap` optionally keeps at most that many benign rows per period in
  the training view only; it does not touch validation or test;
- **no SMOTE and no synthetic rows**. SMOTE before splitting leaks synthetic
  copies across splits; on a graph, a synthetic edge also has no real
  source/destination host, so it cannot be explained or placed in context.
  The graph-safe balance is class weights plus an optional benign cap.

The four figures show class support in the P1 split, class weights per period,
the benign-cap effect, and split proportions by period.


## Running step 05
`05_build_graphs.py` is the bridge from tabular data to PyG. It consumes only
frozen inputs: the cleaned table, the period-1 train-only preprocessor state,
the frozen split masks/manifest, and the step-04 training indices.

```bash
# fast contract check (no torch needed): split hash, sizes, node features
python src/main/05_build_graphs.py --dry-run --node-features constant,reputation

# canonical graph: constant eight-dimensional nodes
python src/main/05_build_graphs.py --dataset NF-UNSW-NB15-v3

# host reputation variant: causal, label-free prior-period features
python src/main/05_build_graphs.py --dataset NF-UNSW-NB15-v3 --node-features reputation

# both blocks: 8 constants + 8 reputation features
python src/main/05_build_graphs.py --dataset NF-UNSW-NB15-v3 --node-features constant+reputation

# build more than one variant in one invocation
python src/main/05_build_graphs.py --node-features constant,reputation
```

Outputs:

```text
datasets/graphs/<dataset>/<mode>/period_{1..5}.pt
datasets/graphs/<dataset>/<mode>/graphs_manifest.json
results/text/data_audit/graphs/<dataset>/<mode>.json
results/images/graphs/<dataset>/fig01..fig05.png
```

The graph manifest records the mode, node-feature names and dimension, split
hash, per-period edge/node counts, graph file hashes, and the preprocessor
state. Reputation uses periods `< p` only; period 1 is a cold start. The
schedule for depth/context experiments is documented in
`analysis/experiments/reputation_and_hop_plan.md`.


## Running step 06 (training comparison)
Step 06 loads the frozen graph snapshots, trains sequentially over periods, and
compares node-feature and depth variants under the same seed. It does not
re-clean, re-split, re-fit the preprocessor, or resample validation/test rows.

```bash
# fast engineering smoke (one period, one epoch, small masks; CPU OK)
python src/main/06_train.py --smoke --device cpu

# single variant
python src/main/06_train.py --dataset NF-UNSW-NB15-v3 --node-features constant --depths 2 --seeds 42

# depth/reputation comparison
python src/main/06_train.py --dataset NF-UNSW-NB15-v3 \
  --node-features constant,reputation,constant+reputation \
  --depths 2,3,4 --strategy ft --seeds 42

# resume a stopped Kaggle kernel; completed configurations are skipped
python src/main/06_train.py --dataset NF-UNSW-NB15-v3 \
  --node-features constant,reputation,constant+reputation \
  --depths 2,3,4 --strategy ft --seeds 42,43,44,45,46 --resume
```

The five graph snapshots per mode are loaded once, kept on the training device,
and reused for every depth and seed; step 05 is never called from this script.
Progress is written after every completed run, so a stopped kernel loses at most
one run. `--resume` skips runs whose full signature (variant, depth, seed,
epochs, mask caps, balancing, period count) was already written.

Outputs:

```text
results/text/train/<dataset>/runs.json
results/text/train/<dataset>/runs.csv
results/images/train/<dataset>/fig01..fig05.png
analysis/train/<dataset>.md
```

The R matrix entry `R[i,j]` is test macro-F1 on period `j` after the checkpoint
trained through period `i`. By default all lower-triangular cells are filled so
final average F1 and A11 forgetting can be computed; use `--diagonal-only` for a
faster but partial matrix. `--use-balance` trains on the step-04 selected
indices instead of the natural train mask. Smoke and pilot runs are tests, not
paper evidence; the final evidence is the 5-seed Kaggle matrix.

Imbalance experiments are opt-in and leave the frozen default unchanged:
`--loss focal --focal-gamma 2` (focal loss), `--class-weight-cap 100` (clip
extreme class weights), `--balanced-batches` (sample real training edges with
replacement ∝ 1/class_count; graph-valid rare-class exposure),
`--node-noise 0.1` (Gaussian node-feature jitter during training only), and
`--coarse-taxonomy` (diagnostic 7-class merge; not the frozen label space).
`--save-checkpoints` writes per-period model payloads under
`datasets/processed/<dataset>/models/<run_id>/period_<i>.pt` for explanations.


## Running step 07 (explanation corpus)
Step 07 never retrains. It loads a step-06 checkpoint and its frozen graph for
each anchor period, selects correct/confident test anchors with a fixed graph
context, and explains every anchor with every later checkpoint `M_t (t >= j)`.
The sampled context is stored once and reused for all checkpoints.

```bash
# contract test: one epoch-friendly small run, two anchors, both explainers
python src/main/07_explain.py --dataset NF-UNSW-NB15-v3 \
  --run-id ft-none-d2-s42 --periods 1 --max-anchors 2 --max-explanations 4

# full seed-42 corpus (all periods, all checkpoints; default 2-hop/10-fanout/2000-edge context)
python src/main/07_explain.py --dataset NF-UNSW-NB15-v3 \
  --run-id ft-none-d2-s42 --periods 1,2,3,4,5

# context ablation (separate anchor/cache namespace; same checkpoint)
python src/main/07_explain.py --dataset NF-UNSW-NB15-v3 \
  --run-id ft-none-d2-s42 --periods 1 --max-anchors 5 \
  --hops 3 --edge-cap 5000
```

Outputs:

```text
datasets/processed/<dataset>/explanations/anchors/<run_id>/period_<j>.pt
results/text/explanations/                  # explanation cache (gitignored)
results/text/tables/explain_index_main.csv  # cache index
results/text/data_audit/explanations/<dataset>/<run_id>/summary.json
```

`--anchor-tag` names a context variant; the default canonical context is
`2-hop / fanout 10 / 2000 edges`. The cache key includes the context digest, so
a 3-hop explanation can never be mistaken for the canonical 2-hop one.

**Completed diagnostic corpus.** The seed-42 baseline run `ft-none-d2-s42`
(constant 8-d node features, depth 2) produced 1,200 anchors
(406/208/209/235/142 by period), 4,101 anchor-checkpoint pairs, and 8,202
IG + GNNExplainer records with full coverage. The full cache is local-only under
`src/kaggle_lane/outputs/xdrift-step07-corpus/`; the tracked manifest is
`results/text/data_audit/explanations/NF-UNSW-NB15-v3/ft-none-d2-s42/corpus_manifest.json`.
This corpus is **not** the signed S7 corpus: its anchor count and checkpoint
model differ, so it must be analysed separately.
`--checkpoint-root` and `--graph-root` point at downloaded Kaggle artifacts
without copying them into the repo. The writer currently supports constant
8-d node features; reputation-model explanations need the anchor builder to
store the sampled node features first.

## Running step 08 (raw drift, no calibration yet)
Step 08 reads the step-07 corpus, recomputes IG drift from one attribution vector
per row (so top-5 and top-10 are not double-counted), and writes raw drift tables.
It does **not** build a null floor and does **not** make an H1 claim.

```bash
# tables only
python src/main/08_drift.py --corpus-root src/kaggle_lane/outputs/xdrift-step07-corpus/corpus \\
  --run-id ft-none-d2-s42 --no-figures

# tables + four figures
python src/main/08_drift.py --corpus-root src/kaggle_lane/outputs/xdrift-step07-corpus/corpus \\
  --run-id ft-none-d2-s42
```

Outputs:

```text
results/text/tables/drift_main/drift_rows_all.csv
results/text/tables/drift_main/drift_rows_cross.csv
results/text/tables/drift_main/drift_by_class.csv
results/text/tables/drift_main/drift_by_period_pair.csv
results/text/data_audit/drift/drift_raw_summary.json
results/images/drift/fig01..fig04.png
analysis/experiments/step08_raw_drift.md
```

A `scipy`-based Spearman is used when available; a pandas/NumPy fallback keeps
the stage runnable in the bundled local Python. The primary metric is
`1 - Jaccard(top-5)`; the diagonal `t=j` rows must be zero.

## Running step 08B (matching null floor)
Step 08B trains fresh "null" models for each period (same architecture and data
period, different seed), explains the **same stored anchors/contexts** with IG,
and records the drift between null model pairs. This is the matching floor for
08A: models that saw no real update between checkpoints.

```bash
python src/main/08b_null.py --dataset NF-UNSW-NB15-v3 \\
  --run-id ft-none-d2-s42 --periods 1,2,3,4,5 \\
  --max-anchors-per-period 100 --null-seeds 5 --epochs 30
```

Outputs:

```text
results/text/tables/null_ig/null_rows.csv
results/text/tables/null_ig/null_floor.csv
results/text/data_audit/null_ig/null_summary.json
```

The floor uses per-class q95 when a class has at least 30 null rows, otherwise a
global q95 fallback. Null rows are model-null; they are not explainer-repeat
rows. The Kaggle kernel `src/kaggle_lane/kernels/step08b_null.py` runs the same
logic with GPU training and exports these three files.

## Running step 08C (calibrated drift)
Step 08C attaches the 08B floor to every 08A cross row, then reports how often
real drift exceeds that floor. It is a diagnostic on the new corpus, not the
signed S8 result; GNNExplainer still has no matching floor.

```bash
python src/main/08c_calibrate.py
```

Outputs:

```text
results/text/tables/drift_calibrated/calibrated_rows.csv
results/text/tables/drift_calibrated/calibrated_by_class.csv
results/text/tables/drift_calibrated/calibrated_by_period_pair.csv
results/text/data_audit/drift/drift_calibrated_summary.json
results/images/drift/fig01_above_floor_vs_expected.png
results/images/drift/fig02_excess_distribution.png
results/images/drift/fig03_above_floor_by_class.png
analysis/experiments/step08_calibrated.md
```

The 5% expectation is the no-drift binomial rate. A fraction below 5% (and a
negative mean excess) means real drift is not larger than model-null noise;
that is the current diagnostic result.

## Running step 09 (exploratory RQ2)

Step 09 joins the calibrated IG drift rows to the local 2-hop host-neighbourhood
change computed from the frozen period graphs and the step-07 anchor bundles.
It is a new-corpus diagnostic, **not** signed S10, and it has no matched
null-update comparison yet.

```bash
python src/main/09_rq2_structural.py \
  --corpus-root src/kaggle_lane/outputs/xdrift-step07-corpus/step07_corpus \
  --run-id ft-none-d2-s42
```

Outputs:

```text
results/text/tables/rq2_new/rq2_anchor_level.csv
results/text/tables/rq2_new/rq2_correlations.csv
results/text/data_audit/rq2_new/summary.json
results/images/rq2_new/fig01_scatter.png
results/images/rq2_new/fig02_spearman_by_pair.png
analysis/experiments/step09_rq2_new_corpus.md
```

The host-neighbourhood change is almost discrete on this 40-host graph
(`0` or `0.0625`), so the analysis is deliberately reported as exploratory.
Partial correlations controlling anchor degree are usually near zero or
negative; do not interpret raw Spearman values as H2 support.

## Dependency ordering
Step 01 is intentionally import-safe: it is written with the standard library
plus optional imports, so it can report a missing `torch`/`pyyaml` stack
without crashing before it writes the report. Later stages may import the full
project stack directly.
