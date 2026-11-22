# XDRIFT — graph-based network intrusion detection pipeline

This repository contains the **complete, runnable XDRIFT pipeline** for temporal
flow classification with graph neural networks. It is a code and configuration
repository: raw datasets, processed tensors, run outputs and paper artifacts are
not committed.

The pipeline is a sequence of numbered, notebook-style Python stages. Each stage
prints numbered cells and writes a machine-readable JSON report.

## Pipeline

| Stage | Script | Purpose |
|---|---|---|
| 01 | `src/main/01_setup_environment.py` | Verifies Python, packages, GPU/CUDA, project paths and dataset presence. |
| 02 | `src/main/02_clean_preprocess.py` | Cleans the CSV, deduplicates exact rows, builds five periods, fits the period-1 train-only preprocessor and writes diagnostic figures. |
| 03 | `src/main/03_eda.py` | Explores class support, period composition, missingness, feature distributions and temporal shift. |
| 04 | `src/main/04_split_balance.py` | Builds the deterministic grouped 70/10/20 split and class weights; supports optional benign undersampling. No SMOTE. |
| 05 | `src/main/05_build_graphs.py` | Builds one PyG graph per period with constant, reputation or combined node features. |
| 06 | `src/main/06_train.py` | Trains sequential fine-tuning/replay/EWC models and builds the lower-triangular R matrix. |
| 07 | `src/main/07_explain.py` | Produces IG and GNNExplainer explanations from saved checkpoints. |
| 08 | `src/main/08_drift.py` | Computes raw explanation drift for IG and GNNExplainer. |
| 08B | `src/main/08b_null.py` | Trains model-null checkpoints and builds matching null floors. |
| 08C | `src/main/08c_calibrate.py` | Compares real drift with the null q95 floor. |
| 09 | `src/main/09_rq2_structural.py` | Exploratory RQ2 association between drift and neighbourhood change. |
| 10 | `src/main/10_decoder_compare.py` | Compares fine, prior-adjusted and hierarchical decoders on saved checkpoints. |

`src/main/README.md` contains the detailed stage-by-stage commands, flags,
output paths and caveats.

## Repository layout

```text
.
├─ README.md
├─ requirements.txt
├─ configs/                  # dataset, path and training configuration
├─ platforms/                # lane requirements and setup helpers
└─ src/
   ├─ main/                  # numbered end-to-end pipeline stages
   ├─ data/                  # raw loading, cleaning, periods, preprocessing
   ├─ models/                # E-GraphSAGE and GCN definitions
   ├─ train/                 # trainer and metrics
   ├─ cl/                    # fine-tune, replay, EWC strategies
   ├─ eval/                  # continual-learning evaluation
   ├─ anchors/               # anchor selection and graph contexts
   ├─ explain/               # IG, GNNExplainer and explanation cache
   ├─ analysis/              # reusable drift/noise/statistical analysis code
   ├─ analysis_wiring/       # adapters between pipeline stages
   ├─ orchestration/         # pipeline/run orchestration
   ├─ pilot/                 # smoke/pilot helpers
   ├─ utils/                 # paths, config, logging
   ├─ scripts/               # legacy wrappers kept for compatibility
   ├─ tests/                 # test suite
   ├─ verification/          # independent verification scripts
   └─ kaggle_lane/           # Kaggle dataset/kernel helpers and kernels
```

## Requirements

```bash
python -m pip install -r requirements.txt
```

`torch` and `torch-geometric` are installed separately for the chosen lane; see
`platforms/requirements/` and `platforms/setup/`.

## Data

Place the raw CSV at:

```text
datasets/raw/<dataset>/<dataset>.csv
```

The default supported datasets are configured in `configs/datasets/` and
`configs/audit.yaml`:

- `NF-UNSW-NB15-v3`
- `NF-ToN-IoT-v3`
- `NF-BoT-IoT-v3`
- `NF-CSE-CIC-IDS2018-v3`

Data is not committed. Stage 02 also accepts a custom data root through the
existing `XDRIFT_DATA` path configuration.

## Quick start

```bash
python src/main/01_setup_environment.py
python src/main/02_clean_preprocess.py --dataset NF-UNSW-NB15-v3
python src/main/03_eda.py --dataset NF-UNSW-NB15-v3
python src/main/04_split_balance.py --dataset NF-UNSW-NB15-v3
python src/main/05_build_graphs.py --dataset NF-UNSW-NB15-v3
python src/main/06_train.py --dataset NF-UNSW-NB15-v3 --node-features constant --depths 2 --seeds 42
```

The later stages reuse outputs from earlier stages. Use `--help` on every stage
for the full argument list.

## Outputs

Generated data and results stay local and are ignored by Git:

```text
datasets/processed/<dataset>/
results/text/
results/images/
```

Stage reports are written under `results/text/data_audit/`.

## Notes

- The pipeline is research code, not a production service.
- The split, preprocessing and training stages are intentionally separated so
  preprocessing and balancing are fitted on training partitions only.
- Smoke/pilot runs are engineering checks and are not paper evidence.
- The repository does not include the raw datasets, model checkpoints or paper
  artifacts.
