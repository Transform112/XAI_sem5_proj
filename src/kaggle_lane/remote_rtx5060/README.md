# RTX 5060 Execution Kit — start here

> **Historical kit note.** This folder is a self-contained transfer kit for the RTX 5060 PC.
> Commands below use paths relative to the kit root (`remote_rtx5060/`); the live repository
> code now lives under `src/` and the main workflow in the root `README.md`.

**Status: kit v8 (2026-11-02).** Gate G1c has passed and Protocol v1 is frozen: the data stream
(C1 session-aware 1+4), splits/masks (frozen grouped hash `010a80d9…`), thresholds and
rare-class/label policies are final. This kit ships the frozen split masks + preprocessor state in
`frozen/`, so the RTX 5060 PC can build graphs and run G2 timing without re-running the audit.
It contains the **T27 optimized trainer** (D-059: `batching="graph_batch"` canonical — transductive
full-node propagation with precomputed static edge aggregation, 1024-target-edge loss batches,
774 updates/epoch) plus the gated `matrix`/`explain` drivers. Expected full-size epoch time on the
RTX 5060: **~2–3 s/epoch for E-GraphSAGE, ~3–4 s/epoch for GCN** (legacy 3 GB card: 3.66 / 7.35 s).
Enabled stages: `env`, `graphs`, `train` (2-epoch G2 timing), `matrix`/`explain` (gated), optionally
`audit`/`graph_lift`.
The output collector now sends the **new-run** results/manifests/timing summary (not stale
`repro/results` copies), and the PowerShell stages are hardened against native stderr warnings.

This folder is a **self-contained kit** for running the first canonical-lane checks on the user's
RTX 5060 PC (8 GB, Blackwell / sm_120) and bringing the results back.

---

## What is in this folder

```
remote_rtx5060/
  README.md                  <- this file
  SETUP.md                   <- prerequisites, expected hardware/software, troubleshooting
  scripts/
    01_check_prereqs.ps1     <- check Python/GPU/disk before installing
    02_setup_env.ps1         <- create .venv, install torch cu128 + PyG + deps, env check
    03_get_code.ps1          <- extract the code snapshot AND seed the frozen C1 artifacts
    04_get_data.ps1          <- copy/verify the NetFlow-v3 data (or download with a Kaggle token)
    05_run_stage.ps1         <- run a stage: env | graphs | train | audit | graph_lift
    06_make_output.ps1       <- collect everything into output/ and zip it for transfer back
  frozen/                    <- frozen protocol artifacts (C1 masks, preprocessor state, manifest)
  code/
    xdrift_repo_1810b07.zip  <- tracked-files snapshot of the repository (generated on the main PC)
    README.md                <- how the snapshot was produced / how to regenerate
  configs/
    run_stages.example.yaml  <- stage definitions used by 05_run_stage.ps1
  output/                    <- ALL results/logs go here; copy this folder back to the main PC
    README.md
    PASTE_HERE.txt
  secrets/                   <- nothing secret is committed; see secrets/README.md
```

## Data layout on the RTX 5060 PC (copy the extracted folders, no zips needed)

If you already have the extracted `data/raw` on the main PC, copy the four dataset folders into
the kit so the layout is exactly:

```
remote_rtx5060/
  data/
    raw/
      NF-UNSW-NB15-v3/NF-UNSW-NB15-v3.csv          (~577 MB)
      NF-ToN-IoT-v3/NF-ToN-IoT-v3.csv              (~5.3 GB)
      NF-BoT-IoT-v3/NF-BoT-IoT-v3.csv              (~3.8 GB)
      NF-CSE-CIC-IDS2018-v3/NF-CICIDS2018-v3.csv   (~4.2 GB, note the file name!)
```

- Copy the **folders**, not the `.zip` archives; zips are no longer needed for running.
- `NF-CSE-CIC-IDS2018-v3` contains `NF-CICIDS2018-v3.csv` (verified file name); the verification
  script accepts any single large CSV inside each folder.
- `Elliptic/` is optional (fallback lane only) and is not needed for the primary study.
- Then run `04_get_data.ps1 -Mode verify` (no Kaggle token needed); it prints `data_layout.json`.

## Quick start (copy this whole folder to the RTX 5060 PC first)

The kit ships **frozen protocol artifacts** in `frozen/` (C1 split masks, preprocessor state,
split manifest). `03_get_code.ps1` seeds them into `data/splits/**` and `results/data_audit/**`
and records their SHA-256s in `output/env/frozen_artifacts.txt`; this is why `graphs`/`train` need
no audit re-run on the PC. Do not edit or regenerate them.

Open **PowerShell** (Windows PowerShell 5.1 is fine) in this folder and run, in order:

```powershell
# 1) check machine: Python version, NVIDIA driver/GPU, free disk
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\01_check_prereqs.ps1

# 2) create the Python env (large download: torch cu128 ~5 GB; be patient)
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\02_setup_env.ps1

# 3) unpack the code snapshot AND seed the frozen C1 split masks / preprocessor state
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\03_get_code.ps1

# 4) get the data (choose ONE)
#    a) if you copied data/raw from the main PC into .\data\raw:
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\04_get_data.ps1 -Mode verify
#    b) if you want to download it on this PC (needs the Kaggle token):
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\04_get_data.ps1 -Mode download

# 5) run the G0/G2 stages
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\05_run_stage.ps1 -Stage env
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\05_run_stage.ps1 -Stage graphs
# G2 TIMING runs: 2 epochs only (the full matrix comes later from the main workstation, see below)
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\05_run_stage.ps1 -Stage train -Model egraphsage -Seed 42 -Period 1 -Epochs 2
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\05_run_stage.ps1 -Stage train -Model gcn -Seed 42 -Period 1 -Epochs 2

# (optional on this kit: audit / graph_lift against the frozen C1 state)
# powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\05_run_stage.ps1 -Stage audit
# powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\05_run_stage.ps1 -Stage graph_lift

# 6) collect outputs and zip them
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\06_make_output.ps1
```

**G2 timing expectation:** the full-size epoch time is the number that decides the matrix budget.
On the old GTX 1050 the sampled canonical trainer takes ~148 s/epoch; the RTX 5060 should be
much faster (best case ~15–20 s/epoch if GPU-bound, worst case ~100–140 s/epoch if per-step
Python/sampler overhead dominates). Report the measured `timing.epoch_seconds_mean` and
`max_memory_allocated_bytes` from each training run; that is the G2 measurement. If an epoch takes
the slow-bound time, stop after the first model — that itself triggers the sampler optimization.

**Full matrix / explanations (GATED — do not run until the Lead says go):**
the drivers are included so the same kit can launch them after G2 + audit sign-off:

```powershell
# full strategy matrix (resume-friendly; --dry-run first)
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\05_run_stage.ps1 -Stage matrix -Models egraphsage,gcn -Strategies ft,replay,ewc -Seeds 42,43,44,45,46 -DryRun
# S6 anchors + S7 IG/GNNExplainer batch (resume/cache-aware; staged)
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\05_run_stage.ps1 -Stage explain -Models egraphsage,gcn -Strategies ft,replay,ewc -Seeds 42,43,44,45,46 -DryRun
```

Then copy **`output/`** back to the main PC (or send the `output_*.zip` that step 6 prints).
The `output/` contract is described in `output/README.md`; you can also paste console text into
`output/PASTE_HERE.txt`.

## Rules

- Do not edit files inside `repro/` by hand; it is a snapshot of a git commit and its SHA is
  recorded in `output/env/code_snapshot.txt`.
- Do not edit or regenerate anything in `frozen/` (immutable protocol) — the seeded copies must
  match the SHA-256 table in `frozen/README.md`; if they differ, stop and send the output back.
- Do not create results outside `output/`; the collection script looks there.
- If a stage fails, keep the log in `output/logs/` and send it back; do not retry blindly.
- The Kaggle token is **not** needed on this PC for the enabled stages (parallel Kaggle runs are
  launched from the main workstation).
