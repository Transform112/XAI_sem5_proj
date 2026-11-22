# Frozen protocol artifacts (Protocol v1, D-042) — do not edit

These files are the **immutable** data-stage outputs needed to build graphs and train on the
RTX 5060 PC without re-running the (slow) audit stage. They were produced on the main workstation
from the no-drop C1 freeze and are copied into the kit by
`src/kaggle_lane/remote_rtx5060/scripts/03_get_code.ps1`.

| Path (after seeding) | SHA-256 | What it is |
| --- | --- | --- |
| `datasets/splits/NF-UNSW-NB15-v3/session_aware_1_4/period_1_masks.npz` | `dec795d2…` | frozen grouped split masks, period 1 (train/val/test) |
| `…/period_2_masks.npz` | `64bbafae…` | period 2 |
| `…/period_3_masks.npz` | `f3500d34…` | period 3 |
| `…/period_4_masks.npz` | `c2a5acec…` | period 4 |
| `…/period_5_masks.npz` | `d1daf60e…` | period 5 |
| `…/host_pair_disjoint/period_{1..5}_masks.npz` | see `frozen_artifacts.txt` | robustness-variant masks (not used for headline runs) |
| `results/text/data_audit/preprocessor_state.json` | `f83dd76b…` | preprocessor fitted on period-1 TRAIN ONLY (docs/09) |
| `results/text/data_audit/split_manifest.json` | `eee0f46e…` | frozen split manifest (grouped hash `010a80d9…`) |
| `results/text/data_audit/feature_manifest.csv` | `1ba3fc81…` | kept/dropped feature decisions |

Rules:
- **Do not modify or regenerate** these on the PC; they are the frozen protocol (docs/DECISIONS.md D-042).
- `src/kaggle_lane/remote_rtx5060/scripts/03_get_code.ps1` records the seeded SHA-256s in `output/env/frozen_artifacts.txt`.
- If a hash does not match this table, stop and send the output back — do not proceed.
