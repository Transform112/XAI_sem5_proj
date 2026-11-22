# `src/cl` — continual-learning update rules

The continual-learning package defines **what changes when a new period
arrives**. Training data, epochs, seeds and evaluation are shared with
`src/train`; only the update rule differs, so any difference is attributable to
the strategy.

| Module | Strategy |
|---|---|
| `strategies.py` | `FineTuneStrategy`, `ReplayStrategy` (`ReplayBuffer`), `EWCStrategy`; factory `build_strategy`. |
| `ear.py` | Retained EAR-A implementation (`EARStrategy`, `build_ear_strategy`) kept for a future extension. |

## Strategies

- **Fine-tune (FT)** — continue training on the new period. The reference that
  forgets.
- **Corrected replay** — mix a fresh class-balanced buffer batch with new data;
  replay CE is averaged, not summed (D-072).
- **EWC λ=100** — quadratic penalty on weights important on the old period;
  Fisher importance from old-period train edges; λ chosen by the pre-registered
  seed-42 pilot (D-073).
- **EAR-A** — explanation-anchored replay; retained code only. No signed EAR
  result exists in this paper (RQ4 is deferred/dropped — see
  `docs/DECISIONS.md` and `analysis/reports/ear_comparison.md`).

## Canonical commands

Strategies are selected by `src/scripts/run_train.py` / `run_matrix.py`:

```powershell
.\.venv\Scripts\python.exe src\scripts\run_matrix.py `
  --dataset NF-UNSW-NB15-v3 --models egraphsage,gcn `
  --strategies ft,replay,ewc --seeds 42,43,44,45,46 --periods 5 `
  --lane cpu --device cpu
```

Reward/penalty hyperparameters live in `configs/`; do not tune them on the test
periods.

## Important status

`src/cl/ear.py` is **inactive research code**. Do not cite an EAR result, and do
not present it as a completed mitigation; the paper's RQ4 is explicitly
deferred/dropped.

## Tests

`src/tests/test_cl*.py` covers strategy construction, replay-buffer sampling and
the EWC penalty. Run after installing the environment:
`python -m pytest src\tests -q`.
