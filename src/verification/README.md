# `src/verification/` — independent verifier scripts

The verifier scripts implement the **independent checks** used by the verifier
role. They read signed artifacts and recompute or cross-check key quantities;
they do not write production tables.

## Contents

```text
src/verification/scripts/
├─ v_artifact_manifest.py     # package/file manifest + SHA checks
├─ v_artifact_state.py        # artifact presence/state checks
├─ v_claims.py                # claim ↔ evidence checks
├─ v_compare.py               # table-to-table comparison
├─ v_governance.py            # protocol/decision governance checks
├─ v_identity.py              # raw dataset identity
├─ v_leak.py                  # leakage checks
├─ v_period_counts.py         # period/row-count checks
├─ v_prereg_audit.py          # pre-registration audit
├─ v_splits.py                # frozen split checks
├─ v_stats.py                 # statistics recomputation
├─ v_structure.py             # graph structure checks
├─ v_s1_audit.py … v_s4_audit.py
├─ v_t21_* … v_t49_*          # per-task independent audits
└─ v_matrix_full_audit.py     # 30-group matrix audit
```

Evidence (JSON/MD) lives under `results/text/verification/data_audit/`.
The final sign-off is `docs/VERIFICATION_REPORT_FINAL.md`; its machine-readable
counterpart is `results/text/verification/data_audit/final_paper_audit.json`.

## Execution

Run from the repository root; the scripts add the root to `sys.path`.

```powershell
# e.g. final artifact/package checks
.\.venv\Scripts\python.exe src\verification\scripts\v_artifact_manifest.py --help
.\.venv\Scripts\python.exe src\verification\scripts\v_claims.py --help

# e.g. S8 final audit
.\.venv\Scripts\python.exe src\verification\scripts\v_t49_s8_final_audit.py --help
```

Many scripts accept explicit paths so they can audit a downloaded Kaggle run.
The original downloaded run directories (`incoming_rtx5060/...`) were removed in
the cleanup; point `--run-dir` at a freshly downloaded run or an archived copy:

```powershell
.\.venv\Scripts\python.exe src\verification\scripts\v_t42_noise_v7_audit.py `
  --run-dir <path-to-downloaded-run>
```

## Signed check status (final)

| Gate | Result |
|---|---|
| CL matrix audit | `480/0/0` (pass/warn/fail) |
| S6/S7 strict validators | `16/28/48` |
| S8 strict validator | `75/0/0` |
| S9 strict validator | `103/0/0` |
| Signed package | 194/194 files SHA-matched at sign-off |
| Final verifier sign-off | PASS (paper content), see `docs/VERIFICATION_REPORT_FINAL.md` |

## Rules

- Never silently repair a failed check; record the failure and the decision in
  `docs/DECISIONS.md`.
- Do not use the verifier to generate signed numbers; it only audits them.
- Historical old audit trees (`verification/`, `analysis/`) were removed in the
  restructure; their final evidence is preserved under
  `results/text/verification/`.
