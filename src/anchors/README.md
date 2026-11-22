# `src/anchors` — canonical anchor selection

Anchors are the flows whose explanations are tracked across checkpoints. They
are selected **once**, before any drift is measured, and reused for every
period pair and explainer.

| Module | Responsibility |
|---|---|
| `build.py` | `select_anchors`, retained-fraction summaries, anchor manifests. |

Selection is deterministic for a seed and uses the pre-registered criteria
(correct on the period, confident, class-balanced/stratified). The selected
anchor set is the `1,008`-row canonical set per period group documented in the
paper; every explanation row carries the anchor id and context hash.

## Canonical command

```powershell
.\.venv\Scripts\python.exe src\scripts\build_anchors.py --help
```

Typical invocation (one group):

```powershell
.\.venv\Scripts\python.exe src\scripts\build_anchors.py `
  --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage --seed 42 `
  --lane cpu --device cpu
```

Outputs:

- `results/text/anchors/<group>/anchors.csv` and `anchors_manifest.json`;
- retained-fraction tables used by RQ2.

## Invariants

- Never re-select anchors after seeing drift results; that would leak the
  outcome into the design.
- The anchor manifest is hashed and checked by the explanation runner before
  reuse.
- Anchor selection is seed-42 canonical in the paper; other seeds are
  diagnostics unless explicitly labelled.

## Tests

`src/tests/test_anchors*.py`. Run after installing the environment:
`python -m pytest src\tests -q`.
