# Main-pipeline diagnostic tools

This folder contains **diagnostic helpers**, not numbered pipeline stages. They
are allowed to read the raw inputs and write audit reports, but they must not
modify raw data or create model artifacts.

## `profile_raw_dataset.py`

Standard-library streaming profiler for the raw NF-UNSW-NB15-v3 CSV. It exists
because the main cleaning script must be written from evidence, and because the
system Python used for planning may not have pandas installed.

What it records:

- schema, row count, file size and SHA-256;
- full-row duplicate count and connection-key duplicate count;
- Label/Attack counts and label consistency;
- timestamp parseability, start/end ordering and basic time range;
- per-column empty/NA count, non-numeric count, negative count, zero count,
  min/max/mean/std, number of exact unique values (capped), and examples.

Run from the repository root:

```bash
python src/main/tools/profile_raw_dataset.py \
  --csv datasets/raw/NF-UNSW-NB15-v3/NF-UNSW-NB15-v3.csv \
  --out results/text/data_audit/raw_dataset_profile.json
```

Defaults:

| Option | Default |
|---|---|
| `--csv` | `datasets/raw/NF-UNSW-NB15-v3/NF-UNSW-NB15-v3.csv` |
| `--out` | `results/text/data_audit/raw_dataset_profile.json` |
| `--chunk-size` | 200,000 rows |
| `--unique-cap` | 20,000 exact unique values per column |
| `--examples` | 3 examples per column |

It also writes a human-readable `.txt` dump next to the JSON and prints a
compact summary table. It is read-only: it never edits the CSV and never writes
under `datasets/`.

The output feeds `analysis/raw_dataset_profile.md`, which records the per-feature
preprocessing recommendations. Do not put this profiler in the numbered main
chain; it is a one-off decision aid, not a production stage.
