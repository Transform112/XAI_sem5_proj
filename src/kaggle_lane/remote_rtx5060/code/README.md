# Code snapshot

`xdrift_repo_<sha>.zip` is a **git archive** of the repository at a specific commit: tracked
runtime paths only (`src/ configs/ docs/ platforms/ paper_writing/` + top-level policy files),
no data, no virtualenv, no runs, no audit evidence. It is what
`src/kaggle_lane/remote_rtx5060/scripts/03_get_code.ps1` extracts into `repro/`.

- The SHA is reported in `output/env/code_snapshot.txt` and in every run manifest.
- If you need a newer snapshot, the main PC regenerates it with:

```powershell
# on the main workstation, from the repository root:
powershell -NoProfile -ExecutionPolicy Bypass -File .\src\kaggle_lane\remote_rtx5060\make_code_snapshot.ps1
```

- Do not edit files inside `repro/` by hand — patch on the main PC and regenerate the snapshot.

**Kit build checklist (provenance):** keep `code/snapshot_info.txt` in the kit (it records the
snapshot SHA, zip sha256, included paths and dirty-tree count); when cleaning the kit folder before
zipping, do not delete it. The shipped `xdrift_repo_<sha>.zip` is a `git archive` of the runtime
paths at that commit; manifests record the code fingerprint of the extracted snapshot bytes, so if
the snapshot bytes are ever re-zipped/re-normalised (e.g. line endings) the fingerprint changes
even though the code is semantically identical — document any such re-normalisation in the output
report (verifier T21/T25/kit-preflight practice).
