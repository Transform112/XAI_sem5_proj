# `src/verification/scripts/` — verifier command-line scripts

Independent audit scripts. They read signed artifacts and recompute/compare
key quantities; they do not produce signed tables.

## How to execute

Run from the repository root with the project environment, passing explicit
artifact/run paths where supported. See `../README.md` for the full list,
examples, and the historical path caveat:

```powershell
.\.venv\Scripts\python.exe srcerification\scripts_artifact_manifest.py --help
```
