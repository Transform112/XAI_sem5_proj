"""Shared relocation of recorded artifact paths (T37/T49).

Kaggle-run indexes record absolute container paths such as
``.../stage/results/text/explanations/<sha>.npz``.  When the artifacts are
downloaded and assembled on another host, those paths do not exist; the file is
found by taking the suffix after ``results/`` and joining it to the local results
root while leaving the canonical index untouched.

This module is intentionally dependency-free (stdlib only) so both the analysis
validators and the S8 wiring can use exactly the same rule.
"""

from __future__ import annotations

from pathlib import Path


def infer_results_root(index_path: str | Path) -> Path:
    """Infer ``.../results`` from an index CSV path (``results/text/tables`` or ``results/text/explanations``)."""
    path = Path(index_path)
    for parent in (path.parent, *path.parents):
        if parent.name == "results":
            return parent
    return path.parent


def resolve_artifact_path(
    value: str | Path,
    index_path: str | Path,
    *,
    results_root: str | Path | None = None,
) -> Path:
    """Resolve a path recorded in an artifact index against the local results root.

    Rules (same as ``src.analysis.validators._resolve_cache_path``, now shared):

    * an existing literal path is authoritative (local runs);
    * relative paths resolve against the index file's directory;
    * absolute paths (remote POSIX like a Kaggle working path or Windows) fall
      back to the suffix after the last ``results/`` component, joined to the
      local results root;
    * if the suffix candidate is absent, return the conventional
      ``results_root/explanations/<basename>`` fallback so callers can report the
      missing file explicitly instead of silently dropping the row.
    """
    raw = str(value).strip()
    path = Path(raw)
    if path.is_file():
        return path
    posix_absolute = raw.startswith("/") and not raw.startswith("//")
    if not path.is_absolute() and not posix_absolute:
        return Path(index_path).parent / path
    root = Path(results_root) if results_root is not None else infer_results_root(index_path)
    parts = path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "results":
            candidate = root.joinpath(*parts[index + 1 :])
            if candidate.is_file():
                return candidate
            break
    return root / "text" / "explanations" / path.name


__all__ = ["infer_results_root", "resolve_artifact_path"]
