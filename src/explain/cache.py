"""S7 explanation cache keyed by (checkpoint_sha, anchor_id, explainer, seed, k, config_sha).

Explanations are expensive and stochastic; the cache makes every generated
explanation reproducible and avoids recomputation. Arrays are stored in ``.npz``
and scalar/list metadata in ``.json`` under the configured results root
(gitignored), with a CSV index for fast lookup and for the analysis stage.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

INDEX_COLUMNS = [
    "key",
    "checkpoint_sha",
    "anchor_id",
    "explainer",
    "seed",
    "k",
    "config_sha",
    "created_at_utc",
    "npz_path",
    "json_path",
    # docs/05 §6 requires per-entry fidelity + runtime in the S7 evidence index
    # (T18/D-048: these were a WARN; they are now part of the writer schema).
    "fidelity_sufficiency",
    "fidelity_comprehensiveness",
    "fidelity_over_random",
    "runtime_s",
]

#: Keys the writer must provide for every cache entry (D-048/§6).
REQUIRED_INDEX_FIELDS = ("fidelity_sufficiency", "fidelity_comprehensiveness", "runtime_s")


def cache_key(
    *,
    checkpoint_sha: str,
    anchor_id: Any,
    explainer: str,
    seed: int,
    k: int,
    config_sha: str,
) -> str:
    payload = "|".join(
        [
            str(checkpoint_sha),
            str(anchor_id),
            str(explainer),
            str(int(seed)),
            str(int(k)),
            str(config_sha),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ExplanationCache:
    """Filesystem explanation cache with a CSV index."""

    def __init__(self, root: str | Path, *, index_csv: str | Path | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = Path(index_csv) if index_csv is not None else self.root / "index.csv"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.index_path.is_file():
            with self.index_path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(INDEX_COLUMNS)

    def paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.npz", self.root / f"{key}.json"

    def has(self, key: str) -> bool:
        npz, meta = self.paths(key)
        return npz.is_file() and meta.is_file()

    def put(
        self,
        key: str,
        *,
        arrays: Mapping[str, np.ndarray],
        metadata: Mapping[str, Any],
        index_row: Mapping[str, Any],
    ) -> Path:
        missing = [field for field in REQUIRED_INDEX_FIELDS if field not in index_row]
        if missing:
            raise ValueError(
                "S7 cache index rows must carry per-entry fidelity/runtime fields "
                f"(docs/05 §6, D-048); missing {missing}. Provide fidelity_sufficiency, "
                "fidelity_comprehensiveness and runtime_s."
            )
        npz_path, json_path = self.paths(key)
        np.savez_compressed(npz_path, **{name: np.asarray(value) for name, value in arrays.items()})
        payload = dict(metadata)
        payload["key"] = key
        payload["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload["fidelity_sufficiency"] = index_row.get("fidelity_sufficiency")
        payload["fidelity_comprehensiveness"] = index_row.get("fidelity_comprehensiveness")
        payload["fidelity_over_random"] = index_row.get("fidelity_over_random")
        payload["runtime_s"] = index_row.get("runtime_s")
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        with self.index_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    key,
                    index_row.get("checkpoint_sha", ""),
                    index_row.get("anchor_id", ""),
                    index_row.get("explainer", ""),
                    index_row.get("seed", ""),
                    index_row.get("k", ""),
                    index_row.get("config_sha", ""),
                    payload["created_at_utc"],
                    str(npz_path),
                    str(json_path),
                    index_row.get("fidelity_sufficiency", ""),
                    index_row.get("fidelity_comprehensiveness", ""),
                    index_row.get("fidelity_over_random", ""),
                    index_row.get("runtime_s", ""),
                ]
            )
        return json_path

    def get(self, key: str) -> dict[str, Any] | None:
        npz_path, json_path = self.paths(key)
        if not npz_path.is_file() or not json_path.is_file():
            return None
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as archive:
            metadata["arrays"] = {name: archive[name] for name in archive.files}
        return metadata

    def index_rows(self) -> list[dict[str, str]]:
        with self.index_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


def cache_from_config(config: Mapping[str, Any], *, results_root: str | Path) -> ExplanationCache:
    """Build the cache from the S7 config section, rooted under the results dir."""
    cache_cfg = dict(config.get("cache") or {})
    subdir = str(cache_cfg.get("subdir", "explanations"))
    index_name = str(cache_cfg.get("index_csv", "index.csv"))
    return ExplanationCache(Path(results_root) / subdir, index_csv=Path(results_root) / subdir / index_name)


__all__ = ["ExplanationCache", "INDEX_COLUMNS", "cache_from_config", "cache_key"]
