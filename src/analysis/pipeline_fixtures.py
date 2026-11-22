"""Synthetic S6/S7 pipeline fixtures for validator tests and the demo (T20).

Why this module exists
----------------------
The real S6 anchor sets (``results/text/anchors/period_*.pt``) and S7 explanation cache
(``results/text/explanations/index.csv`` + ``.npz``/``.json`` entries) are produced by
the engineer's pipeline (``src/scripts/build_anchors.py``, ``src/scripts/run_explain.py``).
The validators must be testable *before* those expensive runs exist, so this
module synthesises small, schema-faithful artifacts under an
:class:`analysis.artifacts.ArtifactLayout` root (the demo writes them under
``src/analysis/examples/``; tests use ``tmp_path``).

The schemas mirror the engineer implementations (verified by reading
``src/anchors/build.py`` and ``src/explain/cache.py`` on 2026-11-01):

* anchor set dict: ``period, anchors[], n_eligible, n_selected, strata,
  rare_classes, rare_stratum, class_counts, config`` where each anchor carries
  ``anchor_id, period_local_edge_index, period, label, predicted_class,
  confidence, stratum, host_src_global, host_dst_global, edge_ts, sub_edge_index,
  sub_edge_attr, sub_edge_ids, sub_node_global_ids, target_local_index,
  target_edge, n_sub_nodes, n_sub_edges``;
* retention table columns ``anchor_period,later_checkpoint,retained,total,
  retained_fraction`` (engineer path ``results/text/anchors/retained_fraction.csv``);
* explanation index columns of ``src/explain/cache.py`` (``key, checkpoint_sha,
  anchor_id, explainer, seed, k, config_sha, created_at_utc, npz_path,
  json_path``) and per-entry metadata/arrays matching the S7 writers.

Everything here is explicitly synthetic and never written into the real
``results/`` tree by the toolkit.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.analysis import artifacts
from src.analysis import synthetic

#: Explanation-index columns; mirrors ``src/explain/cache.py::INDEX_COLUMNS``.
EXPLANATION_INDEX_COLUMNS: tuple[str, ...] = (
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
    "fidelity_sufficiency",
    "fidelity_comprehensiveness",
    "fidelity_over_random",
    "runtime_s",
)

#: Retained-fraction columns; mirrors ``src/scripts/build_anchors.py``.
RETAINED_FRACTION_COLUMNS: tuple[str, ...] = (
    "anchor_period",
    "later_checkpoint",
    "retained",
    "total",
    "retained_fraction",
)

ANCHOR_KEYS: tuple[str, ...] = (
    "anchor_id",
    "period_local_edge_index",
    "period",
    "label",
    "predicted_class",
    "confidence",
    "stratum",
    "host_src_global",
    "host_dst_global",
    "edge_ts",
    "sub_edge_index",
    "sub_edge_attr",
    "sub_edge_ids",
    "sub_node_global_ids",
    "target_local_index",
    "target_edge",
    "n_sub_nodes",
    "n_sub_edges",
)

N_PERIODS = synthetic.N_PERIODS


class PipelineFixtureError(RuntimeError):
    """Raised when a synthetic pipeline fixture cannot be written."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_synthetic_anchors(layout: artifacts.ArtifactLayout, *, seed: int = synthetic.SEED) -> dict[str, Path]:
    """Write a schema-faithful synthetic S6 anchor set (``results/text/anchors/``)."""
    import torch

    anchors_root = layout.results_root / "anchors"
    anchors_root.mkdir(parents=True, exist_ok=True)
    class_names = list(synthetic.CLASSES)
    rng = np.random.default_rng([seed, 601])
    written: dict[str, Path] = {}
    anchor_ids: dict[int, list[int]] = {}
    for period in range(1, N_PERIODS + 1):
        anchors: list[dict[str, Any]] = []
        # 3 anchors per period across different classes / rarely classes
        label_choices = [(period - 1) % len(class_names), (period + 2) % len(class_names), (period + 6) % len(class_names)]
        for slot, label in enumerate(label_choices):
            anchor_id = 100_000 * period + slot
            local_index = slot * 17 + 3
            n_edges = 4
            sub_edge_ids = torch.arange(local_index, local_index + n_edges, dtype=torch.long)
            anchors.append(
                {
                    "anchor_id": int(anchor_id),
                    "period_local_edge_index": int(local_index),
                    "period": int(period),
                    "label": int(label),
                    "predicted_class": int(label),
                    "confidence": float(np.clip(rng.normal(0.95, 0.02), 0.9, 0.999)),
                    "stratum": str(label),
                    "host_src_global": int(10_000 + local_index),
                    "host_dst_global": int(20_000 + local_index),
                    "edge_ts": int(1_700_000_000 + period * 1000 + slot),
                    "sub_edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                    "sub_edge_attr": torch.as_tensor(rng.normal(0, 1, size=(n_edges, synthetic.FEATURE_DIM)), dtype=torch.float32),
                    "sub_edge_ids": sub_edge_ids,
                    "sub_node_global_ids": torch.tensor([10_000 + local_index, 20_000 + local_index, 10_000 + local_index + 1], dtype=torch.long),
                    "target_local_index": torch.tensor([0], dtype=torch.long),
                    "target_edge": sub_edge_ids[:1],
                    "n_sub_nodes": 3,
                    "n_sub_edges": n_edges,
                }
            )
        anchor_ids[period] = [anchor["anchor_id"] for anchor in anchors]
        anchor_set = {
            "period": period,
            "anchors": anchors,
            "n_eligible": 3 * 500,
            "n_selected": len(anchors),
            "strata": {str(anchor["label"]): 1 for anchor in anchors},
            "rare_classes": [int(label) for label in label_choices if label in (7, 8, 9)],
            "rare_stratum": "rare",
            "class_counts": {str(label): 3 * 100 for label in range(len(class_names))},
            "config": {
                "confidence_min": 0.9,
                "max_per_class": 100,
                "max_per_period": 1000,
                "rare_min_eligible": 10,
                "rare_stratum": "rare",
                "sampling_seed": seed,
                "num_hops": 2,
                "fanout": 10,
            },
        }
        path = anchors_root / f"period_{period}.pt"
        torch.save(anchor_set, path)
        written[f"period_{period}"] = path

    retained_rows: list[dict[str, Any]] = []
    for period in range(1, N_PERIODS + 1):
        for later in range(period + 1, N_PERIODS + 1):
            total = len(anchor_ids[period])
            retained = max(0, total - (later - period))
            retained_rows.append(
                {
                    "anchor_period": period,
                    "later_checkpoint": later,
                    "retained": retained,
                    "total": total,
                    "retained_fraction": retained / total,
                }
            )
    retained_csv = anchors_root / "retained_fraction.csv"
    with retained_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RETAINED_FRACTION_COLUMNS))
        writer.writeheader()
        writer.writerows(retained_rows)
    canonical_retention = layout.table("anchor_retention")  # D-048(b) canonical committed table
    canonical_retention.parent.mkdir(parents=True, exist_ok=True)
    per_checkpoint: dict[int, list[float]] = {}
    for row in retained_rows:
        per_checkpoint.setdefault(int(row["later_checkpoint"]), []).append(float(row["retained_fraction"]))
    canonical_rows: list[dict[str, Any]] = []
    previous = 1.0
    for checkpoint in sorted(per_checkpoint):
        value = float(np.mean(per_checkpoint[checkpoint]))
        previous = min(previous, value)
        canonical_rows.append({"strategy": "ft", "model": "egraphsage", "class": "all", "checkpoint_t": checkpoint, "retained_fraction": previous})
    with canonical_retention.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["strategy", "model", "class", "checkpoint_t", "retained_fraction"])
        writer.writeheader()
        writer.writerows(canonical_rows)
    retained_json = anchors_root / "retained_fraction.json"
    retained_json.write_text(json.dumps({"rows": retained_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "NF-UNSW-NB15-v3",
        "strategy": "ft",
        "model": "egraphsage",
        "seed": seed,
        "synthetic": True,
        "periods": {
            str(period): {
                "path": str((anchors_root / f"period_{period}.pt")),
                "n_selected": len(anchor_ids[period]),
                "n_eligible": 1500,
            }
            for period in range(1, N_PERIODS + 1)
        },
        "retained_fraction": {"path": str(retained_json), "csv": str(retained_csv), "rows": len(retained_rows)},
    }
    manifest_path = anchors_root / "anchors_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.update({"retained_fraction_csv": retained_csv, "retained_fraction_json": retained_json, "anchors_manifest": manifest_path, "anchor_retention_canonical": canonical_retention})
    return written


def write_synthetic_explanation_cache(
    layout: artifacts.ArtifactLayout,
    *,
    seed: int = synthetic.SEED,
    max_checkpoints: int = 2,
) -> dict[str, Path]:
    """Write a schema-faithful synthetic S7 explanation cache (``results/text/explanations/``)."""
    anchors_root = layout.results_root / "anchors"
    cache_root = layout.results_root / "explanations"
    cache_root.mkdir(parents=True, exist_ok=True)
    import torch

    anchor_sets: dict[int, Mapping[str, Any]] = {}
    for period in range(1, max_checkpoints + 1):
        path = anchors_root / f"period_{period}.pt"
        if not path.is_file():
            raise PipelineFixtureError(f"require synthetic anchor set first: {path}")
        anchor_sets[period] = torch.load(path, weights_only=False)
    index_path = cache_root / "index.csv"
    rng = np.random.default_rng([seed, 701])
    rows: list[dict[str, Any]] = []
    written: dict[str, Path] = {}
    config_sha = _sha256_text("synthetic-explain-config")
    for checkpoint in range(1, max_checkpoints + 1):
        checkpoint_sha = _sha256_text(f"synthetic-M{checkpoint}")
        for anchor_period in range(1, checkpoint + 1):
            for anchor in anchor_sets[anchor_period].get("anchors", []):
                for explainer, k_values, array_name in (
                    ("ig", (5, 10), "attributions"),
                    ("gnn_explainer", (0,), "edge_mask"),
                ):
                    for k in k_values:
                        value_seed = int(rng.integers(1_000_000_000))
                        key = _sha256_text(f"{checkpoint_sha}|{anchor['anchor_id']}|{explainer}|{seed}|{k}|{config_sha}")
                        npz_path = cache_root / f"{key}.npz"
                        json_path = cache_root / f"{key}.json"
                        if explainer == "ig":
                            array = rng.normal(0.0, 0.1, size=(synthetic.FEATURE_DIM,)).astype(np.float32)
                            fidelity = {
                                "sufficiency": {str(value): float(np.clip(rng.normal(0.5, 0.05), 0, 1)) for value in (5, 10)},
                                "comprehensiveness": {str(value): float(np.clip(rng.normal(0.3, 0.05), 0, 1)) for value in (5, 10)},
                            }
                        else:
                            array = np.abs(rng.normal(0.0, 0.2, size=(int(anchor["n_sub_edges"]),))).astype(np.float32)
                            fidelity = {"sufficiency": {"5": float(np.clip(rng.normal(0.4, 0.05), 0, 1))}}
                        np.savez_compressed(npz_path, **{array_name: array})
                        runtime_s = float(abs(rng.normal(1.0, 0.1)))
                        fidelity_sufficiency = float(np.clip(rng.normal(0.5, 0.05), 0, 1))
                        fidelity_comprehensiveness = float(np.clip(rng.normal(0.3, 0.05), 0, 1))
                        fidelity_over_random = float(abs(rng.normal(2.0, 0.2)))
                        metadata = {
                            "explainer": explainer,
                            "anchor_id": int(anchor["anchor_id"]),
                            "seed": seed,
                            "k": k,
                            "config_sha": config_sha,
                            "checkpoint_period": checkpoint,
                            "anchor_period": anchor_period,
                            "class": str(anchor["label"]),
                            "label": int(anchor["label"]),
                            "strategy": "ft",
                            "model": "egraphsage",
                            "dataset": "NF-UNSW-NB15-v3",
                            "synthetic": True,
                            "fidelity": fidelity,
                            "runtime_s": runtime_s,
                            "value_seed": value_seed,
                        }
                        json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                        rows.append(
                            {
                                "key": key,
                                "checkpoint_sha": checkpoint_sha,
                                "anchor_id": int(anchor["anchor_id"]),
                                "explainer": explainer,
                                "seed": seed,
                                "k": k,
                                "config_sha": config_sha,
                                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "npz_path": npz_path.name,
                                "json_path": json_path.name,
                                "fidelity_sufficiency": fidelity_sufficiency,
                                "fidelity_comprehensiveness": fidelity_comprehensiveness,
                                "fidelity_over_random": fidelity_over_random,
                                "runtime_s": runtime_s,
                            }
                        )
                        written[f"{explainer}_{anchor_period}_{checkpoint}_{anchor['anchor_id']}_{k}"] = json_path
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EXPLANATION_INDEX_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    written["index_csv"] = index_path
    canonical_index = layout.table("explain_index")  # D-048 canonical committed evidence index
    canonical_index.parent.mkdir(parents=True, exist_ok=True)
    with canonical_index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EXPLANATION_INDEX_COLUMNS))
        writer.writeheader()
        for row in rows:
            # canonical evidence index lives in results/text/tables/: store absolute paths
            # so a single row resolves from either copy (D-048: same cache-key rows).
            enriched = dict(row)
            enriched["npz_path"] = str((cache_root / row["npz_path"]).resolve())
            enriched["json_path"] = str((cache_root / row["json_path"]).resolve())
            writer.writerow(enriched)
    written["explain_index_canonical"] = canonical_index
    return written


def write_synthetic_noise_caches(layout: artifacts.ArtifactLayout, *, seed: int = synthetic.SEED, max_anchor_period: int = 1) -> dict[str, Any]:
    """Write schema-faithful synthetic S8 noise caches (D-048(d)).

    Emits explainer-repeat (R=10) and model-null (R=5) explanation caches with
    evidence-carrying indexes, the combined-null CSV and the noise-floor manifest
    expected by `src.analysis.validators.validate_noise_cache_artifacts`.
    """
    import torch

    anchors_root = layout.results_root / "anchors"
    noise_root = layout.results_root / "explanations_noise"
    noise_root.mkdir(parents=True, exist_ok=True)
    anchor_sets = {}
    for period in range(1, max_anchor_period + 1):
        path = anchors_root / f"period_{period}.pt"
        if not path.is_file():
            raise PipelineFixtureError(f"require synthetic anchor set first: {path}")
        anchor_sets[period] = torch.load(path, weights_only=False)
    config_sha = _sha256_text("synthetic-drift-config")
    rng = np.random.default_rng([seed, 801])
    written = {}
    channel_indexes = {}
    per_channel_rows = {}
    for channel, repeats in (("model_null", 5), ("explainer_repeat", 10)):
        channel_root = noise_root / channel
        channel_root.mkdir(parents=True, exist_ok=True)
        index_rows = []
        for period, anchor_set in anchor_sets.items():
            for anchor in anchor_set.get("anchors", []):
                for explainer in ("ig", "gnn_explainer"):
                    array_name = "attributions" if explainer == "ig" else "edge_mask"
                    if explainer == "ig":
                        base_array = rng.normal(0.0, 0.1, size=(synthetic.FEATURE_DIM,)).astype(np.float32)
                    else:
                        base_array = np.abs(rng.normal(0.0, 0.2, size=(int(anchor["n_sub_edges"]),))).astype(np.float32)
                    for repeat in range(repeats):
                        noise_seed = int(seed) + 100 + repeat if channel == "model_null" else int(seed) * 1000 + repeat
                        if explainer == "ig":
                            array = base_array.copy()  # deterministic IG: repeat canary = 0
                        else:
                            array = np.clip(base_array + rng.normal(0, 0.005, size=base_array.shape), 0, None).astype(np.float32)
                        key = _sha256_text(f"{channel}|{period}|{anchor['anchor_id']}|{explainer}|{repeat}|{config_sha}")
                        npz_path = channel_root / f"{key}.npz"
                        json_path = channel_root / f"{key}.json"
                        np.savez_compressed(npz_path, **{array_name: array})
                        runtime_s = float(abs(rng.normal(0.5, 0.05)))
                        fidelity_sufficiency = float(np.clip(rng.normal(0.5, 0.05), 0, 1))
                        fidelity_comprehensiveness = float(np.clip(rng.normal(0.3, 0.05), 0, 1))
                        fidelity_over_random = float(abs(rng.normal(2.0, 0.2)))
                        metadata = {
                            "explainer": explainer,
                            "anchor_id": int(anchor["anchor_id"]),
                            "seed": noise_seed,
                            "checkpoint_period": int(period),
                            "anchor_period": int(period),
                            "class": str(anchor["label"]),
                            "label": int(anchor["label"]),
                            "strategy": "ft",
                            "model": "egraphsage",
                            "noise_channel": channel,
                            "config_sha": config_sha,
                            "fidelity": {"sufficiency_ratio": fidelity_sufficiency, "comprehensiveness": fidelity_comprehensiveness},
                            "runtime_s": runtime_s,
                            "synthetic": True,
                        }
                        json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                        index_rows.append({
                            "key": key,
                            "checkpoint_sha": f"synthetic-null-{channel}-{period}",
                            "anchor_id": int(anchor["anchor_id"]),
                            "explainer": explainer,
                            "seed": noise_seed,
                            "k": 5 if explainer == "ig" else 0,
                            "config_sha": config_sha,
                            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "npz_path": npz_path.name,
                            "json_path": json_path.name,
                            "fidelity_sufficiency": fidelity_sufficiency,
                            "fidelity_comprehensiveness": fidelity_comprehensiveness,
                            "fidelity_over_random": fidelity_over_random,
                            "runtime_s": runtime_s,
                        })
        index_path = channel_root / "index.csv"
        with index_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(EXPLANATION_INDEX_COLUMNS))
            writer.writeheader()
            writer.writerows(index_rows)
        channel_indexes[channel] = index_path
        per_channel_rows[channel] = index_rows
        written[channel + "_index"] = index_path
    combined = noise_root / "combined_null.csv"
    combined_rows = []
    for channel, rows in per_channel_rows.items():
        for row in rows:
            combined_rows.append({**row, "channel": channel})
    with combined.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EXPLANATION_INDEX_COLUMNS) + ["channel"])
        writer.writeheader()
        writer.writerows(combined_rows)
    written["combined_null"] = combined
    audit_root = layout.results_root / "data_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "NF-UNSW-NB15-v3",
        "synthetic": True,
        "channels": {
            channel: {
                "entries": len(rows),
                "index_csv": str(channel_indexes[channel]),
                "repeats": 5 if channel == "model_null" else 10,
                "ig_repeat_canary_drift_max": 0.0 if channel == "explainer_repeat" else None,
            }
            for channel, rows in per_channel_rows.items()
        },
        "combined_null": {"csv": str(combined), "rows": len(combined_rows)},
    }
    manifest_path = audit_root / "noise_floor_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written["noise_floor_manifest"] = manifest_path
    return written


def write_synthetic_s6_s7(layout: artifacts.ArtifactLayout, *, seed: int = synthetic.SEED) -> dict[str, dict[str, Path]]:
    """Write both synthetic S6 and S7 artifacts (convenience for demo/tests)."""
    return {
        "s6": write_synthetic_anchors(layout, seed=seed),
        "s7": write_synthetic_explanation_cache(layout, seed=seed),
        "noise": write_synthetic_noise_caches(layout, seed=seed),
    }


__all__ = [
    "ANCHOR_KEYS",
    "EXPLANATION_INDEX_COLUMNS",
    "N_PERIODS",
    "PipelineFixtureError",
    "RETAINED_FRACTION_COLUMNS",
    "write_synthetic_anchors",
    "write_synthetic_explanation_cache",
    "write_synthetic_noise_caches",
    "write_synthetic_s6_s7",
]
