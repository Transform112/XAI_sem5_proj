"""T49 regression tests: remote relocation, metadata group filtering, R-aware noise loading."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest


def _write_entry(
    root: Path,
    *,
    name: str,
    explainer: str,
    k: int,
    anchor_id: int,
    metadata: dict,
    remote_subdir: str,
    array: np.ndarray,
) -> dict:
    entry_dir = root / "results" / remote_subdir
    entry_dir.mkdir(parents=True, exist_ok=True)
    npz_path = entry_dir / f"{name}.npz"
    json_path = entry_dir / f"{name}.json"
    key = "attributions" if explainer == "ig" else "edge_mask"
    np.savez(npz_path, **{key: np.asarray(array, dtype=float)})
    json_path.write_text(json.dumps(metadata), encoding="utf-8")
    return {
        "key": name,
        "checkpoint_sha": metadata.get("checkpoint_sha", f"sha_{name}"),
        "anchor_id": str(anchor_id),
        "explainer": explainer,
        "seed": str(metadata.get("seed", 42)),
        "k": str(k),
        "config_sha": "cfg",
        "created_at_utc": "now",
        "npz_path": f"/kaggle/working/stage/results/{remote_subdir}/{name}.npz",
        "json_path": f"/kaggle/working/stage/results/{remote_subdir}/{name}.json",
        "fidelity_sufficiency": "0.5",
        "fidelity_comprehensiveness": "0.5",
        "fidelity_over_random": "1.0",
        "runtime_s": "0.1",
    }


def _write_index(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cache_row(tmp: Path, *, name: str, anchor_id: int, group: dict, checkpoint: int = 1) -> dict:
    metadata = {
        "explainer": "ig",
        "anchor_id": anchor_id,
        "seed": group["seed"],
        "checkpoint_period": checkpoint,
        "anchor_period": 1,
        "class": "4",
        "label": 4,
        "strategy": group["strategy"],
        "model": group["model"],
    }
    return _write_entry(
        tmp,
        name=name,
        explainer="ig",
        k=5,
        anchor_id=anchor_id,
        metadata=metadata,
        remote_subdir="explanations",
        array=np.arange(5, dtype=float) + checkpoint,
    )


def test_loader_relocates_remote_paths_and_filters_group_metadata(tmp_path):
    from src.analysis_wiring.frames import load_explanations_from_cache

    rows = [
        _cache_row(tmp_path, name="ft_a", anchor_id=7, group={"strategy": "ft", "model": "egraphsage", "seed": 42}),
        _cache_row(tmp_path, name="replay_a", anchor_id=7, group={"strategy": "replay", "model": "egraphsage", "seed": 42}),
        _cache_row(tmp_path, name="ft_b", anchor_id=8, group={"strategy": "ft", "model": "egraphsage", "seed": 42}, checkpoint=2),
    ]
    index = tmp_path / "results" / "tables" / "explain_index.csv"
    _write_index(index, rows)
    # Paths in the index are /kaggle/...; files are under the local results root.
    explanations, meta = load_explanations_from_cache(
        index, explainer="ig", k=5, strategy="ft", model="egraphsage", seed=42
    )
    assert len(explanations) == 2, "same anchor/checkpoint rows from another group must be filtered"
    assert set(explanations) == {(1, 7), (2, 8)}
    assert {entry["strategy"] for entry in meta.values()} == {"ft"}
    assert meta[7]["anchor_period"] == 1


def test_loader_requires_group_metadata_when_filtering(tmp_path):
    from src.analysis_wiring.frames import AnalysisWiringError, load_explanations_from_cache

    # old cache entry without strategy/model/seed metadata
    row = _write_entry(
        tmp_path,
        name="old",
        explainer="ig",
        k=5,
        anchor_id=1,
        metadata={"anchor_id": 1, "checkpoint_period": 1, "anchor_period": 1, "class": "0"},
        remote_subdir="explanations",
        array=np.arange(4, dtype=float),
    )
    index = tmp_path / "results" / "explanations" / "index.csv"
    _write_index(index, [row])
    with pytest.raises(AnalysisWiringError, match="lacks 'strategy'"):
        load_explanations_from_cache(index, explainer="ig", k=5, strategy="ft")


def test_noise_loader_preserves_all_r_reseeds_and_repeats(tmp_path):
    from src.analysis_wiring.frames import load_noise_explanations_from_cache

    rows = []
    for seed in range(142, 147):
        metadata = {
            "explainer": "ig",
            "anchor_id": 99,
            "seed": seed,
            "checkpoint_period": 1,  # producer records anchor period for every reseed
            "anchor_period": 1,
            "class": "2",
            "label": 2,
            "strategy": "ft",
            "model": "egraphsage",
        }
        rows.append(
            _write_entry(
                tmp_path,
                name=f"null_{seed}",
                explainer="ig",
                k=5,
                anchor_id=99,
                metadata=metadata,
                remote_subdir="explanations_noise/model_null",
                array=np.arange(4, dtype=float) + seed,
            )
        )
    index = tmp_path / "results" / "explanations_noise" / "model_null" / "index.csv"
    _write_index(index, rows)
    grouped = load_noise_explanations_from_cache(index, explainer="ig", k=5, strategy="ft", model="egraphsage", seed=None)
    assert len(grouped[99]) == 5, "all R=5 null reseeds must survive despite the shared checkpoint_period"
    assert [entry["seed"] for entry in grouped[99]] == list(range(142, 147))

    repeat_rows = []
    for seed in range(42000, 42010):
        metadata = {
            "explainer": "ig",
            "anchor_id": 99,
            "seed": seed,
            "checkpoint_period": 1,
            "anchor_period": 1,
            "class": "2",
            "label": 2,
            "strategy": "ft",
            "model": "egraphsage",
        }
        repeat_rows.append(
            _write_entry(
                tmp_path,
                name=f"repeat_{seed}",
                explainer="ig",
                k=5,
                anchor_id=99,
                metadata=metadata,
                remote_subdir="explanations_noise/explainer_repeat",
                array=np.arange(4, dtype=float) + seed,
            )
        )
    repeat_index = tmp_path / "results" / "explanations_noise" / "explainer_repeat" / "index.csv"
    _write_index(repeat_index, repeat_rows)
    repeats = load_noise_explanations_from_cache(
        repeat_index, explainer="ig", k=5, strategy="ft", model="egraphsage", seed=None
    )
    assert len(repeats[99]) == 10
    assert [entry["seed"] for entry in repeats[99]] == list(range(42000, 42010))


def test_gnn_explainer_k_is_normalised_to_zero(tmp_path):
    from src.analysis_wiring.frames import load_explanations_from_cache

    row = _write_entry(
        tmp_path,
        name="mask",
        explainer="gnn_explainer",
        k=0,
        anchor_id=3,
        metadata={
            "explainer": "gnn_explainer",
            "anchor_id": 3,
            "seed": 42,
            "checkpoint_period": 1,
            "anchor_period": 1,
            "class": "6",
            "label": 6,
            "strategy": "ft",
            "model": "egraphsage",
        },
        remote_subdir="explanations",
        array=np.abs(np.arange(6, dtype=float)),
    )
    index = tmp_path / "results" / "tables" / "explain_index.csv"
    _write_index(index, [row])
    explanations, _ = load_explanations_from_cache(
        index, explainer="gnn_explainer", k=5, strategy="ft", model="egraphsage", seed=42
    )
    assert (1, 3) in explanations


def test_shared_result_path_relocation_helper(tmp_path):
    from src.utils.artifact_paths import resolve_artifact_path

    local = tmp_path / "results" / "explanations" / "abc.npz"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"x")
    index = tmp_path / "results" / "tables" / "explain_index.csv"
    index.parent.mkdir(parents=True)
    index.write_text("", encoding="utf-8")
    resolved = resolve_artifact_path("/kaggle/working/stage/results/text/explanations/abc.npz", index)
    assert resolved == local
