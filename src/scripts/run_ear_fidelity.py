"""Bounded C3L-inspired anchor-fidelity diagnostic for EAR-A (T57).

Compares IG sufficiency/comprehensiveness on a deterministic bounded sample of the
canonical anchors between EAR-A and the signed replay/FT baselines at the same
checkpoint(s), using the stored period-j contexts (no re-sampling).

This is deliberately bounded: ``--per-class`` anchors per (period, class) and one
(or a small list of) checkpoint(s); it is a diagnostic, not a full S7 re-run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.analysis import stats as st
from src.explain.fidelity import feature_fidelity
from src.explain.ig import integrated_gradients
from src.models.edge_sage import build_model

NODE_FEATURE_DIM = 8


def _load_anchor_sets(anchors_dir: Path) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(anchors_dir.glob("period_*.pt")):
        period = int(path.stem.split("_")[1])
        payload = torch.load(path, weights_only=False)
        out[period] = list(payload.get("anchors", []))
    return out


def select_anchor_sample(
    anchor_sets: dict[int, list[dict[str, Any]]], *, per_class: int
) -> list[dict[str, Any]]:
    """Deterministic stratified sample: first ``per_class`` anchors per (period,class)."""
    selected: list[dict[str, Any]] = []
    for period in sorted(anchor_sets):
        buckets: dict[int, list[dict[str, Any]]] = {}
        for anchor in anchor_sets[period]:
            buckets.setdefault(int(anchor["label"]), []).append(anchor)
        for label in sorted(buckets):
            for anchor in sorted(buckets[label], key=lambda item: int(item["anchor_id"]))[: int(per_class)]:
                selected.append(anchor)
    return selected


def _anchor_data(anchor: dict[str, Any]) -> Data:
    return Data(
        x=torch.ones((int(anchor["n_sub_nodes"]), NODE_FEATURE_DIM), dtype=torch.float32),
        edge_index=anchor["sub_edge_index"].long(),
        edge_attr=anchor["sub_edge_attr"].float(),
        y=torch.tensor([int(anchor["label"])], dtype=torch.long),
    )


def _load_model(checkpoint: Path, model_name: str, edge_dim: int, device: str):
    payload = torch.load(checkpoint, weights_only=False)
    model = build_model(
        model_name,
        node_dim=NODE_FEATURE_DIM,
        edge_dim=int(edge_dim),
        config={"model": payload.get("model_config") or {}},
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model


def evaluate_fidelity(
    *,
    anchors_dir: Path,
    groups: Sequence[tuple[str, str, int]],
    runs_roots: Sequence[Path],
    checkpoints: Sequence[int],
    per_class: int,
    k: int = 5,
    random_trials: int = 2,
    device: str = "cpu",
    seed: int = 42,
    out_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchor_sets = _load_anchor_sets(anchors_dir)
    anchors = select_anchor_sample(anchor_sets, per_class=per_class)
    rows: list[dict[str, Any]] = []
    for strategy, model_name, group_seed in groups:
        for checkpoint in checkpoints:
            checkpoint_path = None
            for root in runs_roots:
                candidate = Path(root) / strategy / model_name / f"seed{group_seed}" / f"M_{checkpoint}.pt"
                if candidate.is_file():
                    checkpoint_path = candidate
                    break
            if checkpoint_path is None:
                rows.append(
                    {
                        "strategy": strategy,
                        "model": model_name,
                        "seed": group_seed,
                        "checkpoint": checkpoint,
                        "status": "missing_checkpoint",
                    }
                )
                continue
            edge_dim = int(anchors[0]["sub_edge_attr"].shape[1])
            model = _load_model(checkpoint_path, model_name, edge_dim, device)
            for anchor in anchors:
                data = _anchor_data(anchor)
                target = int(anchor["target_local_index"][0])
                ig = integrated_gradients(model, data, target, steps=32, device=device)
                fidelity = feature_fidelity(
                    model,
                    data,
                    target,
                    ig["attributions"],
                    k_values=(k,),
                    random_trials=random_trials,
                    seed=int(seed) + int(anchor["anchor_id"]),
                    device=device,
                )
                value = fidelity[0] if fidelity else {}
                rows.append(
                    {
                        "strategy": strategy,
                        "model": model_name,
                        "seed": group_seed,
                        "checkpoint": checkpoint,
                        "anchor_id": int(anchor["anchor_id"]),
                        "period": int(anchor["period"]),
                        "class": int(anchor["label"]),
                        "k": k,
                        "sufficiency_ratio": value.get("sufficiency_ratio"),
                        "comprehensiveness": value.get("comprehensiveness"),
                        "comprehensiveness_over_random": value.get("comprehensiveness_over_random"),
                        "status": "ok",
                    }
                )
    frame = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

    # Paired EAR-A vs each baseline on the same anchors/checkpoints.
    deltas: list[dict[str, Any]] = []
    ok = frame[frame["status"] == "ok"].copy()
    if not ok.empty:
        ear = ok[ok["strategy"] == "ear_a"]
        for baseline in sorted(set(ok["strategy"]) - {"ear_a"}):
            base = ok[ok["strategy"] == baseline]
            for metric in ("sufficiency_ratio", "comprehensiveness", "comprehensiveness_over_random"):
                left = (
                    ear.groupby(["model", "seed", "checkpoint", "anchor_id"])[metric]
                    .mean()
                    .reset_index()
                )
                right = (
                    base.groupby(["model", "seed", "checkpoint", "anchor_id"])[metric]
                    .mean()
                    .reset_index()
                )
                merged = left.merge(
                    right,
                    on=["model", "seed", "checkpoint", "anchor_id"],
                    suffixes=("_ear", "_base"),
                    how="inner",
                ).dropna()
                if len(merged) < 2:
                    continue
                paired = st.paired_bootstrap_diff(
                    merged[f"{metric}_ear"].to_numpy(dtype=float),
                    merged[f"{metric}_base"].to_numpy(dtype=float),
                    n_resamples=1000,
                    alpha=0.05,
                    seed=seed,
                )
                effect = st.effect_size_summary(
                    merged[f"{metric}_ear"].to_numpy(dtype=float),
                    merged[f"{metric}_base"].to_numpy(dtype=float),
                    paired=True,
                )
                deltas.append(
                    {
                        "comparison": f"ear_a_minus_{baseline}",
                        "metric": metric,
                        "n_pairs": int(paired.n),
                        "mean_ear": float(merged[f"{metric}_ear"].mean()),
                        "mean_baseline": float(merged[f"{metric}_base"].mean()),
                        "diff": paired.diff,
                        "ci_lo": paired.lo,
                        "ci_hi": paired.hi,
                        "p_value": paired.p_value,
                        "cohens_d": effect["cohens_d"],
                        "rank_biserial": effect["rank_biserial"],
                    }
                )
    delta_frame = pd.DataFrame(deltas)
    delta_frame.to_csv(out_path.with_name("fidelity_deltas.csv"), index=False)
    summary = {
        "status": "ok" if not ok.empty else "no_fidelity_rows",
        "anchors_per_group": len(anchors),
        "groups": [f"{s}:{m}:{seed}" for s, m, seed in groups],
        "checkpoints": list(checkpoints),
        "per_class": int(per_class),
        "k": int(k),
        "fidelity_rows": int(len(frame)),
        "ok_rows": int(len(ok)),
        "delta_rows": int(len(delta_frame)),
        "deltas": delta_frame.to_dict("records"),
        "caveat": "bounded fidelity diagnostic, not a full S7 re-run; association only",
    }
    out_path.with_name("fidelity_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    return frame, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_ear_fidelity", description=__doc__.splitlines()[0])
    parser.add_argument("--anchors-dir", required=True)
    parser.add_argument("--groups", required=True, help="e.g. ear_a:egraphsage:42,replay:egraphsage:42")
    parser.add_argument("--runs-roots", required=True, help="comma list of runs roots searched per group")
    parser.add_argument("--checkpoints", default="5")
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/text/tables/s12_ear/fidelity.csv")
    args = parser.parse_args(argv)
    groups = []
    for entry in args.groups.split(","):
        strategy, model, seed = entry.split(":")
        groups.append((strategy, model, int(seed)))
    runs_roots = [Path(item) for item in args.runs_roots.split(",") if item.strip()]
    evaluate_fidelity(
        anchors_dir=Path(args.anchors_dir),
        groups=groups,
        runs_roots=runs_roots,
        checkpoints=[int(item) for item in args.checkpoints.split(",") if item.strip()],
        per_class=int(args.per_class),
        k=int(args.k),
        device=args.device,
        seed=int(args.seed),
        out_path=Path(args.out),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
