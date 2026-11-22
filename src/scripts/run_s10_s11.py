"""S10 RQ2 (structural change vs excess drift) + S11 RQ3 (drift -> later forgetting).

CPU-only analysis of the signed S8/S9 artifacts (T55).  Metric definitions are
delegated to ``analysis.tables`` / ``analysis.rq3`` / ``analysis.stats``; this
script only assembles frames from frozen/signed artifacts, computes anchor
model-confidence predictors, caches neighbourhood sets, and writes tables.

Outputs (never touching signed S8/S9):
  results/text/tables/s10_rq2/{anchor_level.csv, correlations.csv, summary.json}
  results/text/tables/s11_rq3/{features.csv, forgetting_labels.csv, auc.csv,
                          delta_auc.csv, coefficients.csv, auc_by_group.csv,
                          delta_auc_by_group.csv, summary.json}
  results/text/tables/s10_s11/anchor_predictions.csv  (cache)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.analysis import metrics_cl as mc
from src.analysis import rq3 as rq3mod
from src.analysis import stats as st
from src.analysis import tables as tables_mod
from src.analysis_wiring.frames import build_neighbourhood_frame
from src.models.edge_sage import build_model

NODE_FEATURE_DIM = 8
SEED = 42
GROUPS: tuple[tuple[str, str], ...] = (
    ("ft", "egraphsage"),
    ("ft", "gcn"),
    ("replay", "egraphsage"),
    ("replay", "gcn"),
    ("ewc", "egraphsage"),
    ("ewc", "gcn"),
)
EXPLAINERS: tuple[str, ...] = ("ig", "gnn_explainer")


def _load_anchors(anchors_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    anchor_sets: dict[int, dict[str, Any]] = {}
    anchor_meta: dict[int, dict[str, Any]] = {}
    for path in sorted(anchors_dir.glob("period_*.pt")):
        period = int(path.stem.split("_")[1])
        payload = torch.load(path, weights_only=False)
        anchor_sets[period] = payload
        for anchor in payload.get("anchors", []):
            anchor_meta[int(anchor["anchor_id"])] = anchor
    return anchor_sets, anchor_meta


def _anchor_data(anchor: Mapping[str, Any], node_feature_dim: int = NODE_FEATURE_DIM) -> Data:
    return Data(
        x=torch.ones((int(anchor["n_sub_nodes"]), int(node_feature_dim)), dtype=torch.float32),
        edge_index=anchor["sub_edge_index"].long(),
        edge_attr=anchor["sub_edge_attr"].float(),
        y=torch.tensor([int(anchor["label"])], dtype=torch.long),
    )


def compute_anchor_predictions(
    *,
    anchors_dir: Path,
    runs_root: Path,
    graphs_dir: Path,
    out_path: Path,
) -> pd.DataFrame:
    """Cache confidence/margin/CE/retention for each (group, checkpoint, anchor)."""
    anchor_sets, anchor_meta = _load_anchors(anchors_dir)
    graph_files = sorted(graphs_dir.glob("period_*.pt"))
    graph = torch.load(graph_files[0], weights_only=False)
    node_feature_dim = int(graph.x.shape[1])
    data_cache = {
        anchor_id: _anchor_data(anchor, node_feature_dim)
        for anchor_id, anchor in anchor_meta.items()
    }
    rows: list[dict[str, Any]] = []
    for strategy, model_name in GROUPS:
        for checkpoint in range(1, 6):
            checkpoint_path = runs_root / strategy / model_name / f"seed{SEED}" / f"M_{checkpoint}.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"missing checkpoint {checkpoint_path}")
            payload = torch.load(checkpoint_path, weights_only=False)
            model = build_model(
                model_name,
                node_dim=node_feature_dim,
                edge_dim=int(anchor_meta[next(iter(anchor_meta))]["sub_edge_attr"].shape[1]),
                config={"model": payload.get("model_config") or {}},
            )
            model.load_state_dict(payload["state_dict"])
            model.eval()
            with torch.no_grad():
                for anchor_id, anchor in anchor_meta.items():
                    anchor_period = int(anchor["period"])
                    if anchor_period > checkpoint:
                        continue
                    data = data_cache[anchor_id]
                    logits = model(data.x, data.edge_index, data.edge_attr)
                    target = int(anchor["target_local_index"][0])
                    target_logits = logits[target].detach().float()
                    probabilities = torch.softmax(target_logits, dim=-1)
                    label = int(anchor["label"])
                    confidence = float(probabilities.max())
                    margin = float(torch.topk(probabilities, k=2).values[0] - torch.topk(probabilities, k=2).values[1])
                    ce = float(-torch.log(probabilities[label].clamp_min(1e-12)))
                    predicted = int(probabilities.argmax())
                    rows.append(
                        {
                            "strategy": strategy,
                            "model": model_name,
                            "seed": SEED,
                            "checkpoint": checkpoint,
                            "anchor_id": anchor_id,
                            "anchor_period": anchor_period,
                            "class": str(label),
                            "predicted_class": predicted,
                            "confidence": confidence,
                            "margin": margin,
                            "ce": ce,
                            "retained": int(predicted == label),
                        }
                    )
    frame = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def _host_presence(
    neighbourhood: pd.DataFrame,
    anchor_meta: Mapping[int, Mapping[str, Any]],
    graphs: Mapping[int, Any],
) -> pd.Series:
    host_sets = {period: {int(value) for value in graph.node_global_ids.tolist()} for period, graph in graphs.items()}
    presence: list[bool] = []
    for row in neighbourhood.itertuples(index=False):
        anchor = anchor_meta[int(row.anchor_id)]
        hosts = host_sets[int(row.checkpoint_t)]
        presence.append(
            int(anchor["host_src_global"]) in hosts and int(anchor["host_dst_global"]) in hosts
        )
    return pd.Series(presence, index=neighbourhood.index, name="hosts_present")


def build_s10_frames(
    *,
    s8_tables: Path,
    full_root: Path,
    graphs_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchors_dir = full_root / "results" / "anchors" / "canonical"
    anchor_sets, anchor_meta = _load_anchors(anchors_dir)
    graphs = {
        int(path.stem.split("_")[1]): torch.load(path, weights_only=False)
        for path in sorted(graphs_dir.glob("period_*.pt"))
    }
    neighbourhood_frames: list[pd.DataFrame] = []
    for strategy, model_name in GROUPS:
        neighbourhood_frames.append(
            build_neighbourhood_frame(
                graphs,
                anchor_sets,
                strategy=strategy,
                model=model_name,
                explainer="ig",
                seed=SEED,
            )
        )
    neighbourhood = pd.concat(neighbourhood_frames, ignore_index=True)
    neighbourhood = neighbourhood.drop(columns=["explainer"])
    neighbourhood["hosts_present"] = _host_presence(neighbourhood, anchor_meta, graphs)
    signed = pd.read_csv(s8_tables / "rq1_anchor_level.csv")
    signed = signed[signed["explainer"].isin(EXPLAINERS)].copy()
    signed["class"] = signed["class"].astype(str)
    neighbourhood["class"] = neighbourhood["class"].astype(str)
    merge_keys = ["strategy", "model", "seed", "anchor_id", "class", "checkpoint_j", "checkpoint_t"]
    frame = signed.merge(neighbourhood, on=merge_keys, how="inner", validate="many_to_one")
    return frame, neighbourhood


def run_s10(
    *,
    s8_tables: Path,
    full_root: Path,
    graphs_dir: Path,
    out_dir: Path,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor_level, neighbourhood = build_s10_frames(
        s8_tables=s8_tables, full_root=full_root, graphs_dir=graphs_dir
    )
    predictions_small = predictions[
        ["strategy", "model", "seed", "checkpoint", "anchor_id", "retained", "confidence", "margin", "ce"]
    ].rename(columns={"checkpoint": "checkpoint_t"})
    anchor_level = anchor_level.merge(
        predictions_small,
        on=["strategy", "model", "seed", "checkpoint_t", "anchor_id"],
        how="left",
        validate="many_to_one",
    )
    anchor_level.to_csv(out_dir / "anchor_level.csv", index=False)
    neighbourhood.to_csv(out_dir / "neighbourhood_frame.csv", index=False)
    cfg = None
    frames: list[pd.DataFrame] = []
    variants = {
        "all": anchor_level,
        "presence_conditioned": anchor_level[anchor_level["hosts_present"].astype(bool)],
        "retention_conditioned": anchor_level[anchor_level["retained"] == 1],
    }
    for variant, subset in variants.items():
        if subset.empty:
            continue
        correlations = tables_mod.build_rq2_correlations(subset, cfg=cfg)
        correlations["variant"] = variant
        frames.append(correlations)
    # Null-update comparison: the reference null-mean association is the negative control.
    null_corr = tables_mod.build_rq2_correlations(anchor_level, value_column="null_mean", cfg=cfg)
    null_corr["variant"] = "null_mean_control"
    frames.append(null_corr)
    correlations = pd.concat(frames, ignore_index=True)
    correlations.to_csv(out_dir / "correlations.csv", index=False)
    summary = {
        "status": "ok",
        "anchor_level_rows": int(len(anchor_level)),
        "neighbourhood_rows": int(len(neighbourhood)),
        "variant_rows": {name: int(len(frame)) for name, frame in variants.items()},
        "hosts_present_fraction": float(anchor_level["hosts_present"].astype(float).mean()),
        "retained_fraction": float(anchor_level["retained"].astype(float).mean()),
        "correlation_groups": int(len(correlations)),
        "controls": "partial Spearman controls class+degree; period pair is the group",
        "null_comparison": "null_mean_control (reference noise-floor association) plus permutation p",
        "caveat": "association only; structural change is not labelled legitimate/harmful (docs/05 S10)",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_cl_tensors(per_class_csv: Path) -> dict[tuple[str, str, int], mc.RMetricTensor]:
    frame = pd.read_csv(per_class_csv)
    classes = list(dict.fromkeys(frame["class"].astype(str)))
    if len(classes) != 10:
        raise ValueError(f"expected 10 classes, got {len(classes)}")
    tensors: dict[tuple[str, str, int], mc.RMetricTensor] = {}
    for (strategy, model_name, seed), group in frame.groupby(["strategy", "model", "seed"], sort=True):
        r_class = np.full((5, 5, len(classes)), np.nan, dtype=float)
        for _, row in group.iterrows():
            i = int(row["checkpoint"]) - 1
            j = int(row["eval_period"]) - 1
            c = classes.index(str(row["class"]))
            r_class[i, j, c] = float(row["f1"])
        r_macro = np.nanmean(r_class, axis=2)
        tensors[(str(strategy), str(model_name), int(seed))] = mc.RMetricTensor(
            r_macro,
            R_class=r_class,
            classes=classes,
            strategy=str(strategy),
            model=str(model_name),
            seed=int(seed),
        )
    return tensors


def run_s11(
    *,
    s8_tables: Path,
    out_dir: Path,
    predictions: pd.DataFrame,
    seed: int = SEED,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class_frame = pd.read_csv(s8_tables / "per_class_f1.csv")
    class_order = list(dict.fromkeys(per_class_frame["class"].astype(str)))
    class_map = {str(index): name for index, name in enumerate(class_order)}
    predictions = predictions.copy()
    predictions["class"] = (
        predictions["class"].astype(str).map(class_map).fillna(predictions["class"].astype(str))
    )
    tensors = _build_cl_tensors(s8_tables / "per_class_f1.csv")
    label_frames: list[pd.DataFrame] = []
    for group_key, tensor in tensors.items():
        labels = rq3mod.build_forgetting_labels(tensor, delta=mc.DEFAULT_FORGETTING_DELTA, horizon="next_or_final")
        labels["explainer"] = "ig"
        label_frames.append(labels)
    labels_all = pd.concat(label_frames, ignore_index=True)
    labels_all.to_csv(out_dir / "forgetting_labels.csv", index=False)

    signed = pd.read_csv(s8_tables / "rq1_anchor_level.csv")
    signed["class"] = signed["class"].astype(str).map(class_map).fillna(signed["class"].astype(str))
    drift = signed[signed["explainer"] == "ig"].copy()
    drift = drift.merge(
        predictions[
            ["strategy", "model", "seed", "checkpoint", "anchor_id", "confidence", "margin", "ce", "retained"]
        ].rename(columns={"checkpoint": "checkpoint_t"}),
        on=["strategy", "model", "seed", "checkpoint_t", "anchor_id"],
        how="left",
        validate="many_to_one",
    )
    drift = drift.rename(columns={"checkpoint_t": "checkpoint"})
    retention_frame = (
        predictions.groupby(["strategy", "model", "checkpoint", "class"], dropna=False, sort=True)["retained"]
        .mean()
        .rename("retained_fraction")
        .reset_index()
    )
    features = rq3mod.build_rq3_features(
        labels_all,
        drift,
        retention_frame=retention_frame,
        drift_value_column="excess",
        checkpoint_column="checkpoint",
        class_column="class",
    )
    ce_frame = (
        predictions.groupby(["strategy", "model", "checkpoint", "class"], dropna=False, sort=True)["ce"]
        .mean()
        .rename("ce_mean")
        .reset_index()
    )
    features = features.merge(
        ce_frame,
        on=["strategy", "model", "checkpoint", "class"],
        how="left",
        validate="many_to_one",
    )
    features.to_csv(out_dir / "features.csv", index=False)
    config = rq3mod.RQ3Config(
        group_column="checkpoint", seed=seed, alpha=0.05, n_resamples=1000, delta=mc.DEFAULT_FORGETTING_DELTA
    )
    baseline_columns = ("confidence_mean", "margin_mean", "ce_mean", "retained_fraction", "f1")
    result = rq3mod.competing_baseline_analysis(
        features,
        baseline_columns=baseline_columns,
        cfg=config,
    )
    auc = result.auc_table.copy()
    delta = result.delta_auc_table.copy()
    if not delta.empty:
        delta["p_holm"] = st.holm_correction(delta["p_value"].to_numpy(dtype=float))
        delta["h3_supported_holm"] = (delta["ci_lo"] > 0.0) & (delta["p_holm"] < config.alpha)
    auc.to_csv(out_dir / "auc.csv", index=False)
    delta.to_csv(out_dir / "delta_auc.csv", index=False)
    result.coefficient_table.to_csv(out_dir / "coefficients.csv", index=False)

    # Per-group exploratory AUC/delta (Holm across all group x comparison tests).
    group_auc_rows: list[dict[str, Any]] = []
    group_delta_rows: list[dict[str, Any]] = []
    for (strategy, model_name, seed_value), group in features.groupby(
        ["strategy", "model", "seed"], sort=True, dropna=False
    ):
        try:
            group_result = rq3mod.competing_baseline_analysis(
                group,
                baseline_columns=baseline_columns,
                cfg=rq3mod.RQ3Config(
                    group_column="checkpoint",
                    seed=int(seed_value),
                    alpha=0.05,
                    n_resamples=500,
                    delta=mc.DEFAULT_FORGETTING_DELTA,
                ),
            )
        except rq3mod.RQ3Error as exc:
            group_delta_rows.append(
                {
                    "strategy": strategy,
                    "model": model_name,
                    "seed": int(seed_value),
                    "comparison": "not_evaluable",
                    "delta_auc": np.nan,
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                    "p_value": np.nan,
                    "note": str(exc),
                }
            )
            continue
        for row in group_result.auc_table.to_dict("records"):
            group_auc_rows.append({"strategy": strategy, "model": model_name, "seed": int(seed_value), **row})
        for row in group_result.delta_auc_table.to_dict("records"):
            group_delta_rows.append(
                {"strategy": strategy, "model": model_name, "seed": int(seed_value), **row}
            )
    group_auc = pd.DataFrame(group_auc_rows)
    group_delta = pd.DataFrame(group_delta_rows)
    if not group_delta.empty and "p_value" in group_delta.columns:
        finite = group_delta["p_value"].notna()
        group_delta.loc[finite, "p_holm"] = st.holm_correction(
            group_delta.loc[finite, "p_value"].to_numpy(dtype=float)
        )
    group_auc.to_csv(out_dir / "auc_by_group.csv", index=False)
    group_delta.to_csv(out_dir / "delta_auc_by_group.csv", index=False)

    summary = {
        "status": "ok",
        "label_rows": int(len(labels_all)),
        "feature_rows": int(len(features)),
        "label_positive": int(labels_all["label"].sum()),
        "drift_features": list(rq3mod.DRIFT_FEATURE_COLUMNS),
        "baseline_columns": list(baseline_columns),
        "cv_group": "checkpoint (leave-one-period-out over 1..4)",
        "pooled_auc": auc.to_dict("records"),
        "pooled_delta_auc": delta.to_dict("records"),
        "group_auc_rows": int(len(group_auc)),
        "group_delta_rows": int(len(group_delta)),
        "holm_scope": "within pooled delta comparisons; per-group Holm across group x comparison",
        "caveat": "exploratory, T=5; AUC/CI are association only, not causal (docs/05 S11)",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_s10_s11", description=__doc__.splitlines()[0])
    parser.add_argument("--s8-tables", required=True)
    parser.add_argument("--full-root", required=True)
    parser.add_argument("--graphs-dir", default="datasets/processed/NF-UNSW-NB15-v3")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out-root", default="results/text/tables")
    parser.add_argument("--skip-predictions", action="store_true")
    args = parser.parse_args(argv)
    s8_tables = Path(args.s8_tables)
    full_root = Path(args.full_root)
    graphs_dir = Path(args.graphs_dir)
    out_root = Path(args.out_root)
    cache_dir = out_root / "s10_s11"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = cache_dir / "anchor_predictions.csv"
    if args.skip_predictions and prediction_path.is_file():
        predictions = pd.read_csv(prediction_path)
    else:
        predictions = compute_anchor_predictions(
            anchors_dir=full_root / "results" / "anchors" / "canonical",
            runs_root=Path(args.runs_root),
            graphs_dir=graphs_dir,
            out_path=prediction_path,
        )
    s10 = run_s10(
        s8_tables=s8_tables,
        full_root=full_root,
        graphs_dir=graphs_dir,
        out_dir=out_root / "s10_rq2",
        predictions=predictions,
    )
    s11 = run_s11(
        s8_tables=s8_tables,
        out_dir=out_root / "s11_rq3",
        predictions=predictions,
    )
    print(json.dumps({"s10": s10, "s11": s11}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
