"""Graph-lift pilot: does the host-flow graph carry signal beyond edge features?

T12/D-024 decisive gate item. Three conditions, identical features, class
weights, loss, optimiser, epoch budget and seeds:

(a) **MLP on edge features** - no graph at all;
(b) **E-GraphSAGE** - 2 edge-aware message-passing layers on the real graph;
(c) **E-GraphSAGE + randomised structure** - same features/labels, endpoints
    permuted so the graph topology carries no relation to the labels.

Verdict (pre-registered D-024):
    * Delta_mlp  = macro-F1(E-GraphSAGE) - macro-F1(MLP)  >= 1.0 point
    * Delta_rand = macro-F1(E-GraphSAGE) - macro-F1(randomised) >= 1.0 point
    * both on the 3-seed mean, with bootstrap 95% CI lower bound > 0.

The pilot runs on a stratified sample of period-1 edges so it is feasible on the
legacy 3 GB GPU lane; timing is measured (not guessed) and full-size timings are
explicitly projected and labelled as such.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data.preprocess import Preprocessor
from src.data.registry import DatasetSpec
from src.utils.logging import get_logger
from src.utils.manifest import code_fingerprint, git_info, utc_now, write_json
from src.utils.seed import seed_everything
from src.utils.torch_support import torch_available, torch_status, torch_unavailable_reason

LOGGER = get_logger("xdrift.pilot")

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": "auto",
    "period": 1,
    "sample_edges": 20_000,
    "sample_frac": 0.05,
    "hidden_dim": 64,
    "epochs": 15,
    "batch_size": 1024,
    "lr": 0.001,
    "seeds": [42, 43, 44],
    "prototype_randomised_structure": True,
    "bootstrap_samples": 500,
    "min_edges_per_class": 20,
    "node_feature_dim": 8,
}

VERDICT_THRESHOLD_POINTS = 1.0  # pre-registered (configs/audit.yaml also carries it)


def resolve_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    resolved = dict(DEFAULT_CONFIG)
    if config:
        for key, value in dict(config).items():
            if value is not None and key != "enabled":
                resolved[key] = value
    return resolved


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def _period_local_masks(frame: pd.DataFrame, splits: Any, period: int) -> dict[str, np.ndarray]:
    positions = np.flatnonzero(frame["_period"].to_numpy(dtype="int64") == period)
    local: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        global_mask = splits.masks[period][name]
        mapping = np.searchsorted(positions, global_mask)
        local[name] = mapping.astype("int64")
    return local


def _stratified_positions(labels: np.ndarray, frac: float, max_edges: int, min_per_class: int, rng) -> np.ndarray:
    chosen: list[np.ndarray] = []
    for class_id in np.unique(labels):
        positions = np.flatnonzero(labels == class_id)
        take = max(int(np.ceil(frac * len(positions))), min(min_per_class, len(positions)))
        take = min(take, len(positions))
        chosen.append(rng.choice(positions, size=take, replace=False))
    positions = np.sort(np.concatenate(chosen))
    if len(positions) > max_edges:
        positions = np.sort(rng.choice(positions, size=max_edges, replace=False))
    return positions


def _build_sample(
    frame: pd.DataFrame,
    splits: Any,
    preprocessor: Preprocessor,
    period: int,
    cfg: Mapping[str, Any],
    rng,
) -> dict[str, Any]:
    period_frame = frame[frame["_period"] == period].reset_index(drop=True)
    labels_all = period_frame["_class_id"].to_numpy(dtype="int64")
    positions = _stratified_positions(
        labels_all,
        float(cfg["sample_frac"]),
        int(cfg["sample_edges"]),
        int(cfg["min_edges_per_class"]),
        rng,
    )
    sampled = period_frame.iloc[positions]
    features = preprocessor.transform(sampled).astype("float32")
    labels = labels_all[positions]
    src_ids = sampled["_host_src_id"].to_numpy(dtype="int64")
    dst_ids = sampled["_host_dst_id"].to_numpy(dtype="int64")
    nodes = np.unique(np.concatenate([src_ids, dst_ids]))
    src = np.searchsorted(nodes, src_ids).astype("int64")
    dst = np.searchsorted(nodes, dst_ids).astype("int64")
    local_masks = _period_local_masks(frame, splits, period)
    position_set = {int(value): index for index, value in enumerate(positions.tolist())}
    masks: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        local = local_masks[name]
        indices = [position_set[int(value)] for value in local.tolist() if int(value) in position_set]
        masks[name] = np.array(indices, dtype="int64")
    return {
        "period_frame": period_frame,
        "positions": positions,
        "features": features,
        "labels": labels,
        "edge_index": np.stack([src, dst]),
        "n_nodes": int(len(nodes)),
        "masks": masks,
        "class_counts": {int(key): int(value) for key, value in zip(*np.unique(labels, return_counts=True))},
        "period_edges": int(len(period_frame)),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {str(int(label)): float(score) for label, score in zip(labels, scores)}


def _bootstrap_macro_f1(y_true, y_pred, labels, *, samples: int, rng) -> dict[str, float]:
    n = len(y_true)
    values = np.empty(samples, dtype="float64")
    for index in range(samples):
        take = rng.integers(0, n, size=n)
        values[index] = _macro_f1(y_true[take], y_pred[take], labels)
    return {"lower": float(np.percentile(values, 2.5)), "upper": float(np.percentile(values, 97.5))}


def _bootstrap_delta(y_true, pred_a, pred_b, labels, *, samples: int, rng) -> np.ndarray:
    n = len(y_true)
    values = np.empty(samples, dtype="float64")
    for index in range(samples):
        take = rng.integers(0, n, size=n)
        values[index] = _macro_f1(y_true[take], pred_a[take], labels) - _macro_f1(
            y_true[take], pred_b[take], labels
        )
    return values


# ---------------------------------------------------------------------------
# Models (torch imported lazily so the data stage can ignore this module)
# ---------------------------------------------------------------------------
def _build_mlp(torch, n_features: int, n_classes: int, hidden: int):
    return torch.nn.Sequential(
        torch.nn.Linear(n_features, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, n_classes),
    )


def _build_egsage(torch, n_features: int, n_classes: int, hidden: int, node_dim: int):
    from torch_geometric.nn import MessagePassing

    class EdgeSAGELayer(MessagePassing):
        def __init__(self, in_node: int, in_edge: int, out: int):
            super().__init__(aggr="mean")
            self.lin = torch.nn.Linear(in_node + in_edge, out)

        def forward(self, x, edge_index, edge_attr):
            aggregated = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            return torch.relu(self.lin(torch.cat([x, aggregated], dim=-1)))

        def message(self, edge_attr):
            return edge_attr

    class EdgeSAGEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = EdgeSAGELayer(node_dim, n_features, hidden)
            self.layer2 = EdgeSAGELayer(hidden, n_features, hidden)
            self.head = torch.nn.Linear(2 * hidden + n_features, n_classes)

        def forward(self, x, edge_index, edge_attr):
            hidden = self.layer1(x, edge_index, edge_attr)
            hidden = self.layer2(hidden, edge_index, edge_attr)
            src, dst = edge_index
            return self.head(torch.cat([hidden[src], hidden[dst], edge_attr], dim=-1))

    return EdgeSAGEModel()


def _train_and_evaluate(
    torch,
    condition: str,
    sample: Mapping[str, Any],
    n_classes: int,
    cfg: Mapping[str, Any],
    seed: int,
    device: Any,
) -> dict[str, Any]:
    features = torch.as_tensor(sample["features"], dtype=torch.float32, device=device)
    labels = torch.as_tensor(sample["labels"], dtype=torch.long, device=device)
    edge_index = torch.as_tensor(sample["edge_index"], dtype=torch.long, device=device)
    n_nodes = int(sample["n_nodes"])
    node_x = torch.ones((n_nodes, int(cfg["node_feature_dim"])), dtype=torch.float32, device=device)
    train_idx = torch.as_tensor(sample["masks"]["train"], dtype=torch.long, device=device)
    val_idx = torch.as_tensor(sample["masks"]["val"], dtype=torch.long, device=device)
    test_idx = torch.as_tensor(sample["masks"]["test"], dtype=torch.long, device=device)

    torch.manual_seed(seed)
    if condition == "mlp":
        model = _build_mlp(torch, features.shape[1], n_classes, int(cfg["hidden_dim"])).to(device)
    else:
        model = _build_egsage(
            torch, features.shape[1], n_classes, int(cfg["hidden_dim"]), int(cfg["node_feature_dim"])
        ).to(device)
    n_params = sum(parameter.numel() for parameter in model.parameters())

    counts = torch.bincount(labels[train_idx], minlength=n_classes).float()
    weights = torch.where(counts > 0, counts.sum() / (counts.clamp_min(1) * (counts > 0).sum()), torch.zeros_like(counts))
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])

    def logits_for(indices: Any = None):
        if condition == "mlp":
            return model(features)
        return model(node_x, edge_index, features)

    best_state = None
    best_val = -1.0
    epoch_seconds: list[float] = []
    start_train = time.monotonic()
    for epoch in range(epochs):
        epoch_start = time.monotonic()
        model.train()
        permutation = torch.randperm(len(train_idx), device=device)
        for start in range(0, len(permutation), batch_size):
            batch = train_idx[permutation[start : start + batch_size]]
            optimiser.zero_grad()
            logits = logits_for(batch)[batch]
            loss = criterion(logits, labels[batch])
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val_logits = logits_for()[val_idx]
            predictions = val_logits.argmax(dim=-1)
            val_f1 = _macro_f1(
                sample["labels"][sample["masks"]["val"]],
                predictions.detach().cpu().numpy(),
                np.unique(sample["labels"]),
            )
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        epoch_seconds.append(time.monotonic() - epoch_start)
    train_seconds = time.monotonic() - start_train
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    start_predict = time.monotonic()
    with torch.no_grad():
        test_logits = logits_for()[test_idx]
        predictions = test_logits.argmax(dim=-1).detach().cpu().numpy()
    predict_seconds = time.monotonic() - start_predict
    y_true = sample["labels"][sample["masks"]["test"]]
    labels_present = np.union1d(np.unique(y_true), np.unique(predictions))
    macro = _macro_f1(y_true, predictions, labels_present)
    return {
        "condition": condition,
        "macro_f1": macro,
        "per_class_f1": _per_class_f1(y_true, predictions, labels_present),
        "val_macro_f1_best": best_val,
        "timing": {
            "train_seconds": round(train_seconds, 3),
            "epoch_seconds_mean": round(float(np.mean(epoch_seconds)), 4),
            "predict_seconds": round(predict_seconds, 3),
            "n_params": int(n_params),
            "device": str(device),
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "_predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def _pinning(period_strategy: str | None, config_sha256: str | None) -> dict[str, Any]:
    """Code/config pinning embedded in the gate artifact (PR-5, D-030).

    Gate artifacts must pin the exact code and config even when the manifest is
    produced on a machine without git on PATH: the git SHA is resolved through
    the full-path fallbacks in ``src.utils.manifest`` and a code fingerprint is
    always recorded.
    """
    try:
        from src.utils.paths import find_project_root

        project_root = find_project_root()
    except Exception:  # noqa: BLE001 - pinning must not break the pilot
        project_root = None
    git = git_info(project_root)
    code = code_fingerprint(project_root)
    return {
        "git_sha": git.get("sha"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
        "git_commit_date": git.get("commit_date"),
        "git_available": git.get("available"),
        "git_executable": git.get("executable"),
        "code_fingerprint_sha256": code.get("sha256"),
        "code_file_count": code.get("file_count"),
        "config_sha256": config_sha256,
        "period_strategy": period_strategy,
    }


def run_graph_lift_pilot(
    frame: pd.DataFrame,
    splits: Any,
    preprocessor: Preprocessor,
    spec: DatasetSpec,
    *,
    config: Mapping[str, Any] | None = None,
    seed: int = 42,
    dataset: str | None = None,
    out_dir: str | Path | None = None,
    thresholds: Mapping[str, float] | None = None,
    period_strategy: str | None = None,
    config_sha256: str | None = None,
    run_label: str | None = None,
) -> dict[str, Any]:
    """Run the three-way graph-lift pilot and write JSON + markdown artifacts."""
    cfg = resolve_config(config)
    dataset_name = dataset or spec.name
    device_requested = "cuda"
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "graph_lift_pilot",
        "run_label": run_label,
        "dataset": dataset_name,
        "period": int(cfg["period"]),
        "generated_at_utc": utc_now(),
        "config": cfg,
        "torch": torch_status(),
        "period_strategy": period_strategy,
        "config_sha256": config_sha256,
        "pinning": _pinning(period_strategy, config_sha256),
        # D-030: what "randomised structure" means precisely - only the
        # destination endpoint is permuted, so the source out-degree sequence is
        # exact and the destination in-degree sequence is preserved as a multiset
        # while the edge-to-label relation of the topology is destroyed.
        "randomised_structure_semantics": (
            "destination endpoints permuted (out-degree exact, in-degree multiset preserved); "
            "same features, labels, loss, epochs and seeds as the real-graph arm"
        ),
        # D-030: the exact hierarchical CI method so the paper can cite it.
        "ci_method": {
            "per_seed": (
                f"{int(cfg.get('bootstrap_samples', 500))} unstratified bootstrap resamples of the "
                "period-1 test edges per seed; macro-F1 delta recomputed per resample"
            ),
            "across_seeds": (
                "1000 bootstrap resamples over the 3 seed-level per-seed delta distributions "
                "(seeds resampled with replacement, their bootstrap means averaged); 95% percentile CI "
                "of the 3-seed mean delta"
            ),
            "stratified": False,
            "bootstrap_samples_per_seed": int(cfg.get("bootstrap_samples", 500)),
            "seed_resamples": 1000,
        },
        "verdict_thresholds": {
            "graph_lift_macro_f1_min": float(
                (thresholds or {}).get("graph_lift_macro_f1_min", VERDICT_THRESHOLD_POINTS)
            ),
            "graph_lift_vs_randomised_min": float(
                (thresholds or {}).get("graph_lift_vs_randomised_min", VERDICT_THRESHOLD_POINTS)
            ),
        },
    }
    if not torch_available():
        report.update(
            {
                "status": "skipped",
                "reason": f"torch unavailable on this lane: {torch_unavailable_reason()}",
            }
        )
        if out_dir is not None:
            _write_artifacts(report, Path(out_dir))
        return report

    import torch  # noqa: PLC0415 - guarded by torch_available() above

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(value) for value in cfg.get("seeds", [seed])]
    period = int(cfg["period"]) - 1  # config uses 1-based period numbers
    per_seed: list[dict[str, Any]] = []
    sample_info: dict[str, Any] | None = None
    for run_seed in seeds:
        seed_everything(run_seed)
        rng = np.random.default_rng(run_seed)
        sample = _build_sample(frame, splits, preprocessor, period, cfg, rng)
        if sample_info is None:
            sample_info = {
                "sampled_edges": int(len(sample["labels"])),
                "period_edges": int(sample["period_edges"]),
                "sampled_fraction": float(len(sample["labels"]) / max(sample["period_edges"], 1)),
                "n_nodes": sample["n_nodes"],
                "class_counts": sample["class_counts"],
                "split_sizes": {name: int(len(mask)) for name, mask in sample["masks"].items()},
            }
        n_classes = int(max(sample["labels"])) + 1
        conditions = {"mlp": None, "egraphsage": None}
        results: dict[str, dict[str, Any]] = {}
        for condition in ("mlp", "egraphsage"):
            results[condition] = _train_and_evaluate(
                torch, condition, sample, n_classes, cfg, run_seed, device
            )
        if bool(cfg.get("prototype_randomised_structure", True)):
            randomised = dict(sample)
            rng_perm = np.random.default_rng(run_seed + 10_000)
            permutation = rng_perm.permutation(sample["edge_index"].shape[1])
            randomised["edge_index"] = np.stack(
                [sample["edge_index"][0], sample["edge_index"][1][permutation]]
            )
            results["egraphsage_randomised"] = _train_and_evaluate(
                torch, "egraphsage", randomised, n_classes, cfg, run_seed + 10_000, device
            )
        y_true = sample["labels"][sample["masks"]["test"]]
        labels_present = np.union1d(np.unique(y_true), np.unique(np.concatenate([r["_predictions"] for r in results.values()])))
        rng_boot = np.random.default_rng(run_seed + 20_000)
        bootstrap_samples = int(cfg.get("bootstrap_samples", 500))
        for condition, result in results.items():
            result["ci95_macro_f1"] = _bootstrap_macro_f1(
                y_true, result["_predictions"], labels_present, samples=bootstrap_samples, rng=rng_boot
            )
        delta_mlp = _bootstrap_delta(
            y_true,
            results["egraphsage"]["_predictions"],
            results["mlp"]["_predictions"],
            labels_present,
            samples=bootstrap_samples,
            rng=np.random.default_rng(run_seed + 30_000),
        )
        delta_rand = _bootstrap_delta(
            y_true,
            results["egraphsage"]["_predictions"],
            results["egraphsage_randomised"]["_predictions"],
            labels_present,
            samples=bootstrap_samples,
            rng=np.random.default_rng(run_seed + 40_000),
        )
        per_seed.append(
            {
                "seed": run_seed,
                "models": {
                    condition: {
                        key: value
                        for key, value in result.items()
                        if key != "_predictions"
                    }
                    for condition, result in results.items()
                },
                "delta_macro_f1_egsage_minus_mlp": float(
                    results["egraphsage"]["macro_f1"] - results["mlp"]["macro_f1"]
                ),
                "delta_macro_f1_egsage_minus_randomised": float(
                    results["egraphsage"]["macro_f1"] - results["egraphsage_randomised"]["macro_f1"]
                ),
                "delta_points_egsage_minus_mlp": float(
                    100.0 * (results["egraphsage"]["macro_f1"] - results["mlp"]["macro_f1"])
                ),
                "delta_points_egsage_minus_randomised": float(
                    100.0
                    * (
                        results["egraphsage"]["macro_f1"]
                        - results["egraphsage_randomised"]["macro_f1"]
                    )
                ),
                "bootstrap_delta_mlp": delta_mlp,
                "bootstrap_delta_randomised": delta_rand,
            }
        )
        LOGGER.info(
            "pilot seed %s: MLP %.4f | E-GS %.4f | E-GS-rand %.4f",
            run_seed,
            results["mlp"]["macro_f1"],
            results["egraphsage"]["macro_f1"],
            results["egraphsage_randomised"]["macro_f1"],
        )

    model_keys = ["mlp", "egraphsage", "egraphsage_randomised"]
    mean_scores = {
        key: float(np.mean([entry["models"][key]["macro_f1"] for entry in per_seed]))
        for key in model_keys
    }
    mean_delta_mlp = float(np.mean([entry["delta_macro_f1_egsage_minus_mlp"] for entry in per_seed]))
    mean_delta_rand = float(
        np.mean([entry["delta_macro_f1_egsage_minus_randomised"] for entry in per_seed])
    )

    def _seed_bootstrap_ci(arrays: Sequence[np.ndarray], *, samples: int = 1000) -> dict[str, float]:
        rng = np.random.default_rng(12345)
        values = np.empty(samples)
        for index in range(samples):
            chosen = rng.integers(0, len(arrays), size=len(arrays))
            values[index] = float(np.mean([arrays[int(position)].mean() for position in chosen]))
        return {"lower": float(np.percentile(values, 2.5)), "upper": float(np.percentile(values, 97.5))}

    ci_mlp = _seed_bootstrap_ci([entry["bootstrap_delta_mlp"] for entry in per_seed])
    ci_rand = _seed_bootstrap_ci([entry["bootstrap_delta_randomised"] for entry in per_seed])
    thresholds_used = report["verdict_thresholds"]
    # Pre-registered thresholds are in macro-F1 POINTS (1.0 = one point); the
    # deltas are stored as fractions, so convert for the verdict and keep both
    # values in the report to avoid unit ambiguity.
    delta_points_mlp = 100.0 * mean_delta_mlp
    delta_points_rand = 100.0 * mean_delta_rand
    ci_mlp_points = {"lower": 100.0 * ci_mlp["lower"], "upper": 100.0 * ci_mlp["upper"]}
    ci_rand_points = {"lower": 100.0 * ci_rand["lower"], "upper": 100.0 * ci_rand["upper"]}

    report.update(
        {
            "status": "success",
            "device": str(device),
            "torch": torch_status(),
            "sample": sample_info,
            "seeds": seeds,
            "per_seed": per_seed,
            "mean_over_seeds": {
                "models": mean_scores,
                "delta_macro_f1_egsage_minus_mlp": mean_delta_mlp,
                "delta_macro_f1_egsage_minus_randomised": mean_delta_rand,
                "delta_points_egsage_minus_mlp": delta_points_mlp,
                "delta_points_egsage_minus_randomised": delta_points_rand,
            },
            "verdict": {
                "graph_lift_macro_f1": {
                    "mean_delta_points": delta_points_mlp,
                    "mean_delta_fraction": mean_delta_mlp,
                    "ci95_points": ci_mlp_points,
                    "ci95_fraction": ci_mlp,
                    "threshold_points": thresholds_used["graph_lift_macro_f1_min"],
                    "pass": bool(
                        delta_points_mlp >= thresholds_used["graph_lift_macro_f1_min"]
                        and ci_mlp_points["lower"] > 0.0
                    ),
                },
                "graph_lift_vs_randomised": {
                    "mean_delta_points": delta_points_rand,
                    "mean_delta_fraction": mean_delta_rand,
                    "ci95_points": ci_rand_points,
                    "ci95_fraction": ci_rand,
                    "threshold_points": thresholds_used["graph_lift_vs_randomised_min"],
                    "pass": bool(
                        delta_points_rand >= thresholds_used["graph_lift_vs_randomised_min"]
                        and ci_rand_points["lower"] > 0.0
                    ),
                },
            },
            "timing": _timing_table(per_seed, sample_info or {}),
        }
    )
    report["verdict"]["overall"] = (
        "pass"
        if report["verdict"]["graph_lift_macro_f1"]["pass"]
        and report["verdict"]["graph_lift_vs_randomised"]["pass"]
        else "fail"
    )
    # D-037: report the same statistics on nested seed subsets so a 10-seed
    # stability run shows both its own result and the pre-registered 3-seed
    # protocol result in one pinned artifact.
    subset_sizes = sorted({3, len(seeds)}) if len(seeds) > 3 else [len(seeds)]
    subset_summaries: dict[str, Any] = {}
    for size in subset_sizes:
        subset = per_seed[:size]
        if len(subset) != size:
            continue
        scores = {
            key: float(np.mean([entry["models"][key]["macro_f1"] for entry in subset]))
            for key in model_keys
        }
        d_mlp = float(np.mean([entry["delta_macro_f1_egsage_minus_mlp"] for entry in subset]))
        d_rand = float(np.mean([entry["delta_macro_f1_egsage_minus_randomised"] for entry in subset]))
        ci_mlp_subset = _seed_bootstrap_ci([entry["bootstrap_delta_mlp"] for entry in subset])
        ci_rand_subset = _seed_bootstrap_ci([entry["bootstrap_delta_randomised"] for entry in subset])
        subset_summaries[str(size)] = {
            "seeds": [entry["seed"] for entry in subset],
            "models": scores,
            "delta_points_egsage_minus_mlp": 100.0 * d_mlp,
            "delta_points_egsage_minus_randomised": 100.0 * d_rand,
            "ci95_points_egsage_minus_mlp": {
                "lower": 100.0 * ci_mlp_subset["lower"],
                "upper": 100.0 * ci_mlp_subset["upper"],
            },
            "ci95_points_egsage_minus_randomised": {
                "lower": 100.0 * ci_rand_subset["lower"],
                "upper": 100.0 * ci_rand_subset["upper"],
            },
            "verdict": {
                "graph_lift_macro_f1_pass": bool(
                    100.0 * d_mlp >= thresholds_used["graph_lift_macro_f1_min"]
                    and ci_mlp_subset["lower"] > 0.0
                ),
                "graph_lift_vs_randomised_pass": bool(
                    100.0 * d_rand >= thresholds_used["graph_lift_vs_randomised_min"]
                    and ci_rand_subset["lower"] > 0.0
                ),
            },
        }
    report["subset_summaries"] = subset_summaries
    if out_dir is not None:
        _write_artifacts(report, Path(out_dir))
    return report


def _timing_table(per_seed: Sequence[Mapping[str, Any]], sample_info: Mapping[str, Any]) -> dict[str, Any]:
    """Measured sample-scale timing + explicitly projected full-size timings."""
    model_keys = ["mlp", "egraphsage", "egraphsage_randomised"]
    table: dict[str, Any] = {"measured_sample_scale": {}, "projected_full_size": {}}
    for key in model_keys:
        epoch_means = [
            entry["models"][key]["timing"]["epoch_seconds_mean"]
            for entry in per_seed
            if key in entry["models"]
        ]
        per_epoch = float(np.mean(epoch_means)) if epoch_means else None
        table["measured_sample_scale"][key] = {
            "epoch_seconds_mean": round(per_epoch, 4) if per_epoch is not None else None,
            "train_seconds_mean": round(
                float(
                    np.mean(
                        [
                            entry["models"][key]["timing"]["train_seconds"]
                            for entry in per_seed
                            if key in entry["models"]
                        ]
                    )
                ),
                3,
            )
            if epoch_means
            else None,
            "device": next(
                (
                    entry["models"][key]["timing"]["device"]
                    for entry in per_seed
                    if key in entry["models"]
                ),
                None,
            ),
            "n_params": next(
                (entry["models"][key]["timing"]["n_params"] for entry in per_seed if key in entry["models"]),
                None,
            ),
        }
        sampled = int(sample_info.get("sampled_edges", 0))
        full = int(sample_info.get("period_edges", 0))
        if per_epoch is not None and sampled:
            table["projected_full_size"][key] = {
                "projected_epoch_seconds": round(per_epoch * (full / sampled), 2),
                "projected_full_period_train_seconds_15_epochs": round(
                    per_epoch * (full / sampled) * 15, 1
                ),
                "projected_5_period_seconds": round(per_epoch * (full / sampled) * 15 * 5, 1),
                "note": "linear projection from measured sample timing; marked projected, verify on the RTX 5060 canonical lane",
            }
    return table


def _verdict_markdown(report: Mapping[str, Any]) -> str:
    verdict = report.get("verdict", {})
    mean = report.get("mean_over_seeds", {})
    models = mean.get("models", {})
    lines = [
        f"# Graph-lift pilot: {report['dataset']}",
        "",
        f"- **Status**: {report.get('status')}",
        f"- **Generated (UTC)**: {report.get('generated_at_utc')}",
        f"- **Period**: {report.get('period')} | **Device**: {report.get('device')}",
        f"- **Seeds**: {report.get('seeds')}",
        f"- **Sample**: {report.get('sample')}",
        "",
        "## Models (mean over seeds, period-1 test mask)",
        "",
        "| Condition | Macro-F1 | E-GraphSAGE delta |",
        "| --- | ---: | ---: |",
        f"| MLP on edge features (no graph) | {models.get('mlp', 0):.4f} | - |",
        f"| E-GraphSAGE (real graph) | {models.get('egraphsage', 0):.4f} | "
        f"{mean.get('delta_macro_f1_egsage_minus_mlp', 0):+.4f} vs MLP |",
        f"| E-GraphSAGE (randomised structure) | {models.get('egraphsage_randomised', 0):.4f} | "
        f"{mean.get('delta_macro_f1_egsage_minus_randomised', 0):+.4f} vs randomised |",
        "",
        "## Per-seed results",
        "",
        "| Seed | MLP | E-GS | E-GS randomised | Delta vs MLP | Delta vs randomised |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in report.get("per_seed", []):
        entry_models = entry.get("models", {})
        lines.append(
            f"| {entry['seed']} | {entry_models['mlp']['macro_f1']:.4f} | "
            f"{entry_models['egraphsage']['macro_f1']:.4f} | "
            f"{entry_models['egraphsage_randomised']['macro_f1']:.4f} | "
            f"{entry['delta_points_egsage_minus_mlp']:+.2f} | "
            f"{entry['delta_points_egsage_minus_randomised']:+.2f} |"
        )
    lines += [
        "",
        "Deltas are in macro-F1 **points** (1.0 = one percentage point).",
        "",
        "## D-024 verdict",
        "",
    ]
    if report.get("status") == "success":
        for key, label in (
            ("graph_lift_macro_f1", "Delta(E-GS - MLP)"),
            ("graph_lift_vs_randomised", "Delta(E-GS - randomised)"),
        ):
            block = verdict[key]
            lines.append(
                f"- **{label}**: mean {block['mean_delta_points']:+.2f} points, 95% CI "
                f"[{block['ci95_points']['lower']:+.2f}, {block['ci95_points']['upper']:+.2f}], "
                f"threshold >= {block['threshold_points']} point -> "
                f"**{'PASS' if block['pass'] else 'FAIL'}**"
            )
        lines += [
            "",
            f"**Overall D-024 verdict: {verdict.get('overall', 'unknown').upper()}**",
            "",
        ]
    else:
        lines += [f"- Skipped: {report.get('reason')}", ""]
    lines += [
        "",
        "## Method notes (citable)",
        "",
        f"- **Run label**: {report.get('run_label')}",
        f"- **Randomised-structure arm**: {report.get('randomised_structure_semantics')}",
        f"- **CI, per seed**: {report.get('ci_method', {}).get('per_seed')}",
        f"- **CI, across seeds**: {report.get('ci_method', {}).get('across_seeds')}",
        "",
    ]
    subsets = report.get("subset_summaries") or {}
    if len(subsets) > 1:
        lines += ["### Seed-subset stability summaries", "", "| Seeds | n | Delta vs MLP (pts) | 95% CI | Delta vs rand (pts) | 95% CI | Both pass |", "| --- | ---: | ---: | --- | ---: | --- | --- |"]
        for size, block in sorted(subsets.items(), key=lambda item: int(item[0])):
            verdict = block["verdict"]
            lines.append(
                f"| {block['seeds']} | {size} | {block['delta_points_egsage_minus_mlp']:+.2f} | "
                f"[{block['ci95_points_egsage_minus_mlp']['lower']:+.2f}, {block['ci95_points_egsage_minus_mlp']['upper']:+.2f}] | "
                f"{block['delta_points_egsage_minus_randomised']:+.2f} | "
                f"[{block['ci95_points_egsage_minus_randomised']['lower']:+.2f}, {block['ci95_points_egsage_minus_randomised']['upper']:+.2f}] | "
                f"{'YES' if verdict['graph_lift_macro_f1_pass'] and verdict['graph_lift_vs_randomised_pass'] else 'NO'} |"
            )
        lines.append("")
    lines += [
        "",
        "## Code/config pinning (gate artifact)",
        "",
        f"- git SHA: `{report.get('pinning', {}).get('git_sha')}` "
        f"(branch {report.get('pinning', {}).get('git_branch')}, dirty={report.get('pinning', {}).get('git_dirty')})",
        f"- code fingerprint SHA-256: `{report.get('pinning', {}).get('code_fingerprint_sha256')}` "
        f"({report.get('pinning', {}).get('code_file_count')} files)",
        f"- config SHA-256: `{report.get('config_sha256')}`",
        f"- period strategy: `{report.get('period_strategy')}`",
        "",
    ]
    timing = report.get("timing", {})
    lines += ["## Timing (measured on the sample)", "", "| Condition | s/epoch | s/train (15 ep) | device | params |", "| --- | ---: | ---: | --- | ---: |"]
    for key, block in timing.get("measured_sample_scale", {}).items():
        lines.append(
            f"| {key} | {block['epoch_seconds_mean']} | {block['train_seconds_mean']} | "
            f"{block['device']} | {block['n_params']} |"
        )
    lines += ["", "## Projected full-size timing (clearly projected)", ""]
    for key, block in timing.get("projected_full_size", {}).items():
        lines.append(
            f"- {key}: {block['projected_epoch_seconds']} s/epoch; "
            f"{block['projected_full_period_train_seconds_15_epochs']} s per full period (15 epochs); "
            f"{block['projected_5_period_seconds']} s for 5 periods"
        )
    lines += [
        "",
        "_Sample-scale timings are measured on the legacy lane (GTX 1050 3 GB). Full-size numbers are",
        "linear projections and are marked projected until the canonical RTX 5060 lane runs the pilot._",
        "",
    ]
    return "\n".join(lines)


def _write_artifacts(report: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "graph_lift_pilot.json", dict(report))
    (out_dir / "graph_lift_pilot.md").write_text(_verdict_markdown(report), encoding="utf-8")


__all__ = ["DEFAULT_CONFIG", "resolve_config", "run_graph_lift_pilot"]
