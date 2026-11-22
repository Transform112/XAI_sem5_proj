#!/usr/bin/env python3
"""Compare post-hoc decoders on saved step-06 checkpoints.

The training objective is unchanged.  This script only changes how class
probabilities are turned into a class prediction:

* ``fine``         - ordinary 10-class argmax (the baseline);
* ``prior``        - logit adjustment by the period-1 training prior;
* ``hierarchical`` - choose the coarse group first, then the best fine class
                     inside that group (coarse-to-fine decoding).

It is a diagnostic comparison, not a new training result.  The same checkpoints
and test rows are used for every decoder, so differences are decoding-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.trainer import macro_f1, per_class_f1  # noqa: E402
from src.models.edge_sage import build_model  # noqa: E402
from src.data.registry import load_dataset_spec  # noqa: E402

CLASS_NAMES = [
    "Benign",
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]

# The same coarse grouping used by the step-06 coarse-taxonomy experiment.
COARSE_GROUPS: list[list[int]] = [
    [0],
    [4],
    [5],
    [6],
    [7],
    [8],
    [1, 2, 3, 9],
]
COARSE_NAMES = ["Benign", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Rare"]


def fine_to_coarse() -> np.ndarray:
    mapping = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for coarse, members in enumerate(COARSE_GROUPS):
        for fine in members:
            mapping[fine] = coarse
    return mapping


def load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(
        payload["model"],
        node_dim=int(payload["node_dim"]),
        edge_dim=int(payload["edge_dim"]),
        config={
            "model": {
                "num_layers": int(payload["num_layers"]),
                "hidden_dim": int(payload["hidden_dim"]),
                "dropout": float(payload.get("dropout", 0.0)),
                "num_classes": int(payload["num_classes"]),
            }
        },
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model


def test_indices(graph: Any) -> torch.Tensor:
    return torch.nonzero(graph.test_mask.reshape(-1), as_tuple=False).reshape(-1)


def logits_for(model: Any, graph: Any, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        out = model(
            graph.x.to(device),
            graph.edge_index.to(device),
            graph.edge_attr.to(device),
            edge_batch=indices.to(device),
        )
    return out.detach().cpu()


def decode(
    logits: torch.Tensor,
    *,
    prior: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "fine":
        return logits.argmax(dim=-1).numpy()

    if method == "prior":
        # Post-hoc logit adjustment (Menon et al., 2021): add tau * log(pi_y).
        # tau=1 is the standard balanced-softmax/prior-correction form.
        bias = torch.as_tensor(np.log(np.maximum(prior, 1e-12)), dtype=logits.dtype)
        return (logits + bias).argmax(dim=-1).numpy()

    if method == "hierarchical":
        mapping = fine_to_coarse()
        # Aggregate fine logits into coarse scores with logsumexp.  This keeps
        # all fine alternatives inside a coarse group, then chooses the best
        # fine class only inside the winning group.
        coarse_scores = []
        for members in COARSE_GROUPS:
            coarse_scores.append(torch.logsumexp(logits[:, members], dim=1))
        coarse = torch.stack(coarse_scores, dim=1).argmax(dim=1).numpy()
        predictions = np.empty(len(coarse), dtype=np.int64)
        for row, group in enumerate(coarse):
            members = COARSE_GROUPS[int(group)]
            predictions[row] = int(members[int(logits[row, members].argmax())])
        return predictions

    raise ValueError(f"unknown decoder {method!r}")


def period_prior(graph: Any, n_classes: int) -> np.ndarray:
    train = torch.nonzero(graph.train_mask.reshape(-1), as_tuple=False).reshape(-1)
    counts = np.bincount(graph.y[train].numpy().astype(np.int64), minlength=n_classes)
    counts = np.maximum(counts, 1)
    return counts / counts.sum()


def evaluate_decoders(
    *,
    graphs: list[Any],
    checkpoint_dir: Path,
    methods: list[str],
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    checkpoint = checkpoint_dir / "period_5.pt"
    model = load_checkpoint(checkpoint, device)
    prior = period_prior(graphs[0], len(class_names))
    labels = list(range(len(class_names)))
    out: dict[str, Any] = {"checkpoint": str(checkpoint), "methods": {}}

    for method in methods:
        rows: dict[str, Any] = {}
        for period, graph in enumerate(graphs):
            indices = test_indices(graph).to(device)
            logits = logits_for(model, graph, indices.cpu(), device)
            predicted = decode(logits, prior=prior, method=method)
            truth = graph.y[indices.cpu()].numpy()
            rows[str(period + 1)] = {
                "macro_f1": float(macro_f1(truth, predicted, labels)),
                "per_class_f1": {str(k): float(v) for k, v in per_class_f1(truth, predicted, labels).items()},
                "n": int(len(truth)),
            }
        out["methods"][method] = rows
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="NF-UNSW-NB15-v3")
    parser.add_argument("--graph-mode", default="constant")
    parser.add_argument("--graph-root", type=Path, default=Path("datasets/graphs"))
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(
            "src/kaggle_lane/outputs/xdrift-step06-improve/step06_improve/baseline/"
            "datasets/processed/NF-UNSW-NB15-v3/models"
        ),
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--methods", default="fine,prior,hierarchical")
    parser.add_argument("--out", type=Path, default=Path("results/text/tables/decoder_compare"))
    args = parser.parse_args(argv)

    device = torch.device("cpu")
    spec = load_dataset_spec(args.dataset)
    class_names = list(spec.class_order)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if "hierarchical" in methods and len(class_names) != len(CLASS_NAMES):
        raise ValueError("hierarchical decoding is defined only for the 10-class NF-UNSW taxonomy")
    if "hierarchical" in methods and len(class_names) != 10:
        raise ValueError("hierarchical decoding requires a 10-class dataset")
    graphs = [
        torch.load(args.graph_root / args.dataset / args.graph_mode / f"period_{i}.pt", map_location="cpu", weights_only=False)
        for i in range(1, 6)
    ]
    all_runs: dict[str, Any] = {}
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        run_id = f"ft-none-d2-s{seed}"
        result = evaluate_decoders(
            graphs=graphs,
            checkpoint_dir=args.checkpoint_root / run_id,
            methods=methods,
            device=device,
            class_names=class_names,
        )
        all_runs[run_id] = result

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": args.dataset,
        "graph_mode": args.graph_mode,
        "summary_seed": "mean over seeds; final checkpoint period 5",
        "runs": all_runs,
    }
    (args.out / "decoder_compare.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Report mean final-row (checkpoint 5) metrics per decoder.
    report = ["# Decoder comparison", "", "Final checkpoint (period 5), mean over seeds.", ""]
    report.append("| decoder | " + " | ".join(f"P{i}" for i in range(1, 6)) + " | mean |")
    report.append("|---|" + "---|" * 6)
    for method in methods:
        per_period = []
        for period in range(1, 6):
            values = []
            for run in all_runs.values():
                values.append(run["methods"][method][str(period)]["macro_f1"])
            per_period.append(float(np.mean(values)))
        report.append(f"| {method} | " + " | ".join(f"{v:.4f}" for v in per_period) + f" | {np.mean(per_period):.4f} |")
    report.append("")
    report.append("This is a post-hoc decoding comparison on the same checkpoints, not a new training run.")
    (args.out / "decoder_compare.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
