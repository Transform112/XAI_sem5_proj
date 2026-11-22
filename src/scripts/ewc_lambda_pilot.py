"""Pre-registered EWC lambda pilot (D-072/T41).

For seed 42 and both models, train EWC with lambda in {100, 1000, 10000} on the
canonical 5-period stream and record **old-period validation** macro-F1 only
(no test data used for selection). Selection rule (pre-registered): per model,
choose lambda with the highest mean old-period validation macro-F1 over the
periods j=2..5; ties -> smaller lambda. Short pilot budget: max_epochs=10 (the
matrix uses the frozen 30/patience-5 schedule; this is a selection pilot only).

No Kaggle pushes; legacy/CPU/GPU diagnostic. Writes JSON/CSV under
``results/text/verification/data_audit/ewc_lambda_pilot/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.cl.strategies import EWCStrategy  # noqa: E402
from src.eval.cl_metrics import dual_macro_f1, supported_ids_from_names  # noqa: E402
from src.eval.predict import predict_logits  # noqa: E402
from src.models.edge_sage import build_model  # noqa: E402
from src.train.trainer import TrainSettings  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def _section(config, name):
    value = dict(config.get(name) or {})
    return value


def _evaluate_old_validation(model, graphs, period, *, device, class_ids, supported_ids):
    """Macro-F1 on validation masks of periods < current period (no test)."""
    scores = []
    model.eval()
    for old_period in range(1, period):
        graph = graphs[old_period]
        val_mask = getattr(graph, "val_mask")
        indices = torch.nonzero(val_mask, as_tuple=False).flatten()
        logits = predict_logits(
            model,
            graph.x.to(device),
            graph.edge_index.to(device),
            graph.edge_attr.to(device),
            indices,
            device=device,
        )
        y_true = graph.y[val_mask].cpu().numpy()
        y_pred = logits.argmax(dim=-1).cpu().numpy()
        dual = dual_macro_f1(y_true, y_pred, class_ids=class_ids, supported_ids=supported_ids)
        scores.append(float(dual["macro_f1_all_classes"]))
    return float(np.mean(scores)) if scores else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ewc_lambda_pilot", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="NF-UNSW-NB15-v3")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--models", default="egraphsage,gcn")
    parser.add_argument("--lambdas", default="100,1000,10000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lane", default=None)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--out", default="results/text/verification/data_audit/ewc_lambda_pilot")
    args = parser.parse_args(argv)

    paths = load_paths(lane=args.lane)
    train_config = load_config(args.config, project_root=paths.project_root, lane=args.lane)
    train_dict = train_config.to_dict()
    model_cfg = _section(train_dict, "model")
    from src.data.registry import load_dataset_spec

    spec = load_dataset_spec(args.dataset, project_root=paths.project_root)
    class_order = list(spec.class_order or [])
    class_ids = list(range(10))
    supported_ids = supported_ids_from_names(
        ["Benign", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Shellcode"],
        class_order,
    )
    graphs_dir = Path(paths.processed_dataset(args.dataset))
    graphs = {
        period: torch.load(graphs_dir / f"period_{period}.pt", weights_only=False)
        for period in range(1, 6)
    }
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    lambdas = [float(value) for value in args.lambdas.split(",") if value.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    selection = {}
    for model_name in models:
        model_scores: dict[float, list[float]] = {}
        for lam in lambdas:
            seed_everything(args.seed)
            model = build_model(
                model_name,
                node_dim=int(graphs[1].x.shape[1]),
                edge_dim=int(graphs[1].edge_attr.shape[1]),
                config={"model": model_cfg},
            )
            strategy = EWCStrategy(
                fisher_edges=2000,
                lam=float(lam),
                accumulate="sum",
                lambda_candidates=tuple(lambdas),
            )
            settings = TrainSettings.from_config(train_dict)
            settings.max_epochs = int(args.max_epochs)
            settings.patience = int(settings.patience)
            epoch_rows = []
            for period in range(1, 6):
                strategy.update(
                    model,
                    graphs[period],
                    settings=settings,
                    seed=int(args.seed),
                    device=device,
                )
                old_val = _evaluate_old_validation(
                    model,
                    graphs,
                    period,
                    device=device,
                    class_ids=class_ids,
                    supported_ids=supported_ids,
                )
                epoch_rows.append({"period": period, "old_val_macro_f1": old_val})
                rows.append(
                    {
                        "model": model_name,
                        "lambda": float(lam),
                        "seed": int(args.seed),
                        "period": period,
                        "old_val_macro_f1": old_val,
                        "max_epochs": int(args.max_epochs),
                    }
                )
                print(
                    f"[pilot] model={model_name} lambda={lam:g} period={period} "
                    f"old_val_macro_f1={None if old_val is None else round(old_val, 4)}",
                    flush=True,
                )
            model_scores[float(lam)] = [row["old_val_macro_f1"] for row in epoch_rows]
        # pre-registered selection: mean over periods 2..5 (old-period validation only)
        means = {
            lam: float(np.mean([value for value in scores[1:] if value is not None]))
            for lam, scores in model_scores.items()
        }
        best = max(means, key=lambda lam: (means[lam], -lam))
        selection[model_name] = {
            "mean_old_val_by_lambda": {str(lam): means[lam] for lam in sorted(means)},
            "selected_lambda": float(best),
            "selection_rule": "max mean old-period validation macro-F1 over periods 2..5; ties -> smaller lambda",
            "test_data_used": False,
            "pilot_max_epochs": int(args.max_epochs),
        }
        print(f"[pilot] SELECTED {model_name}: lambda={best:g} means={ {str(k): round(v,4) for k,v in means.items()} }", flush=True)

    with (out_dir / "pilot_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "selection.json").write_text(json.dumps({"lambdas": lambdas, "seed": args.seed, "selection": selection}, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
