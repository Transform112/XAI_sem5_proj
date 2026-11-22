"""Bounded EAR-A vs signed FT/replay comparison (T57).

Reads per-group S5 ``cl_matrix.json`` files and produces paired seed-wise
comparisons for macro-F1 (all/supported), forgetting and BWT. Drift is only
included if an EAR S7 explanation stage was actually generated; otherwise the
script reports ``drift_not_available`` and makes no drift claim.
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

from src.analysis import metrics_cl as mc
from src.analysis import stats as st

METRICS: dict[str, tuple[str, str]] = {
    "final_all": ("cl_metrics_all_classes", "final_average_macro_f1"),
    "final_supported": ("cl_metrics_supported_set", "final_average_macro_f1"),
    "forgetting_all": ("cl_metrics_all_classes", "forgetting"),
    "forgetting_supported": ("cl_metrics_supported_set", "forgetting"),
    "bwt_all": ("cl_metrics_all_classes", "bwt"),
    "bwt_supported": ("cl_metrics_supported_set", "bwt"),
}
COMPARISONS: tuple[tuple[str, str], ...] = (("ear_a", "replay"), ("ear_a", "ft"), ("replay", "ft"))


def group_dir(root: Path, strategy: str, model: str, seed: int) -> Path:
    return Path(root) / "results" / "tables" / "matrix" / f"{strategy}_{model}_seed{seed}"


def load_tensor(root: Path, strategy: str, model: str, seed: int) -> mc.RMetricTensor | None:
    path = group_dir(root, strategy, model, seed) / "cl_matrix.json"
    if not path.is_file():
        return None
    return mc.load_cl_matrix_json(path)


def value(tensor: mc.RMetricTensor, metric: str) -> float:
    block, key = METRICS[metric]
    payload = getattr(tensor, "R", None)
    # CL metrics are recomputed from the tensor API; the writer payload is not
    # retained, so use the same definitions as analysis.metrics_cl.
    if metric == "final_all":
        return float(tensor.final_macro_f1())
    if metric == "final_supported":
        raise ValueError("supported metric needs explicit supported class set")
    if metric == "forgetting_all":
        return float(tensor.forgetting())
    if metric == "forgetting_supported":
        raise ValueError("supported metric needs explicit supported class set")
    if metric == "bwt_all":
        return float(tensor.bwt())
    raise ValueError("supported metric needs explicit supported class set")


def _values(
    tensors: Mapping[tuple[str, str, int], mc.RMetricTensor | None],
    strategy: str,
    model: str,
    metric: str,
    *,
    supported_classes: Sequence[str],
) -> np.ndarray:
    values: list[float] = []
    for seed in (42, 43, 44, 45, 46):
        tensor = tensors.get((strategy, model, seed))
        if tensor is None:
            values.append(np.nan)
            continue
        excluded = [c for c in tensor.classes if c not in supported_classes]
        if metric == "final_all":
            values.append(float(tensor.final_macro_f1()))
        elif metric == "final_supported":
            values.append(float(tensor.final_macro_f1_by_class(exclude_classes=excluded)))
        elif metric in ("forgetting_supported", "bwt_supported"):
            supported_tensor = mc.RMetricTensor(
                tensor.class_macro_matrix(exclude_classes=excluded),
                strategy=tensor.strategy,
                model=tensor.model,
                seed=tensor.seed,
            )
            values.append(
                float(supported_tensor.forgetting() if metric == "forgetting_supported" else supported_tensor.bwt())
            )
        elif metric == "forgetting_all":
            values.append(float(tensor.forgetting()))
        elif metric == "bwt_all":
            values.append(float(tensor.bwt()))
    return np.asarray(values, dtype=float)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_ear_comparison", description=__doc__.splitlines()[0])
    parser.add_argument("--ear-root", required=True, help="unpacked EAR run root containing results/text/tables/matrix")
    parser.add_argument("--baseline-root", required=True, help="signed matrix root (FT/replay/EWC)")
    parser.add_argument("--models", default="egraphsage,gcn")
    parser.add_argument("--out", default="results/text/tables/s12_ear")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    supported = (
        "Benign", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Shellcode",
    )
    tensors: dict[tuple[str, str, int], mc.RMetricTensor | None] = {}
    for strategy in ("ft", "replay", "ear_a"):
        root = Path(args.ear_root) if strategy == "ear_a" else Path(args.baseline_root)
        for model in models:
            for seed in (42, 43, 44, 45, 46):
                tensors[(strategy, model, seed)] = load_tensor(root, strategy, model, seed)
    rows: list[dict[str, Any]] = []
    for model in models:
        for metric in METRICS:
            for a, b in COMPARISONS:
                values_a = _values(tensors, a, model, metric, supported_classes=supported)
                values_b = _values(tensors, b, model, metric, supported_classes=supported)
                mask = np.isfinite(values_a) & np.isfinite(values_b)
                if mask.sum() < 2:
                    rows.append(
                        {
                            "model": model,
                            "metric": metric,
                            "comparison": f"{a}_minus_{b}",
                            "n": int(mask.sum()),
                            "status": "insufficient_pairs",
                        }
                    )
                    continue
                paired = st.paired_bootstrap_diff(
                    values_a[mask], values_b[mask],
                    n_resamples=int(args.n_resamples), alpha=0.05, seed=int(args.seed),
                )
                effect = st.effect_size_summary(values_a[mask], values_b[mask], paired=True)
                rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "comparison": f"{a}_minus_{b}",
                        "n": int(mask.sum()),
                        "mean_a": float(np.mean(values_a[mask])),
                        "mean_b": float(np.mean(values_b[mask])),
                        "diff": paired.diff,
                        "ci_lo": paired.lo,
                        "ci_hi": paired.hi,
                        "p_raw": paired.p_value,
                        "cohens_d": effect["cohens_d"],
                        "rank_biserial": effect["rank_biserial"],
                        "status": "ok",
                    }
                )
    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "comparison.csv", index=False)
    summary = {
        "status": "ok" if (frame.get("status") == "ok").any() else "no_complete_pairs",
        "models": models,
        "rows": int(len(frame)),
        "drift": "drift_not_available (no signed EAR S7 explanation run in this bounded stage)",
        "metrics": list(METRICS),
        "note": "EAR-A bounded run; compare with signed FT/replay S5. Drift claims require a separate signed EAR S7 explanation stage.",
    }
    (out / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
