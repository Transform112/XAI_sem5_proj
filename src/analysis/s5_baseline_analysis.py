"""T46 baseline CL analysis on the valid S5 matrix (D-075).

Inputs are the three validated Kaggle-run directories plus the corrected headline
CSV. Produces the paired seed-wise comparisons required by docs/05 §13:

* paired 1000-resample percentile bootstrap differences (FT vs fixed replay vs
  EWC-λ100), Cohen's d (paired) and matched-pairs rank-biserial effect sizes,
  raw and Holm-adjusted p-values within each (model, metric) strategy family;
* per-period final-row / diagonal / forgetting tables;
* per-class F1 with seed CIs and period support counts, D-033 rare strata;
* an explicitly labelled EWC-λ100 vs λ1000 sensitivity table.

All randomness is seeded (analysis.stats); no test data is used for tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.analysis.stats import (
    DEFAULT_ALPHA,
    DEFAULT_N_RESAMPLES,
    effect_size_summary,
    holm_correction_labelled,
    paired_bootstrap_diff,
)

SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
MODELS: tuple[str, ...] = ("egraphsage", "gcn")
CLASSES: tuple[str, ...] = (
    "Benign", "Analysis", "Backdoor", "DoS", "Exploits",
    "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
)
RARE_CLASSES: tuple[str, ...] = ("Analysis", "Backdoor", "Worms")  # support-limited
PRIMARY_STRATEGIES: tuple[str, ...] = ("ft", "replay", "ewc")
COMPARISONS: tuple[tuple[str, str], ...] = (("replay", "ft"), ("ewc", "ft"), ("ewc", "replay"))

RUN_ROOTS: dict[str, Path] = {
    "ft": Path(r"incoming_rtx5060\kaggle_runs\t37-matrix-full-90e7635"),
    "replay": Path(r"incoming_rtx5060\kaggle_runs\t42-replay-rerun-9ca4681"),
    "ewc": Path(r"incoming_rtx5060\kaggle_runs\t42-ewc-lambda100-649109a"),
    "ewc_lambda1000": Path(r"incoming_rtx5060\kaggle_runs\t37-matrix-full-90e7635"),
}

METRICS: dict[str, tuple[str, str]] = {
    "final_all": ("cl_metrics_all_classes", "final_average_macro_f1"),
    "final_supported": ("cl_metrics_supported_set", "final_average_macro_f1"),
    "forgetting_all": ("cl_metrics_all_classes", "forgetting"),
    "forgetting_supported": ("cl_metrics_supported_set", "forgetting"),
    "bwt_all": ("cl_metrics_all_classes", "bwt"),
    "bwt_supported": ("cl_metrics_supported_set", "bwt"),
}


def group_dir(root: Path, strategy: str, model: str, seed: int) -> Path:
    return root / "unpacked" / "results" / "tables" / "matrix" / f"{strategy}_{model}_seed{seed}"


def load_group(root: Path, strategy: str, model: str, seed: int) -> dict[str, Any]:
    matrix_path = group_dir(root, strategy, model, seed) / "cl_matrix.json"
    per_class_path = group_dir(root, strategy, model, seed) / "cl_per_class_f1.json"
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    if payload.get("checkpoints_evaluated") != [1, 2, 3, 4, 5]:
        raise ValueError(f"incomplete matrix {matrix_path}: {payload.get('checkpoints_evaluated')}")
    evaluation = payload.get("evaluation") or {}
    if evaluation.get("mode") not in ("edge_batch", "chunked"):
        raise ValueError(f"unexpected eval mode in {matrix_path}: {evaluation.get('mode')}")
    per_class = json.loads(per_class_path.read_text(encoding="utf-8"))
    return {"matrix": payload, "per_class": per_class}


def metric_values(groups: Mapping[tuple[str, str, int], Any], strategy: str, model: str, metric: str) -> np.ndarray:
    block, key = METRICS[metric]
    return np.asarray(
        [float(groups[(strategy, model, seed)]["matrix"][block][key]) for seed in SEEDS], dtype=float
    )


def mean_sd(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0


def comparison_rows(
    groups: Mapping[tuple[str, str, int], Any],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for metric in METRICS:
            family: dict[str, float] = {}
            pending: list[dict[str, Any]] = []
            for index, (a, b) in enumerate(COMPARISONS):
                values_a = metric_values(groups, a, model, metric)
                values_b = metric_values(groups, b, model, metric)
                paired = paired_bootstrap_diff(
                    values_a,
                    values_b,
                    n_resamples=int(n_resamples),
                    alpha=float(alpha),
                    seed=int(seed) + index,
                )
                effects = effect_size_summary(values_a, values_b, paired=True)
                label = f"{a}_minus_{b}"
                family[label] = float(paired.p_value)
                pending.append(
                    {
                        "model": model,
                        "metric": metric,
                        "comparison": label,
                        "mean_a": float(np.mean(values_a)),
                        "mean_b": float(np.mean(values_b)),
                        "diff": paired.diff,
                        "ci_lo": paired.lo,
                        "ci_hi": paired.hi,
                        "p_raw": paired.p_value,
                        "cohens_d": effects["cohens_d"],
                        "rank_biserial": effects["rank_biserial"],
                        "n": paired.n,
                        "n_resamples": paired.n_resamples,
                        "family": f"{model}:{metric}",
                    }
                )
            adjusted = holm_correction_labelled(family)
            for row in pending:
                row["p_holm"] = adjusted[row["comparison"]]
                rows.append(row)
    return rows


def mean_table(groups: Mapping[tuple[str, str, int], Any]) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for strategy in PRIMARY_STRATEGIES:
            for metric in METRICS:
                values = metric_values(groups, strategy, model, metric)
                mean, sd = mean_sd(values)
                rows.append(
                    {"model": model, "strategy": strategy, "metric": metric,
                     "mean": mean, "sd": sd, "n": int(values.size),
                     "values": ";".join(f"{value:.6f}" for value in values)}
                )
    return rows


def per_period_rows(groups: Mapping[tuple[str, str, int], Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    finalrow: list[dict[str, Any]] = []
    diagonal: list[dict[str, Any]] = []
    forgetting: list[dict[str, Any]] = []
    for model in MODELS:
        for strategy in PRIMARY_STRATEGIES:
            for name, block in (("all_classes", "cl_matrix"), ("supported_set", "cl_matrix")):
                matrices = []
                forgets = []
                for seed in SEEDS:
                    payload = groups[(strategy, model, seed)]["matrix"]
                    metric_key = "all_classes" if name == "all_classes" else "supported_set"
                    matrices.append(np.asarray(payload["matrix"][metric_key], dtype=float))
                    cl_key = "cl_metrics_all_classes" if name == "all_classes" else "cl_metrics_supported_set"
                    forgets.append(payload[cl_key].get("forgetting_per_period") or {})
                stack = np.stack(matrices)  # seeds x 5 x 5
                for period in range(5):
                    final_values = stack[:, -1, period]
                    diag_values = np.asarray([stack[s, period, period] for s in range(len(SEEDS))])
                    mean, sd = mean_sd(final_values)
                    finalrow.append({"model": model, "strategy": strategy, "matrix": name,
                                     "period": period + 1, "mean": mean, "sd": sd, "n": int(len(SEEDS))})
                    mean_d, sd_d = mean_sd(diag_values)
                    diagonal.append({"model": model, "strategy": strategy, "matrix": name,
                                     "checkpoint": period + 1, "mean": mean_d, "sd": sd_d, "n": int(len(SEEDS))})
                for period in range(1, 5):  # last period has no later forgetting
                    values = []
                    for row in forgets:
                        value = row.get(str(period))
                        if value is not None:
                            values.append(float(value))
                    mean, sd = mean_sd(np.asarray(values))
                    forgetting.append({"model": model, "strategy": strategy, "matrix": name,
                                       "period": period, "mean": mean, "sd": sd, "n": len(values)})
    return finalrow, diagonal, forgetting


def per_class_rows(
    groups: Mapping[tuple[str, str, int], Any],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for strategy in PRIMARY_STRATEGIES:
            # per class/period values across seeds + support counts
            per_seed: dict[tuple[str, int], list[float]] = {}
            supports: dict[tuple[str, int], list[int]] = {}
            rare_pool: dict[int, list[float]] = {}
            rare_support: dict[int, list[int]] = {}
            for s_index, run_seed in enumerate(SEEDS):
                cell = groups[(strategy, model, run_seed)]["per_class"]
                order = list(cell.get("class_order") or CLASSES)
                for row_key in ("5",):
                    for col in range(1, 6):
                        payload = cell["per_class_f1"][row_key][str(col)]
                        per_class = payload["per_class_f1"]
                        counts = payload["dual_macro_f1"]["class_counts"]
                        for class_index, class_name in enumerate(order):
                            value = float(per_class[str(class_index)])
                            key = (class_name, col)
                            per_seed.setdefault(key, []).append(value)
                            supports.setdefault(key, []).append(int(counts[str(class_index)]))
                        rare_values = [float(per_class[str(order.index(name))]) for name in RARE_CLASSES]
                        rare_pool.setdefault(col, []).append(float(np.mean(rare_values)))
                        rare_support.setdefault(col, []).append(
                            int(sum(int(counts[str(order.index(name))]) for name in RARE_CLASSES))
                        )
            for class_name in CLASSES:
                for col in range(1, 6):
                    values = np.asarray(per_seed[(class_name, col)], dtype=float)
                    support = np.asarray(supports[(class_name, col)], dtype=int)
                    boot = __import__("analysis.stats", fromlist=["bootstrap_ci"]).bootstrap_ci(
                        values, n_resamples=int(n_resamples), alpha=float(alpha), seed=int(seed) + col
                    )
                    mean, sd = mean_sd(values)
                    rows.append(
                        {
                            "model": model,
                            "strategy": strategy,
                            "class": class_name,
                            "evaluated_period": col,
                            "mean_f1": mean,
                            "sd": sd,
                            "ci_lo": boot.lo,
                            "ci_hi": boot.hi,
                            "n": int(values.size),
                            "support_min": int(support.min()),
                            "support_max": int(support.max()),
                            "support_p5": int(support[-1]),
                            "rare_class": bool(support.min() < 10),
                            "rare_stratum_member": class_name in RARE_CLASSES,
                        }
                    )
            for col in range(1, 6):
                values = np.asarray(rare_pool[col], dtype=float)
                support = np.asarray(rare_support[col], dtype=int)
                boot = __import__("analysis.stats", fromlist=["bootstrap_ci"]).bootstrap_ci(
                    values, n_resamples=int(n_resamples), alpha=float(alpha), seed=int(seed) + 100 + col
                )
                mean, sd = mean_sd(values)
                rows.append(
                    {
                        "model": model,
                        "strategy": strategy,
                        "class": "rare_stratum_pooled",
                        "evaluated_period": col,
                        "mean_f1": mean,
                        "sd": sd,
                        "ci_lo": boot.lo,
                        "ci_hi": boot.hi,
                        "n": int(values.size),
                        "support_min": int(support.min()),
                        "support_max": int(support.max()),
                        "support_p5": int(support[-1]),
                        "rare_class": True,
                        "rare_stratum_member": True,
                    }
                )
    return rows


def ewc_lambda_sensitivity_rows(groups: Mapping[tuple[str, str, int], Any]) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for metric in ("final_all", "final_supported", "forgetting_all", "forgetting_supported"):
            lam100 = metric_values(groups, "ewc", model, metric)
            lam1000 = metric_values(groups, "ewc_lambda1000", model, metric)
            diff = paired_bootstrap_diff(lam100, lam1000, n_resamples=1000, alpha=DEFAULT_ALPHA, seed=42)
            effects = effect_size_summary(lam100, lam1000, paired=True)
            mean100, sd100 = mean_sd(lam100)
            mean1000, sd1000 = mean_sd(lam1000)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "lambda100_mean": mean100,
                    "lambda100_sd": sd100,
                    "lambda1000_mean": mean1000,
                    "lambda1000_sd": sd1000,
                    "diff_100_minus_1000": diff.diff,
                    "ci_lo": diff.lo,
                    "ci_hi": diff.hi,
                    "p_raw": diff.p_value,
                    "cohens_d": effects["cohens_d"],
                    "rank_biserial": effects["rank_biserial"],
                    "n": diff.n,
                    "note": "sensitivity only; lambda1000 rows are the old protocol-deviation run, not primary",
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def headline_crosscheck(rows: Sequence[Mapping[str, Any]], headline_path: Path) -> dict[str, Any]:
    headline = {}
    with headline_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["strategy"], row["model"], row["metric"])
            headline[key] = row
    checks = []
    for row in rows:
        strategy = row["strategy"]
        if strategy not in PRIMARY_STRATEGIES:
            continue
        metric_map = {"final_all": "all_classes", "final_supported": "supported_set"}
        metric = metric_map.get(row["metric"])
        if metric is None:
            continue
        key = (strategy, row["model"], metric)
        expected = headline.get(key)
        if expected is None:
            continue
        checks.append(
            {
                "strategy": strategy,
                "model": row["model"],
                "metric": metric,
                "computed_mean": row["mean"],
                "headline_mean": float(expected["final_mean"]),
                "abs_diff": abs(float(row["mean"]) - float(expected["final_mean"])),
            }
        )
    max_diff = max((item["abs_diff"] for item in checks), default=0.0)
    return {"headline_rows_checked": checks, "max_abs_mean_diff": max_diff}


def run_analysis(
    *,
    out_dir: Path,
    headline_path: Path,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], Any] = {}
    for strategy_root in RUN_ROOTS:
        strategy = "ewc" if strategy_root == "ewc_lambda1000" else strategy_root
        if strategy not in PRIMARY_STRATEGIES and strategy_root != "ewc_lambda1000":
            continue
        for model in MODELS:
            for run_seed in SEEDS:
                # only load the primary λ100 EWC from the λ100 run; λ1000 separately
                if strategy_root == "ewc_lambda1000":
                    load_strategy = "ewc"
                else:
                    load_strategy = strategy
                groups[(strategy_root, model, run_seed)] = load_group(
                    RUN_ROOTS[strategy_root], load_strategy, model, run_seed
                )
    # build primary strategy-keyed view (replay/ft/ewc) from their own roots
    primary = {
        (strategy, model, seed): groups[(strategy, model, seed)]
        for strategy in PRIMARY_STRATEGIES
        for model in MODELS
        for seed in SEEDS
    }
    # λ1000 sensitivity view is keyed separately
    lambda1000 = {
        ("ewc_lambda1000", model, seed): groups[("ewc_lambda1000", model, seed)]
        for model in MODELS for seed in SEEDS
    }
    combined = dict(primary)
    combined.update(lambda1000)
    out_dir.mkdir(parents=True, exist_ok=True)
    means = mean_table(primary)
    comparisons = comparison_rows(primary, n_resamples=n_resamples, alpha=alpha, seed=seed)
    finalrow, diagonal, forgetting = per_period_rows(primary)
    per_class = per_class_rows(primary, n_resamples=n_resamples, alpha=alpha, seed=seed)
    sensitivity = ewc_lambda_sensitivity_rows({**primary, **lambda1000})
    _write_csv(out_dir / "baseline_means.csv", means)
    _write_csv(out_dir / "baseline_comparisons.csv", comparisons)
    _write_csv(out_dir / "per_period_finalrow.csv", finalrow)
    _write_csv(out_dir / "per_period_diagonal.csv", diagonal)
    _write_csv(out_dir / "per_period_forgetting.csv", forgetting)
    _write_csv(out_dir / "per_class_f1.csv", per_class)
    _write_csv(out_dir / "ewc_lambda_sensitivity.csv", sensitivity)
    checks = headline_crosscheck(means, headline_path)
    (out_dir / "checks.json").write_text(
        json.dumps(
            {
                "n_resamples": n_resamples,
                "alpha": alpha,
                "seed": seed,
                "n_groups_primary": len(primary),
                "n_groups_lambda1000": len(lambda1000),
                "headline_crosscheck": checks,
                "sources": {key: str(value) for key, value in RUN_ROOTS.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"means": means, "comparisons": comparisons, "per_class": per_class, "sensitivity": sensitivity,
            "checks": checks, "out_dir": str(out_dir)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="s5_baseline_analysis", description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="results/text/tables/s5_baseline")
    parser.add_argument(
        "--headline",
        default=r"incoming_rtx5060\kaggle_runs\t42_headline\headline_table_valid.csv",
    )
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    result = run_analysis(
        out_dir=Path(args.out_dir),
        headline_path=Path(args.headline),
        n_resamples=int(args.n_resamples),
        alpha=float(args.alpha),
        seed=int(args.seed),
    )
    print(json.dumps({"out_dir": result["out_dir"], "checks": result["checks"], "n_comparisons": len(result["comparisons"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
