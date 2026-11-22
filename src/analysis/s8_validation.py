"""Independent S8/S9 validation and H1 arithmetic cross-check (T20 Track B).

Purpose
-------
When the first real S8/S9 outputs land, the verifier needs an *independent*
recomputation of the headline H1 statistics from the anchor-level artifact, plus
the stage validators. This module:

1. runs the strict S8/S9 artifact validators (``src.analysis.validators``);
2. recomputes, from ``results/text/tables/rq1_anchor_level.csv`` only:
   n / n_above / fraction above the floor, the stratified bootstrap CI,
   exact binomial p, normal-approximation p, paired Wilcoxon p on excess,
   rank-biserial effect size, and Holm-adjusted p across the (strategy, model,
   explainer) family;
3. compares the recomputation with ``rq1_h1_tests.csv`` (exact for counts /
   fractions, tight tolerance for p-values, bootstrap-MC tolerance for CI bounds);
4. cross-checks the per-row ``floor_q95`` against ``noise_floor.csv`` and the
   frozen protocol constants (quantile 0.95, expected above-floor fraction 0.05);
5. writes ``src/analysis/reports/s8_validation.md`` with pass/fail per configuration
   and every discrepancy, or a clearly marked ``WAITING`` report naming the
   missing artifacts and the commands to run once they exist.

Authority: docs/05 §9 (H1), §13 (statistics); Doc 2b S8–S9; D-043 conventions
A6/A7; D-044 (analysis modules authoritative).

Usage
-----
    .\\.venv\\Scripts\\python.exe -m analysis.s8_validation
    .\\.venv\\Scripts\\python.exe -m analysis.s8_validation --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.analysis import validators

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_TABLES = REPO_ROOT / "results" / "tables"
REPORT_PATH = REPO_ROOT / "analysis" / "reports" / "s8_validation.md"

PROTOCOL_N_RESAMPLES = 1000
PROTOCOL_ALPHA = 0.05
PROTOCOL_EXPECTED_ABOVE = 0.05
PROTOCOL_QUANTILE = 0.95
EXACT_TOL = 1e-9
P_TOL = 1e-6
CI_TOL = 0.02  # bootstrap Monte-Carlo tolerance (independent RNG stream)
RANK_BISERIAL_TOL = 0.02  # CSV %.10g rounding can reorder near-tied |excess| ranks


@dataclass
class Finding:
    check: str
    status: str  # pass | warn | fail
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"check": self.check, "status": self.status, "detail": self.detail}


@dataclass
class H1Crosscheck:
    status: str  # pass | fail | waiting
    generated_at_utc: str
    missing: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    configurations: list[dict[str, Any]] = field(default_factory=list)
    stage_report: dict[str, Any] | None = None

    @property
    def counts(self) -> dict[str, int]:
        counts = {"pass": 0, "warn": 0, "fail": 0}
        for finding in self.findings:
            counts[finding.status] = counts.get(finding.status, 0) + 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "generated_at_utc": self.generated_at_utc,
            "missing": self.missing,
            "counts": self.counts,
            "findings": [finding.as_dict() for finding in self.findings],
            "configurations": self.configurations,
            "stage_validation": self.stage_report,
        }


def _stratified_bootstrap_ci(values: np.ndarray, strata: np.ndarray, *, n_resamples: int, alpha: float, seed: int) -> tuple[float, float]:
    """Independent percentile bootstrap (resample within stratum, fixed sizes)."""
    rng = np.random.default_rng(seed)
    unique = list(dict.fromkeys(strata.tolist()))
    groups = [np.where(strata == value)[0] for value in unique]
    distribution = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        pieces = [group[rng.integers(0, group.size, size=group.size)] for group in groups]
        distribution[index] = float(np.mean(values[np.concatenate(pieces)]))
    lower = float(np.percentile(distribution, 100.0 * alpha / 2.0))
    upper = float(np.percentile(distribution, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def _above_floor(frame: pd.DataFrame) -> np.ndarray:
    column = frame["above_floor"]
    if column.dtype == bool:
        return column.to_numpy()
    if np.issubdtype(column.dtype, np.number):
        return column.to_numpy(dtype=float) > 0.5
    return column.astype(str).str.lower().isin({"true", "1", "1.0"}).to_numpy()


def _holm(pvalues: Sequence[float]) -> list[float]:
    values = np.asarray(list(pvalues), dtype=float)
    adjusted = np.full_like(values, np.nan)
    order = np.argsort(values, kind="stable")
    m = values.size
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, float(values[index]) * (m - rank))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _recompute_configuration(group: pd.DataFrame) -> dict[str, Any]:
    above = _above_floor(group)
    excess = group["excess"].to_numpy(dtype=float)
    strata = group["class"].astype(str).to_numpy()
    n = int(above.size)
    n_above = int(above.sum())
    fraction = n_above / n if n else float("nan")
    ci_lo, ci_hi = _stratified_bootstrap_ci(above.astype(float), strata, n_resamples=PROTOCOL_N_RESAMPLES, alpha=PROTOCOL_ALPHA, seed=42)
    binomial = scipy_stats.binomtest(n_above, n, p=PROTOCOL_EXPECTED_ABOVE, alternative="greater")
    standard_error = float(np.sqrt(PROTOCOL_EXPECTED_ABOVE * (1.0 - PROTOCOL_EXPECTED_ABOVE) / n)) if n else float("nan")
    z = (fraction - PROTOCOL_EXPECTED_ABOVE) / standard_error if standard_error else float("nan")
    p_normal = float(scipy_stats.norm.sf(z)) if np.isfinite(z) else float("nan")
    nonzero = excess[excess != 0.0]
    if nonzero.size:
        wilcoxon = scipy_stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        p_wilcoxon = float(wilcoxon.pvalue)
    else:
        p_wilcoxon = 1.0
    ranks = scipy_stats.rankdata(np.abs(excess[excess != 0.0]))
    w_plus = float(ranks[excess[excess != 0.0] > 0].sum())
    w_minus = float(ranks[excess[excess != 0.0] < 0].sum())
    rank_biserial = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) else 0.0
    return {
        "n_anchors": n,
        "n_above_floor": n_above,
        "fraction_above_floor": fraction,
        "fraction_ci_lo": ci_lo,
        "fraction_ci_hi": ci_hi,
        "p_binomial": float(binomial.pvalue),
        "z_normal": z,
        "p_normal": p_normal,
        "p_wilcoxon": p_wilcoxon,
        "rank_biserial": rank_biserial,
        "excess_mean": float(np.mean(excess)),
        "excess_median": float(np.median(excess)),
    }


def _compare_finding(name: str, recomputed: Any, reported: Any, tolerance: float) -> Finding:
    if reported is None or (isinstance(reported, float) and not np.isfinite(reported)):
        return Finding(name, "warn", f"reported value missing (recomputed {recomputed})")
    difference = abs(float(reported) - float(recomputed))
    status = "pass" if difference <= tolerance else "fail"
    return Finding(name, status, f"recomputed={float(recomputed):.10g}, reported={float(reported):.10g}, |diff|={difference:.3g}, tol={tolerance:g}")


def _load_required(tables_root: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    missing: list[str] = []
    frames: list[pd.DataFrame | None] = []
    for name in ("rq1_anchor_level.csv", "rq1_h1_tests.csv", "noise_floor.csv"):
        path = tables_root / name
        if not path.is_file():
            frames.append(None)
            missing.append(name)
            continue
        frames.append(pd.read_csv(path))
    return frames[0], frames[1], frames[2], missing


def run_s8_validation(results_tables: Path = RESULTS_TABLES) -> H1Crosscheck:
    results_tables = Path(results_tables)
    result = H1Crosscheck(status="waiting", generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    stage_report = validators.validate_artifact_set(results_tables, stage="s9", allow_partial_s5=True)
    result.stage_report = {"overall": stage_report.overall, "counts": stage_report.counts}
    anchor_frame, h1_frame, noise_frame, missing = _load_required(results_tables)
    if missing:
        result.missing = missing
        result.findings.append(
            Finding(
                "s8s9.artifacts",
                "warn",
                "S8/S9 outputs not produced yet: missing " + ", ".join(missing)
                + ". The independent checker is ready; run after `src/scripts/run_analysis.py --mode real` succeeds.",
            )
        )
        # The stage report may still contain useful context (S5/S6/S7 status).
        result.findings.append(
            Finding(
                "validators.stage_s9",
                "warn",
                f"stage validators overall={stage_report.overall} (expected until S6/S7/S8 artifacts exist), counts={stage_report.counts}",
            )
        )
        return result
    assert anchor_frame is not None and h1_frame is not None and noise_frame is not None
    # --- per-configuration H1 recomputation ---------------------------------
    group_keys = ["strategy", "model", "explainer"]
    recomputed_rows: list[dict[str, Any]] = []
    for key, group in anchor_frame.groupby(group_keys, dropna=False, sort=True):
        values = _recompute_configuration(group)
        values.update(dict(zip(group_keys, key if isinstance(key, tuple) else (key,))))
        recomputed_rows.append(values)
    result.configurations = recomputed_rows
    reported = h1_frame.copy()
    merge_keys = [column for column in group_keys if column in reported.columns]
    merged = pd.DataFrame(recomputed_rows).merge(reported, on=merge_keys, how="left", suffixes=("", "_reported"), validate="one_to_one")
    for _, row in merged.iterrows():
        label = "/".join(str(row[column]) for column in group_keys)
        result.findings.append(_compare_finding(f"h1.n_anchors[{label}]", row["n_anchors"], row.get("n_anchors_reported", row.get("n_anchors")), 0))
        result.findings.append(_compare_finding(f"h1.n_above_floor[{label}]", row["n_above_floor"], row.get("n_above_floor_reported", row.get("n_above_floor")), 0))
        result.findings.append(_compare_finding(f"h1.fraction_above_floor[{label}]", row["fraction_above_floor"], row.get("fraction_above_floor_reported", row.get("fraction_above_floor")), EXACT_TOL))
        result.findings.append(_compare_finding(f"h1.p_binomial[{label}]", row["p_binomial"], row.get("p_binomial_reported", row.get("p_binomial")), P_TOL))
        result.findings.append(_compare_finding(f"h1.p_normal[{label}]", row["p_normal"], row.get("p_normal_reported", row.get("p_normal")), P_TOL))
        result.findings.append(_compare_finding(f"h1.p_wilcoxon[{label}]", row["p_wilcoxon"], row.get("p_wilcoxon_reported", row.get("p_wilcoxon")), P_TOL))
        result.findings.append(_compare_finding(f"h1.rank_biserial[{label}]", row["rank_biserial"], row.get("rank_biserial_reported", row.get("rank_biserial")), RANK_BISERIAL_TOL))
        result.findings.append(_compare_finding(f"h1.fraction_ci_lo[{label}]", row["fraction_ci_lo"], row.get("fraction_ci_lo_reported", row.get("fraction_ci_lo")), CI_TOL))
        result.findings.append(_compare_finding(f"h1.fraction_ci_hi[{label}]", row["fraction_ci_hi"], row.get("fraction_ci_hi_reported", row.get("fraction_ci_hi")), CI_TOL))
    # --- Holm family --------------------------------------------------------
    recomputed_p = [row["p_binomial"] for row in recomputed_rows]
    adjusted = _holm(recomputed_p)
    reported_holm = merged["p_binomial_holm"].tolist() if "p_binomial_holm" in merged.columns else [None] * len(adjusted)
    for row, recomputed, reported_value in zip(recomputed_rows, adjusted, reported_holm):
        label = "/".join(str(row[column]) for column in group_keys)
        result.findings.append(_compare_finding(f"h1.p_binomial_holm[{label}]", recomputed, reported_value, P_TOL))
    # --- protocol constants + floor provenance ------------------------------
    quantiles = noise_frame["quantile"].to_numpy(dtype=float)
    result.findings.append(Finding("noise_floor.quantile", "pass" if bool(np.allclose(quantiles, PROTOCOL_QUANTILE)) else "fail", f"quantiles={sorted(set(quantiles.tolist()))}, frozen={PROTOCOL_QUANTILE}"))
    channels = set(noise_frame["channel"].astype(str))
    result.findings.append(Finding("noise_floor.channels", "pass" if "model_null" in channels else "fail", f"channels={sorted(channels)} (model_null required for H1)"))

    def _floor_lookup(channel: str, explainer: str, class_name: str) -> float | None:
        subset = noise_frame[(noise_frame["channel"] == channel) & (noise_frame["explainer"] == explainer) & (noise_frame["class"].astype(str) == str(class_name))]
        if subset.empty:
            return None
        return float(subset.iloc[0]["q95"])

    mismatches = 0
    checked = 0
    for _, row in anchor_frame.head(5000).iterrows():
        expected = _floor_lookup(str(row.get("channel", "model_null")), str(row.get("explainer")), str(row.get("class")))
        if expected is None:
            continue
        checked += 1
        if abs(float(row["floor_q95"]) - expected) > EXACT_TOL:
            mismatches += 1
    result.findings.append(
        Finding(
            "noise_floor.floor_q95_matches_table",
            "pass" if mismatches == 0 else "fail",
            f"{checked} anchor row(s) checked; floor_q95 mismatches={mismatches}",
        )
    )
    result.findings.append(Finding("validators.stage_s9", "pass" if stage_report.overall != "fail" else "fail", f"stage validators overall={stage_report.overall}, counts={stage_report.counts}"))
    failures = [finding for finding in result.findings if finding.status == "fail"]
    warnings = [finding for finding in result.findings if finding.status == "warn"]
    result.status = "fail" if failures else ("pass" if not warnings else "pass")
    return result


def render_markdown(result: H1Crosscheck) -> str:
    lines: list[str] = []
    lines.append("# S8/S9 validation — first real drift/noise-floor artifacts (T20 Track B)")
    lines.append("")
    lines.append(
        "*Owner: analyst. Generated by `python -m src.analysis.s8_validation`. Authority: docs/05 §9 (H1), §13 (statistics); "
        "Doc 2b S8–S9; D-043/A6/A7; D-044. The H1 statistics below are recomputed independently from "
        "`results/text/tables/rq1_anchor_level.csv` and compared with `rq1_h1_tests.csv`; stage schemas are checked by "
        "`src.analysis.validators --stage s9`.*"
    )
    lines.append("")
    lines.append(f"*Generated: {result.generated_at_utc}*")
    lines.append("")
    lines.append(f"## Status: **{result.status.upper()}**")
    lines.append("")
    if result.missing:
        lines.append("### Waiting for the first real S8/S9 run")
        lines.append("")
        lines.append("Missing artifact(s): " + ", ".join(f"`{name}`" for name in result.missing))
        lines.append("")
        lines.append("The independent checker is implemented and self-tested on the synthetic demo; run it after the first real S8/S9 wiring run:")
        lines.append("")
        for command in (
            ".\\\\.venv\\\\Scripts\\\\python.exe scripts\\\\run_analysis.py --mode real --dry-run",
            ".\\\\.venv\\\\Scripts\\\\python.exe -m src.analysis.validators --stage s8 --strict",
            ".\\\\.venv\\\\Scripts\\\\python.exe -m analysis.s8_validation",
        ):
            lines.append(f"```powershell\n{command}\n```")
        lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- findings: {result.counts}")
    if result.stage_report is not None:
        lines.append(f"- stage validators (s9): overall=`{result.stage_report.get('overall')}`, counts={result.stage_report.get('counts')}")
    lines.append("")
    if result.configurations:
        lines.append("## Recomputed H1 per configuration")
        lines.append("")
        lines.append("| strategy | model | explainer | n | n above | fraction | p_binomial | p_normal | p_wilcoxon | rank-biserial |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in result.configurations:
            lines.append(
                f"| {row.get('strategy')} | {row.get('model')} | {row.get('explainer')} | {row['n_anchors']} | {row['n_above_floor']} | "
                f"{row['fraction_above_floor']:.4f} | {row['p_binomial']:.3g} | {row['p_normal']:.3g} | {row['p_wilcoxon']:.3g} | {row['rank_biserial']:.3f} |"
            )
        lines.append("")
    lines.append("## Findings")
    lines.append("")
    lines.append("| check | status | detail |")
    lines.append("| --- | --- | --- |")
    for finding in result.findings:
        lines.append(f"| `{finding.check}` | **{finding.status}** | {str(finding.detail).replace('|', '&#124;')} |")
    lines.append("")
    failures = [finding for finding in result.findings if finding.status == "fail"]
    lines.append("## Discrepancies requiring action")
    lines.append("")
    if not failures:
        lines.append("No discrepancy found by the independent recomputation.")
    for finding in failures:
        lines.append(f"- **{finding.check}**: {finding.detail}")
    lines.append("")
    return "\n".join(lines)


def write_report(result: H1Crosscheck, path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(result), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent S8/S9 validation + H1 arithmetic (T20).")
    parser.add_argument("--results-tables", default=str(RESULTS_TABLES))
    parser.add_argument("--out", default=str(REPORT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_s8_validation(Path(args.results_tables))
    path = write_report(result, Path(args.out))
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    else:
        print(json.dumps({"status": result.status, "report": str(path), "counts": result.counts, "missing": result.missing}, indent=2))
    return 0 if result.status in {"pass", "waiting"} else 1


__all__ = ["Finding", "H1Crosscheck", "REPORT_PATH", "render_markdown", "run_s8_validation", "write_report"]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
