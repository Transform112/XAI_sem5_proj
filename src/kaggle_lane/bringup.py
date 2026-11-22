"""Kaggle T4 canonical-lane bring-up driver (D-065 / T35).

One command to push the bring-up kernel, wait for the T4 run, download its output,
verify the R matrix and write the local bring-up report.

Prerequisites (already done by the analyst):
  * private dataset ``mysteriousavailable/xdrift-nf-unsw-c1-inputs`` exists
    (raw CSV + frozen C1 artifacts + code snapshot for the refreshed HEAD;
    current fixed-code snapshot ``xdrift_repo_90e7635.zip`` / dataset v3);
  * token at ``~/.kaggle/kgat.txt`` (or ``KAGGLE_API_TOKEN``).

Usage:
    .\\.venv\\Scripts\\python.exe kaggle_lane\\bringup.py            # push + wait + download + report
    .\\.venv\\Scripts\\python.exe kaggle_lane\\bringup.py --no-fetch  # push only, poll later
    .\\.venv\\Scripts\\python.exe kaggle_lane\\bringup.py --timeout-s 21600

The kernel itself is ``src/kaggle_lane/kernels/xdrift_bringup/script.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from run_kernel import OUTPUTS_DIR, push_and_wait

RUNS_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUNS_DIR.parent
KERNEL_DIR = RUNS_DIR / "kernels" / "xdrift_bringup"
KERNEL_SCRIPT = KERNEL_DIR / "script.py"
KERNEL_SLUG = "xdrift-t4-bring-up-canonical-lane-d-065"
DATASET_REF = "mysteriousavailable/xdrift-nf-unsw-c1-inputs"
LOCAL_ROOT = RUNS_DIR / "outputs" / "kaggle_bringup"
KERNEL_OUTPUT_LOCAL = LOCAL_ROOT / "kernel_output"
UNPACKED_DIR = LOCAL_ROOT / "unpacked_output"
REPORT_PATH = LOCAL_ROOT / "BringupReport.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def copy_output(output_dir: Path) -> list[str]:
    if not output_dir.is_dir():
        return []
    KERNEL_OUTPUT_LOCAL.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        target = KERNEL_OUTPUT_LOCAL / path.relative_to(output_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(str(target))
    return copied


def unpack_output_zip() -> dict | None:
    candidates = sorted(KERNEL_OUTPUT_LOCAL.glob("xdrift_bringup_output.zip"))
    if not candidates:
        return None
    if UNPACKED_DIR.exists():
        shutil.rmtree(UNPACKED_DIR)
    UNPACKED_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(candidates[0]) as archive:
        archive.extractall(UNPACKED_DIR)
    summary_path = UNPACKED_DIR / "bringup_summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return None


def write_report(summary: dict | None, *, kernel_summary: dict, copied: list[str]) -> Path:
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    status = (summary or {}).get("status") or kernel_summary.get("status")
    lines = [
        "# Kaggle T4 bring-up report (canonical lane D-065 / T35)",
        "",
        f"*Generated {utc_now()} by `src/kaggle_lane/bringup.py`.*",
        "",
        "## Run",
        "",
        f"- kernel ref: `{kernel_summary.get('ref')}`",
        f"- dataset ref: `{DATASET_REF}`",
        f"- kernel status: **{kernel_summary.get('status')}**",
        f"- GPU enabled: {kernel_summary.get('gpu')} · internet: {kernel_summary.get('internet')}",
        f"- kernel dir: `{kernel_summary.get('kernel_dir')}`",
        f"- output dir: `{kernel_summary.get('output_dir')}`",
        f"- local copy: `{LOCAL_ROOT}` ({len(copied)} file(s))",
        "",
    ]
    if summary:
        env = summary.get("environment", {})
        verification = summary.get("input_verification", {})
        frozen = summary.get("frozen", {})
        matrix = summary.get("matrix", {})
        evaluation = summary.get("eval", {})
        artifacts = dict(summary.get("artifacts") or {})
        if not artifacts:
            local_zip = KERNEL_OUTPUT_LOCAL / "xdrift_bringup_output.zip"
            if local_zip.is_file():
                import hashlib as _hashlib

                artifacts = {
                    "path": str(local_zip),
                    "bytes": local_zip.stat().st_size,
                    "sha256": _hashlib.sha256(local_zip.read_bytes()).hexdigest(),
                }
        code_label = verification.get("code_name") or verification.get("code_zip")
        lines += [
            "## Result",
            "",
            f"- inner status: **{summary.get('status')}** · OOM: {summary.get('oom')} · wall clock: {summary.get('wall_clock_s')} s",
            f"- error: {summary.get('error', 'none')}",
            f"- python {env.get('python')} · torch {env.get('torch')} · CUDA build {env.get('torch_cuda_build')} · PyG {env.get('pyg')}",
            f"- GPU: {env.get('device')} · capability {env.get('capability')} · {env.get('total_memory_gb')} GB",
            f"- inputs: CSV {verification.get('csv_bytes')} B sha256 `{verification.get('csv_sha256')}` verified={verification.get('csv_verified')}",
            f"- code snapshot `{code_label}` (delivery={verification.get('code_delivery')}, {verification.get('code_files')} files, tree sha256 `{verification.get('code_tree_sha256')}`, commit {verification.get('code_snapshot_sha')}; WIP files excluded: {verification.get('dirty_worktree_files_not_in_snapshot')})",
            f"- frozen artifacts verified: {frozen.get('files_verified')}/{frozen.get('files_expected')} · verified={frozen.get('verified')} (delivery={verification.get('frozen_delivery')})",
            f"- R matrix: complete={evaluation.get('matrix_complete')} cells={evaluation.get('cell_defined')}/{evaluation.get('cell_total')} dual_macro_ok={evaluation.get('dual_macro_ok')} eval_seconds={evaluation.get('eval_duration_s')}",
            f"- artifacts zip: `{artifacts.get('path')}` ({artifacts.get('bytes')} B, sha256 `{artifacts.get('sha256')}`)",
            "",
            "## Per-step timings",
            "",
            "| step | rc | duration (s) | log |",
            "| --- | --- | --- | --- |",
        ]
        for record in summary.get("per_step_logs", []):
            lines.append(f"| {record.get('step')} | {record.get('returncode')} | {record.get('duration_s')} | `{record.get('log')}` |")
        lines += ["", "## R matrix summary", "", "```json", json.dumps(evaluation.get("cl_metrics_all_classes"), indent=2, sort_keys=True) if evaluation.get("cl_metrics_all_classes") else "null", "```", ""]
        lines += ["## Matrix group status", "", "```json", json.dumps((matrix.get("matrix_state") or {}).get("groups"), indent=2, sort_keys=True, default=str)[:4000], "```", ""]
    else:
        lines += ["## Result", "", "No inner bring-up summary was found in the downloaded output (see kernel log).", ""]
    lines += [
        "## Reproduce",
        "",
        "```powershell",
        "$env:KAGGLE_API_TOKEN = (Get-Content \"$env:USERPROFILE\\.kaggle\\kgat.txt\" -Raw).Trim()",
        ".venv\\Scripts\\python.exe kaggle_lane\\bringup.py",
        "```",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-s", type=int, default=6 * 3600)
    parser.add_argument("--poll-s", type=int, default=30)
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args(argv)
    if not KERNEL_SCRIPT.is_file():
        raise SystemExit(f"kernel script missing: {KERNEL_SCRIPT}")
    started = time.time()
    kernel_summary = push_and_wait(
        slug=KERNEL_SLUG,
        code_path=KERNEL_SCRIPT,
        gpu=True,
        internet=True,
        title="xdrift T4 bring-up (canonical lane D-065)",
        dataset_sources=[DATASET_REF],
        timeout_s=args.timeout_s,
        poll_s=args.poll_s,
        fetch_output=not args.no_fetch,
    )
    copied = copy_output(Path(kernel_summary.get("output_dir", OUTPUTS_DIR / KERNEL_SLUG)))
    inner = unpack_output_zip() if not args.no_fetch else None
    report = write_report(inner, kernel_summary=kernel_summary, copied=copied)
    print(f"[bringup] local report: {report}")
    print(f"[bringup] kernel wall clock measured locally: {round(time.time() - started, 1)} s")
    ok = kernel_summary.get("status") == "complete" and (inner or {}).get("status") == "success"
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
