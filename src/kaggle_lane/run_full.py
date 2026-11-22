"""Kaggle full-run driver (T37, D-065/D-066): matrix + explanation stages.

Builds/pushes separate T4 kernels for the headline stages, polls them, downloads and
merges artifacts, feeds a resumable state dataset back in on later pushes, and runs
the local strict validators on the downloaded evidence.

Usage (repo root, project venv; token at ~/.kaggle/kgat.txt):

    # 1) one-group matrix smoke (ft/egraphsage/seed42)
    .venv\\Scripts\\python.exe kaggle_lane\\run_full.py --stage matrix --smoke --run-id t37-matrix-smoke

    # 2) 10-anchor S6/S7 explanation smoke (needs the matrix smoke checkpoints = resume state)
    .venv\\Scripts\\python.exe kaggle_lane\\run_full.py --stage explain --smoke --run-id t37-explain-smoke

    # 3) later: full matrix in chunks (Lead green-light only), then full explain slices
    .venv\\Scripts\\python.exe kaggle_lane\\run_full.py --stage matrix --chunk-size 10 --max-pushes 3 --run-id t37-matrix-full
    .venv\\Scripts\\python.exe kaggle_lane\\run_full.py --stage explain --max-jobs 30 --run-id t37-explain-full

Dataset/model configuration flows through a tiny private ``xdrift-runctl`` dataset
(``runctl.json``); the inputs dataset is the T35 carrier (raw CSV + frozen C1 + code
snapshot). Resume artifacts flow through a private ``xdrift-resume`` dataset
(previous ``results/`` + ``runs/`` trees), so matrix/explain runs survive the 12 h cap.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KAGGLE_LANE = Path(__file__).resolve().parent
REPO_ROOT = KAGGLE_LANE.parents[1]
sys.path.insert(0, str(KAGGLE_LANE))

from run_kernel import build_env, discover_username, kaggle_bin, push_and_wait, run_cli  # noqa: E402

INPUTS_DATASET = "mysteriousavailable/xdrift-nf-unsw-c1-inputs"
RUNCTL_DATASET = "mysteriousavailable/xdrift-runctl"
RESUME_DATASET = "mysteriousavailable/xdrift-resume"
LOCAL_ROOT = KAGGLE_LANE / "outputs" / "kaggle_runs"
RUNCTL_STAGE = LOCAL_ROOT / "runctl_stage"
RESUME_STAGE = LOCAL_ROOT / "resume_stage"
RESUME_STATE = LOCAL_ROOT / "resume_state"
KERNEL_SPECS = {
    "matrix": {"dir": KAGGLE_LANE / "kernels" / "xdrift_matrix", "slug": "xdrift-matrix", "title": "xdrift matrix", "template": "matrix"},
    "explain": {"dir": KAGGLE_LANE / "kernels" / "xdrift_explain", "slug": "xdrift-explain", "title": "xdrift explain", "template": "explain"},
    "noise": {"dir": KAGGLE_LANE / "kernels" / "xdrift_noise", "slug": "xdrift-noise", "title": "xdrift noise", "template": "noise"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_kaggle(args: list[str], env: dict[str, str], *, timeout: int = 600, allow_failure: bool = False) -> subprocess.CompletedProcess:
    result = run_cli(args, env, timeout=timeout)
    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(f"kaggle {' '.join(args)} failed: {(result.stderr or result.stdout)[-1500:]}")
    return result


def dataset_version(ref: str, env: dict[str, str]) -> int | None:
    try:
        import requests

        token = env.get("KAGGLE_API_TOKEN") or env.get("KAGGLE_KEY")
        response = requests.get(f"https://www.kaggle.com/api/v1/datasets/view/{ref}", headers={"Authorization": f"Bearer {token}"}, timeout=60)
        if response.ok:
            payload = response.json()
            return payload.get("currentVersionNumber") or payload.get("versionNumber") or payload.get("latestVersionNumber")
    except Exception:
        return None
    return None


def wait_for_dataset_version(ref: str, expected: int | None, env: dict[str, str], *, timeout_s: int = 900) -> int | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        current = dataset_version(ref, env)
        if expected is None or (current is not None and int(current) >= int(expected)):
            return current
        print(f"[run_full] waiting for {ref} version {expected} (current {current}) ...")
        time.sleep(20)
    raise RuntimeError(f"dataset {ref} did not reach version {expected} within {timeout_s}s")


def ensure_dataset(stage_dir: Path, ref: str, metadata: dict[str, Any], message: str, env: dict[str, str]) -> int | None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    previous = dataset_version(ref, env)
    if previous is None:
        command = ["datasets", "create", "-p", str(stage_dir), "-t", "-r", "skip"]
    else:
        command = ["datasets", "version", "-p", str(stage_dir), "-t", "-r", "skip", "-m", message]
    result = run_kaggle(command, env, timeout=1200, allow_failure=True)
    text = ((result.stdout or "") + (result.stderr or "")).strip()
    print(f"[run_full] {ref}: {(text.splitlines()[-1][:220] if text else 'ok')} (rc={result.returncode})")
    deadline = time.time() + 900
    while time.time() < deadline:
        current = dataset_version(ref, env)
        if current is not None and (previous is None or int(current) > int(previous)):
            print(f"[run_full] {ref} version {current} ready")
            return int(current)
        time.sleep(15)
    return dataset_version(ref, env)


def build_kernel_script(stage: str, ref: str) -> Path:
    spec = KERNEL_SPECS[stage]
    template = (KAGGLE_LANE / "kernels" / "_template.py").read_text(encoding="utf-8")
    script = template.replace("__STAGE__", stage).replace("__KERNEL_REF__", ref)
    spec["dir"].mkdir(parents=True, exist_ok=True)
    path = spec["dir"] / "script.py"
    path.write_text(script, encoding="utf-8")
    metadata = {
        "id": ref,
        "title": spec["title"],
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [INPUTS_DATASET, RUNCTL_DATASET],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (spec["dir"] / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def default_runctl(stage: str, args: argparse.Namespace) -> dict[str, Any]:
    matrix: dict[str, Any] = {
        "models": ["egraphsage", "gcn"],
        "strategies": ["ft", "replay", "ewc"],
        "seeds": [42, 43, 44, 45, 46],
        "periods": 5,
        "include_ear": False,
        "epoch_seconds_full_size": 2.402,
        "max_runs": None,
        "rebuild_graphs": False,
        "config": "configs/train.yaml",
    }
    explain: dict[str, Any] = {
        "mode": "cross_checkpoint",
        "groups": [["ft", "replay", "ewc"], ["egraphsage", "gcn"], [42]],
        "anchor_reference": {"strategy": "ft", "model": "egraphsage", "seed": 42},
        "periods": 5,
        "checkpoint_periods": [1, 2, 3, 4, 5],
        "horizon": "all",
        "explainers": ["ig", "gnn_explainer"],
        "k": [5, 10],
        "max_anchors": 250,
        "max_jobs": None,
        "job_offset": 0,
        "max_runs": None,
        "anchors": {"max_per_class": 100, "max_per_period": 250},
        "copy_group_payloads": False,
    }
    noise = {
        "enabled": True,
        "reference": {"strategy": "ft", "model": "egraphsage", "seed": 42},
        "periods": [1, 2, 3, 4, 5],
        "anchors_per_period": 100,
        "repeats": 10,
        "null_reseeds": 5,
        "null_epochs": None,
        "explainers": ["ig", "gnn_explainer"],
        "rare_min_eligible": 10,
        "rebuild_anchors": True,
    }
    if args.smoke and stage in {"matrix", "all"}:
        matrix.update({"models": ["egraphsage"], "strategies": ["ft"], "seeds": [42], "max_runs": 1})
    if args.smoke and stage in {"explain", "all"}:
        explain.update(
            {
                "mode": "cross_checkpoint",
                "groups": [["ft", "ewc"], ["egraphsage"], [42]],
                "periods": 5,
                "checkpoint_periods": [1, 2, 3, 4, 5],
                "horizon": "all",
                "max_anchors": 3,
                "max_jobs": None,
                "job_offset": 0,
                "max_runs": None,
                "anchors": {"max_per_class": 3, "max_per_period": 3},
                "copy_group_payloads": True,
            }
        )
    if args.include_ear:
        matrix["include_ear"] = True
        explain["include_ear"] = True
    if args.matrix_config:
        matrix["config"] = str(args.matrix_config)
    if args.models:
        matrix["models"] = [item.strip() for item in args.models.split(",") if item.strip()]
    if args.strategies:
        matrix["strategies"] = [item.strip() for item in args.strategies.split(",") if item.strip()]
    if args.seeds:
        matrix["seeds"] = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if args.periods:
        matrix["periods"] = int(args.periods)
        explain["periods"] = int(args.periods)
    # stage explain: reuse the explicit model/strategy/seed selection for the explain groups
    if stage in {"explain", "all"} and (args.models or args.strategies or args.seeds):
        explain["groups"] = [
            [item.strip() for item in (args.strategies or "ft").split(",") if item.strip()],
            [item.strip() for item in (args.models or "egraphsage").split(",") if item.strip()],
            [int(item.strip()) for item in (args.seeds or "42").split(",") if item.strip()],
        ]
    if args.chunk_size:
        matrix["max_runs"] = int(args.chunk_size)
    if args.max_runs:
        if stage in {"explain", "all"}:
            explain["max_runs"] = int(args.max_runs)
        else:
            matrix["max_runs"] = int(args.max_runs)
    if args.max_jobs:
        explain["max_jobs"] = int(args.max_jobs)
    if args.max_anchors:
        explain["max_anchors"] = int(args.max_anchors)
    if args.anchors_per_period:
        cap = int(args.anchors_per_period)
        explain["anchors"] = {"max_per_class": min(100, cap), "max_per_period": cap}
        explain["max_anchors"] = cap
    if args.checkpoint_horizon:
        explain["horizon"] = str(args.checkpoint_horizon)
    if args.job_offset:
        explain["job_offset"] = int(args.job_offset)
    if args.explainers:
        explain["explainers"] = [item.strip() for item in args.explainers.split(",") if item.strip()]
    if args.include_noise_floor:
        noise["enabled"] = True
    if args.noise_anchors:
        noise["anchors_per_period"] = int(args.noise_anchors)
    if args.null_reseeds:
        noise["null_reseeds"] = int(args.null_reseeds)
    if args.null_epochs:
        noise["null_epochs"] = int(args.null_epochs)
    if args.noise_reference:
        parts = [item.strip() for item in args.noise_reference.split(",")]
        if len(parts) == 3:
            noise["reference"] = {"strategy": parts[0], "model": parts[1], "seed": int(parts[2])}
    return {
        "schema_version": 1,
        "stage": "all" if stage == "all" else stage,
        "run_id": args.run_id,
        "dataset_version": args.dataset_version,
        "commit": args.commit,
        "created_at_utc": utc_now(),
        "matrix": matrix,
        "explain": explain,
        "noise_floor": noise,
    }


def write_runctl(runctl: dict[str, Any], env: dict[str, str]) -> None:
    global RUNCTL_DATASET
    RUNCTL_STAGE.mkdir(parents=True, exist_ok=True)
    (RUNCTL_STAGE / "runctl.json").write_text(json.dumps(runctl, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "title": RUNCTL_DATASET.split("/")[-1],
        "id": RUNCTL_DATASET,
        "licenses": [{"name": "CC0-1.0"}],
        "subtitle": "XDRIFT Kaggle run control (T37)",
        "description": "Private run-control JSON for the XDRIFT Kaggle full-run kernels (T37).",
    }
    ensure_dataset(RUNCTL_STAGE, RUNCTL_DATASET, metadata, f"runctl {runctl['stage']} {runctl['run_id']}", env)



def reset_resume_groups(
    strategies: Sequence[str],
    *,
    models: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    allow_ft_reset: bool = False,
) -> dict[str, Any]:
    """Reset selected strategy groups from the local resume state (T42/D-072).

    Clears only the requested strategy groups' checkpoints, S5 group dirs, manifests,
    matrix state/index rows and (if present) explanation cache entries. FT is protected
    unless ``allow_ft_reset`` is explicitly set. Removed files are archived under
    ``src/kaggle_lane/outputs/kaggle_runs/reset_archive/<timestamp>/`` with a manifest so the
    invalidation is traceable. The next pushed matrix/explain kernel resumes from the
    cleaned state (FT rows untouched).
    """
    selected = [str(item).strip() for item in strategies if str(item).strip()]
    if not selected:
        raise SystemExit("reset: no strategy selected")
    if "ft" in {item.lower() for item in selected} and not allow_ft_reset:
        raise SystemExit("reset: refusing to delete FT groups (use --allow-ft-reset to override)")
    wanted_models = {str(item) for item in (models or [])}
    wanted_seeds = {int(item) for item in (seeds or [])}
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    archive = LOCAL_ROOT / "reset_archive" / f"{timestamp}_{'-'.join(selected)}"
    archive.mkdir(parents=True, exist_ok=True)
    removed: list[str] = []

    def _archive_and_remove(path: Path) -> None:
        if not path.exists():
            return
        if path.is_file():
            target = archive / path.relative_to(RESUME_STATE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            path.unlink()
        else:
            shutil.copytree(path, archive / path.relative_to(RESUME_STATE), dirs_exist_ok=True)
            shutil.rmtree(path)
        removed.append(str(path))

    results = RESUME_STATE / "results"
    runs = RESUME_STATE / "runs"
    # 1) run checkpoints for the selected groups
    group_keys: list[str] = []
    for strategy in selected:
        strategy_runs = runs / strategy
        if not strategy_runs.is_dir():
            continue
        for model_dir in sorted(strategy_runs.iterdir()):
            if not model_dir.is_dir() or (wanted_models and model_dir.name not in wanted_models):
                continue
            for seed_dir in sorted(model_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                    continue
                try:
                    seed_value = int(seed_dir.name.replace("seed", ""))
                except ValueError:
                    continue
                if wanted_seeds and seed_value not in wanted_seeds:
                    continue
                group_key = f"{strategy}_{model_dir.name}_{seed_dir.name}"
                group_keys.append(group_key)
                _archive_and_remove(seed_dir)
        # prune now-empty model/strategy dirs for the reset strategies
        for model_dir in list(strategy_runs.iterdir()) if strategy_runs.is_dir() else []:
            if model_dir.is_dir() and not any(model_dir.iterdir()):
                model_dir.rmdir()
        if strategy_runs.is_dir() and not any(strategy_runs.iterdir()):
            strategy_runs.rmdir()
    # 2) S5 group dirs + state/index rows + manifests
    tables = results / "tables"
    matrix_state_path = tables / "matrix_state.json"
    if matrix_state_path.is_file():
        payload = json.loads(matrix_state_path.read_text(encoding="utf-8"))
        groups = payload.get("groups") or {}
        for key in group_keys:
            groups.pop(key, None)
            _archive_and_remove(tables / "matrix" / key)
        payload["groups"] = groups
        payload["updated_at_utc"] = utc_now()
        matrix_state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matrix_index = tables / "matrix_index.csv"
    if matrix_index.is_file():
        import csv as _csv

        with matrix_index.open("r", encoding="utf-8", newline="") as handle:
            reader = _csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = [row for row in reader if str(row.get("run_key") or row.get("group") or "") not in set(group_keys)]
        with matrix_index.open("w", encoding="utf-8", newline="") as handle:
            writer = _csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    audit = results / "data_audit"
    if audit.is_dir():
        for key in group_keys:
            for path in list(audit.glob(f"*{key}*.json")):
                _archive_and_remove(path)
    # 3) explanation cache/index entries for the reset groups (S7 artifacts are invalid too)
    removed_explanations = 0
    explanations = results / "explanations"
    removed_keys: set[str] = set()
    if explanations.is_dir():
        for metadata_path in list(explanations.glob("*.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(metadata.get("strategy")) not in set(selected):
                continue
            removed_keys.add(metadata_path.stem)
            removed_explanations += 1
            _archive_and_remove(metadata_path)
            npz_path = metadata_path.with_suffix(".npz")
            if npz_path.is_file():
                _archive_and_remove(npz_path)
    for index_path in (tables / "explain_index.csv", explanations / "index.csv"):
        if not index_path.is_file():
            continue
        import csv as _csv

        with index_path.open("r", encoding="utf-8", newline="") as handle:
            reader = _csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = [row for row in reader if str(row.get("key")) not in removed_keys]
        with index_path.open("w", encoding="utf-8", newline="") as handle:
            writer = _csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "strategies": selected,
        "models": sorted(wanted_models) or "all",
        "seeds": sorted(wanted_seeds) or "all",
        "groups_removed": sorted(group_keys),
        "removed_paths": len(removed),
        "explanation_entries_removed": removed_explanations,
        "archive_dir": str(archive),
        "ft_protected": "ft" not in {item.lower() for item in selected},
    }
    (archive / "reset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def has_resume_state() -> bool:
    for name in ("results", "runs"):
        root = RESUME_STATE / name
        if root.is_dir() and any(path.is_file() for path in root.rglob("*")):
            return True
    return False


def upload_resume(env: dict[str, str], message: str) -> None:
    if not has_resume_state():
        return
    if RESUME_STAGE.exists():
        shutil.rmtree(RESUME_STAGE)
    RESUME_STAGE.mkdir(parents=True, exist_ok=True)
    zip_path = RESUME_STAGE / "resume.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("results", "runs"):
            root = RESUME_STATE / name
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(root).as_posix()
                # canonical anchors are rebuilt deterministically by the kernel; do not
                # upload the multi-hundred-MB anchor payloads with every resume push.
                if name == "results" and (relative.startswith("anchors/") or relative.startswith("anchors_noise/")):
                    continue
                archive.write(path, arcname=f"resume/{name}/{relative}")
    metadata = {
        "title": "xdrift-resume",
        "id": RESUME_DATASET,
        "licenses": [{"name": "CC0-1.0"}],
        "subtitle": "XDRIFT Kaggle resume state (T37)",
        "description": "Private resume state (results/ + runs/ trees) for the XDRIFT Kaggle full-run kernels. Generated by src/kaggle_lane/run_full.py; do not edit.",
    }
    ensure_dataset(RESUME_STAGE, RESUME_DATASET, metadata, message, env)


def merge_resume_state(unpacked: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {"results": 0, "runs": 0}
    if not unpacked.is_dir():
        return merged
    for name in ("results", "runs"):
        source = unpacked / name
        if not source.is_dir():
            continue
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            target = RESUME_STATE / name / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            merged[name] += 1
    return merged


def run_validators(tables_dir: Path, stages: list[str], env: dict[str, str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for stage in stages:
        result = subprocess.run(
            [sys.executable, "-m", "src.analysis.validators", "--results-tables", str(tables_dir), "--stage", stage, "--strict"],
            cwd=str(REPO_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300,
        )
        text = ((result.stdout or "") + (result.stderr or "")).strip()
        counts: dict[str, int] | None = None
        overall = "fail"
        for line in text.splitlines():
            if line.startswith("checks:"):
                try:
                    counts = json.loads(line.split(":", 1)[1].strip().replace("'", '"'))
                except Exception:
                    counts = None
            if "validators: PASS" in line:
                overall = "pass"
        results.append({"stage": stage, "returncode": result.returncode, "overall": overall, "counts": counts, "output_tail": text[-2500:]})
        print(f"[run_full] validators --stage {stage} --strict -> {overall} {counts}")
    return results


def unpack_output(output_dir: Path, run_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    unpacked = run_dir / "unpacked"
    if unpacked.exists():
        shutil.rmtree(unpacked)
    unpacked.mkdir(parents=True, exist_ok=True)
    zip_files = sorted(output_dir.glob("xdrift_run_output.zip"))
    summary = None
    if zip_files:
        with zipfile.ZipFile(zip_files[0]) as archive:
            archive.extractall(unpacked)
        summary_path = unpacked / "run_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return (unpacked if unpacked.is_dir() else None), summary


def explain_groups_from_runctl(explain: dict[str, Any]) -> list[tuple[str, str, int]]:
    groups = explain.get("groups") or [["ft"], ["egraphsage"], [42]]
    if len(groups) == 3 and all(isinstance(item, (list, tuple)) for item in groups):
        strategies, models, seeds = groups
        return [(str(strategy), str(model), int(seed)) for strategy in strategies for model in models for seed in seeds]
    if all(isinstance(item, (list, tuple)) and len(item) == 3 for item in groups):
        return [(str(item[0]), str(item[1]), int(item[2])) for item in groups]
    raise SystemExit(f"runctl explain.groups has an unsupported shape: {groups!r}")


def check_resume_checkpoints(explain: dict[str, Any]) -> dict[str, Any]:
    periods = int(explain.get("periods") or 5)
    groups = explain_groups_from_runctl(explain)
    missing: list[str] = []
    found: list[str] = []
    for strategy, model, seed in groups:
        for period in range(1, periods + 1):
            path = RESUME_STATE / "runs" / strategy / model / f"seed{seed}" / f"M_{period}.pt"
            if path.is_file():
                found.append(f"{strategy}/{model}/seed{seed}/M_{period}.pt")
            else:
                missing.append(f"{strategy}/{model}/seed{seed}/M_{period}.pt")
    return {"groups": [f"{s}_{m}_seed{seed}" for s, m, seed in groups], "checkpoints_found": len(found), "missing": missing}


def write_report(run_dir: Path, meta: dict[str, Any]) -> Path:
    lines = [
        "# Kaggle full-run stage report (T37 / D-065-D-066)",
        "",
        f"*Generated {utc_now()} by `src/kaggle_lane/run_full.py`.*",
        "",
        "## Run",
        "",
        f"- run id: `{meta.get('run_id')}`",
        f"- stage: `{meta.get('stage')}` · dataset: `{INPUTS_DATASET}` v{meta.get('dataset_version')} (commit `{meta.get('commit')}`)",
        f"- kernel ref: `{meta.get('kernel_ref')}` · status: **{meta.get('kernel_status')}**",
        f"- output dir: `{meta.get('output_dir')}`",
        f"- inner status: **{meta.get('inner_status')}** · error: {meta.get('error') or 'none'}",
        f"- inner wall clock: {meta.get('inner_wall_clock_s')} s · local wall clock: {meta.get('local_wall_clock_s')} s",
        f"- resume merged: {json.dumps(meta.get('resume_merge'))}",
        "",
    ]
    inner = meta.get("inner") or {}
    steps = inner.get("per_step_logs") or []
    if steps:
        lines += ["## Steps", "", "| step | rc | duration (s) |", "| --- | --- | --- |"]
        lines += [f"| {step.get('step')} | {step.get('returncode')} | {step.get('duration_s')} |" for step in steps]
        lines.append("")
    matrix = inner.get("matrix") or {}
    if matrix:
        lines += ["## Matrix", "", f"- groups complete: {matrix.get('groups_complete')} / {matrix.get('groups_planned')}", ""]
        timings = matrix.get("train_timings") or []
        if timings:
            lines += ["| manifest | strategy | model | period | epoch s | max_mem GiB |", "| --- | --- | --- | --- | --- | --- |"]
            for row in timings:
                lines.append(f"| {row.get('manifest')} | {row.get('strategy')} | {row.get('model')} | {row.get('period')} | {row.get('epoch_seconds_mean')} | {row.get('max_memory_allocated_gib')} |")
            lines.append("")
    explain = inner.get("explain") or {}
    if explain:
        lines += ["## Explain", "", f"- jobs: {explain.get('jobs_executed')} / {explain.get('jobs_planned')} · failures: {len(explain.get('failures') or [])}",
                  f"- explain index: {json.dumps(explain.get('explain_index'))}", ""]
    validators = meta.get("validators") or []
    if validators:
        lines += ["## Local validators", "", "| stage | overall | counts |", "| --- | --- | --- |"]
        lines += [f"| {item.get('stage')} | {item.get('overall')} | {json.dumps(item.get('counts'))} |" for item in validators]
        lines.append("")
    lines += ["## Exact real-run commands", "", "```powershell",
              f".venv\\Scripts\\python.exe kaggle_lane\\run_full.py --stage {meta.get('stage')} --run-id <id>",
              "# full matrix (Lead green-light): --stage matrix --chunk-size 10 --max-pushes 3",
              "# full explain slices: --stage explain --max-jobs 30", "```", ""]
    path = run_dir / "RunReport.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["matrix", "explain", "noise", "all"], default="matrix")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset-version", type=int, default=None)
    parser.add_argument("--commit", default=None, help="expected dataset code snapshot commit (default: read from API/inputs)")
    parser.add_argument("--models", default=None)
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--periods", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--matrix-config", default=None, help="config path for the matrix kernel (e.g. configs/train_ewc_lambda100.yaml)")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument("--anchors-per-period", type=int, default=None, help="canonical anchor cap per period (D-071: 250; smoke: 3)")
    parser.add_argument("--checkpoint-horizon", choices=["all", "adjacent", "final"], default=None, help="corrected-S7 checkpoint horizon (default all t>=j)")
    parser.add_argument("--job-offset", type=int, default=None, help="skip the first N pending corrected-S7 jobs (normally 0; skip-completed advances automatically)")
    parser.add_argument("--explainers", default=None)
    parser.add_argument("--include-ear", action="store_true")
    parser.add_argument("--include-noise-floor", action="store_true")
    parser.add_argument("--noise-anchors", type=int, default=None, help="stratified anchors per period for the noise floor (default 100)")
    parser.add_argument("--null-reseeds", type=int, default=None, help="model-null reseeds R (default 5)")
    parser.add_argument("--null-epochs", type=int, default=None, help="null-model retraining epochs (default 10)")
    parser.add_argument("--noise-reference", default=None, help="noise-floor reference group strategy,model,seed (default ft,egraphsage,42)")
    parser.add_argument("--max-pushes", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=6 * 3600)
    parser.add_argument("--poll-s", type=int, default=30)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-resume-upload", action="store_true", help="mount the existing resume dataset version without uploading a new one (safe while another kernel is running)")
    parser.add_argument("--reset-groups", default=None, help="comma list of strategies whose groups are cleared from the local resume state before the run (T42: replay[,ewc])")
    parser.add_argument("--reset-models", default=None, help="optional model filter for --reset-groups (default all)")
    parser.add_argument("--reset-seeds", default=None, help="optional seed filter for --reset-groups (default all)")
    parser.add_argument("--reset-only", action="store_true", help="perform --reset-groups and exit without pushing a kernel")
    parser.add_argument("--allow-ft-reset", action="store_true", help="allow --reset-groups to include ft (normally protected)")
    parser.add_argument("--kernel-slug", default=None, help="override the kernel slug/title for this run (e.g. a separate smoke kernel while the full-run kernel is active)")
    parser.add_argument("--kernel-title", default=None)
    parser.add_argument("--runctl-dataset", default=None, help="override the run-control dataset ref (default mysteriousavailable/xdrift-runctl)")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--skip-validators", action="store_true")
    args = parser.parse_args(argv)

    env = build_env()
    owner = discover_username(env)
    global RUNCTL_DATASET
    if args.runctl_dataset:
        RUNCTL_DATASET = args.runctl_dataset
    if args.kernel_slug:
        spec = KERNEL_SPECS[args.stage]
        spec["slug"] = args.kernel_slug
        spec["title"] = args.kernel_title or args.kernel_slug.replace("-", " ")
    run_id = args.run_id or f"t37-{args.stage}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = LOCAL_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    dataset_version_now = wait_for_dataset_version(INPUTS_DATASET, args.dataset_version, env)
    if args.dataset_version is not None and dataset_version_now is not None and int(dataset_version_now) < int(args.dataset_version):
        raise SystemExit(f"dataset version {dataset_version_now} < requested {args.dataset_version}")
    if args.dataset_version and args.commit is None:
        # the env manifest inside the dataset records the exact committed snapshot; verify via core metadata
        pass
    commit = args.commit or ""
    runctl = default_runctl(args.stage, argparse.Namespace(**{**vars(args), "run_id": run_id, "dataset_version": dataset_version_now, "commit": commit}))
    if args.reset_groups:
        reset_manifest = reset_resume_groups(
            [item.strip() for item in args.reset_groups.split(",") if item.strip()],
            models=[item.strip() for item in (args.reset_models or "").split(",") if item.strip()] or None,
            seeds=[int(item) for item in (args.reset_seeds or "").split(",") if item.strip()] or None,
            allow_ft_reset=bool(args.allow_ft_reset),
        )
        print("[run_full] reset summary: " + json.dumps({key: reset_manifest.get(key) for key in ("groups_removed", "removed_paths", "explanation_entries_removed", "archive_dir")}))
        (LOCAL_ROOT / f"reset_{run_id}.json").write_text(json.dumps(reset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.reset_only:
            return 0
    write_runctl(runctl, env)
    refs: list[str] = []
    kernels = ["matrix", "explain", "noise"] if args.stage == "all" else [args.stage]
    for stage_name in kernels:
        spec = KERNEL_SPECS[stage_name]
        ref = f"{owner}/{spec['slug']}"
        refs.append(ref)
        script = build_kernel_script(stage_name, ref)
        print(f"[run_full] built {stage_name} kernel script: {script}")
    if args.stage == "noise":
        reference = runctl["noise_floor"]["reference"]
        raw_noise_periods = runctl["noise_floor"].get("periods", [1, 2, 3, 4, 5])
        noise_period_count = int(raw_noise_periods) if isinstance(raw_noise_periods, int) else len(raw_noise_periods)
        noise_check = check_resume_checkpoints(
            {"groups": [[reference["strategy"]], [reference["model"]], [reference["seed"]]], "periods": noise_period_count}
        )
        print(f"[run_full] noise resume precheck: {json.dumps(noise_check)}")
        if noise_check["missing"]:
            raise SystemExit("noise stage requires the reference group checkpoints in the resume state; missing: " + ", ".join(noise_check["missing"][:10]))
    if args.stage in {"explain", "all"}:
        resume_check = check_resume_checkpoints(runctl["explain"])
        print(f"[run_full] explain resume precheck: {json.dumps(resume_check)}")
        if resume_check["missing"]:
            raise SystemExit(
                "explain stage requires matrix checkpoints in the resume state; missing: "
                + ", ".join(resume_check["missing"][:10])
                + " — run/finish the matrix stage first (its output is merged into xdrift-resume)."
            )
    if args.build_only:
        print(json.dumps({"runctl": runctl, "refs": refs}, indent=2))
        return 0
    if args.no_push:
        print(json.dumps({"runctl": runctl, "refs": refs, "note": "no-push requested"}, indent=2))
        return 0

    results: list[dict[str, Any]] = []
    pushes = 0
    planned_groups = None
    if args.stage in {"matrix", "all"}:
        planned_groups = len(runctl["matrix"]["models"]) * len(runctl["matrix"]["strategies"]) * len(runctl["matrix"]["seeds"])
    while pushes < max(1, args.max_pushes):
        pushes += 1
        if args.chunk_size and planned_groups:
            runctl["matrix"]["max_runs"] = min(planned_groups, int(args.chunk_size) * pushes)
            write_runctl(runctl, env)
            print(f"[run_full] chunk push {pushes}: matrix.max_runs={runctl['matrix']['max_runs']}/{planned_groups}")
        if args.skip_resume_upload:
            print(f"[run_full] resume upload skipped (mounting existing {RESUME_DATASET} version)")
        else:
            upload_resume(env, f"resume before {run_id} push {pushes}")
        for stage_name in kernels:
            spec = KERNEL_SPECS[stage_name]
            ref = f"{owner}/{spec['slug']}"
            sources = [INPUTS_DATASET, RUNCTL_DATASET] + ([RESUME_DATASET] if has_resume_state() else [])
            kernel_summary = push_and_wait(
                slug=spec["slug"],
                code_path=spec["dir"] / "script.py",
                gpu=True,
                internet=True,
                title=spec["title"],
                dataset_sources=sources,
                timeout_s=args.timeout_s,
                poll_s=args.poll_s,
                fetch_output=True,
            )
            output_dir = Path(kernel_summary.get("output_dir", ""))
            for path in output_dir.rglob("*") if output_dir.is_dir() else []:
                if path.is_file():
                    target = run_dir / "kernel_output" / path.relative_to(output_dir)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
            unpacked, inner = unpack_output(output_dir, run_dir)
            merge = merge_resume_state(unpacked) if unpacked else {}
            validators: list[dict[str, Any]] = []
            if not args.skip_validators and unpacked is not None:
                tables = unpacked / "results" / "tables"
                if stage_name == "matrix":
                    # Matrix runs write one S5 artifact set per group under
                    # results/text/tables/matrix/<run_key>/ (T37); validate each group.
                    matrix_root = tables / "matrix"
                    group_dirs = sorted(p for p in matrix_root.glob("*") if p.is_dir()) if matrix_root.is_dir() else []
                    if group_dirs:
                        for group_dir in group_dirs:
                            validators.extend(run_validators(group_dir, ["s5"], env))
                    else:
                        validators = run_validators(tables, ["s5"], env)
                elif stage_name == "noise":
                    noise_root = unpacked / "results" / "explanations_noise"
                    noise_present = (noise_root / "combined_null.csv").is_file() and (noise_root / "model_null" / "index.csv").is_file() and (noise_root / "explainer_repeat" / "index.csv").is_file()
                    validators = [
                        {
                            "stage": "noise_presence",
                            "overall": "pass" if noise_present else "fail",
                            "counts": None,
                            "output_tail": str(noise_root),
                        }
                    ]
                else:
                    validators = run_validators(tables, ["s5", "s6", "s7"], env)
            meta = {
                "run_id": run_id,
                "stage": stage_name,
                "dataset_version": dataset_version_now,
                "commit": commit,
                "kernel_ref": kernel_summary.get("ref"),
                "kernel_status": kernel_summary.get("status"),
                "output_dir": str(run_dir / "kernel_output"),
                "inner_status": (inner or {}).get("status"),
                "error": (inner or {}).get("error"),
                "inner_wall_clock_s": (inner or {}).get("wall_clock_s"),
                "local_wall_clock_s": round(time.time() - started, 1),
                "resume_merge": merge,
                "inner": inner,
                "validators": validators,
                "dataset_sources": sources,
            }
            (run_dir / f"{stage_name}_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            report = write_report(run_dir, meta)
            results.append(meta)
            print(f"[run_full] {stage_name} status={meta['kernel_status']} inner={meta['inner_status']} report={report}")
        # stop early when a full matrix/explain slice already completed everything
        if not any(r.get("inner_status") == "success" for r in results):
            break
        if args.stage == "matrix" and runctl["matrix"].get("max_runs") and len(results) < args.max_pushes:
            continue
        if args.stage in {"explain", "all"}:
            explain_jobs = ((results[-1].get("inner") or {}).get("explain") or {}).get("jobs") or {}
            pending_after = int(explain_jobs.get("pending_after_push") or 0)
            print(f"[run_full] corrected S7 push {pushes}: executed={explain_jobs.get('executed_this_push')} pending_after={pending_after} pairs_completed={explain_jobs.get('pairs_completed_state')}")
            if pending_after > 0 and pushes < args.max_pushes:
                continue
        break
    (run_dir / "run_full_meta.json").write_text(json.dumps({"run_id": run_id, "pushes": pushes, "results": [{k: r.get(k) for k in ("stage", "kernel_ref", "kernel_status", "inner_status", "validators")} for r in results]}, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    ok = all(r.get("kernel_status") == "complete" and r.get("inner_status") == "success" for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
