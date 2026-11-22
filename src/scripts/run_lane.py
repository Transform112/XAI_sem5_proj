#!/usr/bin/env python
"""Canonical lane launcher: resolve lane + config, set env vars, run, record manifest.

Why a Python launcher with thin platform shims
----------------------------------------------
The Windows (``src/scripts/run_lane.ps1``) and POSIX (``src/scripts/run_lane.sh``)
entry points only locate a Python interpreter and delegate here. Keeping the
logic in one place means Windows, Kaggle and CPU lanes cannot drift apart.

What this does
--------------
1. Resolves the repository root and the lane (``configs/paths/<lane>.yaml``).
2. Loads the stage config through :func:`src.utils.config.load_config`
   (base.yaml -> stage yaml -> ``--set`` overrides) and the lane metadata.
3. Validates inputs: lane exists, requirements file exists, config parses,
   entry script exists, Python interpreter exists, and (when the lane declares
   ``cuda_required: true``) that torch sees a CUDA device.
4. Creates the standard directory tree and a per-run manifest
   (``runs/manifests/<run-name>.json``) with git SHA, config hash, versions,
   GPU, seed and timestamps.
5. Runs the entry script as a child process with ``XDRIFT_*`` environment
   variables set from the resolved path layout, so the child never guesses.
6. Finalises the manifest with exit status, duration and child environment.

Examples
--------
    python src/scripts/run_lane.py --lane legacy --config configs/audit.yaml \
        --entry src/scripts/run_data_audit.py -- --dataset NF-UNSW-NB15-v3

    python src/scripts/run_lane.py --lane rtx5060 --config configs/train.yaml \
        --entry src/scripts/run_train.py -- --strategy replay --seed 42

    python src/scripts/run_lane.py --list-lanes
    python src/scripts/run_lane.py --lane cpu --entry src/scripts/run_data_audit.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

# Make ``import src`` work regardless of the current directory.
_LAUNCHER_ROOT = Path(__file__).resolve().parents[2]
if str(_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAUNCHER_ROOT))

from src.utils.config import Config, ConfigError, load_config, load_yaml  # noqa: E402
from src.utils.manifest import RunManifest, git_executable, utc_now  # noqa: E402
from src.utils.paths import LANES, PathConfigError, find_project_root, load_paths  # noqa: E402

DEFAULT_LANE = "legacy"
DEFAULT_CONFIG = "configs/base.yaml"
VENV_DIR = ".venv"
_EXIT_USAGE = 2


class LaunchError(RuntimeError):
    """A launch precondition failed; the message tells the user how to fix it."""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_lane",
        description="Canonical XDRIFT launcher (lane/config resolution + run manifest).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lane",
        default=os.environ.get("XDRIFT_LANE"),
        help=f"execution lane ({', '.join(name for name in LANES if name != 'default')}); "
        f"default: XDRIFT_LANE or {DEFAULT_LANE}",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"stage config YAML relative to the project root (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--entry", default=None, help="Python entry script to run (relative to root)")
    parser.add_argument("--root", default=None, help="project root override (default: auto-discover)")
    parser.add_argument("--run-name", default=None, help="manifest/run name (default: generated)")
    parser.add_argument("--python", default=None, help="Python interpreter override")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, applied last (repeatable)",
    )
    parser.add_argument(
        "--env",
        dest="env_overrides",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="extra environment variable for the child process (repeatable)",
    )
    parser.add_argument("--skip-cuda-check", action="store_true", help="do not verify CUDA on GPU lanes")
    parser.add_argument("--list-lanes", action="store_true", help="print the lane table and exit")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan, run nothing")
    parser.add_argument("--json", action="store_true", help="print the launch plan as JSON")
    parser.add_argument(
        "child_args",
        nargs=argparse.REMAINDER,
        help="arguments after '--' are forwarded verbatim to the entry script",
    )
    return parser


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------
def resolve_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise LaunchError(f"--root does not exist: {root}")
        return root
    try:
        return find_project_root()
    except PathConfigError as exc:
        raise LaunchError(str(exc)) from exc


def resolve_lane(explicit: str | None) -> str:
    lane = explicit or os.environ.get("XDRIFT_LANE") or DEFAULT_LANE
    if lane == "default":
        raise LaunchError(
            "'default' is the shared path base, not an execution lane; "
            f"choose one of: {', '.join(name for name in LANES if name != 'default')}"
        )
    if lane not in LANES:
        raise LaunchError(
            f"unknown lane {lane!r}; choose one of: {', '.join(name for name in LANES if name != 'default')}"
        )
    return lane


def load_lane_metadata(root: Path, lane: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (lane metadata, raw lane-file data) for *lane*."""
    lane_file = root / "configs" / "paths" / f"{lane}.yaml"
    if not lane_file.is_file():
        available = sorted(path.stem for path in (root / "configs" / "paths").glob("*.yaml"))
        raise LaunchError(
            f"lane file missing: {lane_file} (available: {', '.join(available) or 'none'})"
        )
    raw = load_yaml(lane_file)
    meta = raw.get("lane") or {}
    if not isinstance(meta, dict):
        raise LaunchError(f"'lane' block in {lane_file} must be a mapping")
    meta = dict(meta)
    meta.setdefault("name", lane)
    meta["file"] = str(lane_file)
    return meta, raw


def list_lanes(root: Path) -> int:
    rows: list[dict[str, Any]] = []
    for lane_file in sorted((root / "configs" / "paths").glob("*.yaml")):
        if lane_file.stem == "default":
            continue
        data = load_yaml(lane_file)
        meta = data.get("lane") or {}
        rows.append(
            {
                "lane": lane_file.stem,
                "role": meta.get("role"),
                "torch_pin": meta.get("torch_pin"),
                "cuda_required": meta.get("cuda_required"),
                "description": meta.get("description"),
                "file": str(lane_file),
            }
        )
    if not rows:
        print("no lane files found", file=sys.stderr)
        return 1
    headers = ("lane", "role", "torch_pin", "cuda_required", "description")
    widths = {header: max(len(header), *(len(str(row.get(header))) for row in rows)) for header in headers}
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row.get(header)).ljust(widths[header]) for header in headers))
    return 0


def resolve_python(explicit: str | None, root: Path) -> Path:
    """Pick the interpreter: --python, XDRIFT_PYTHON, project venv, else this one."""
    if explicit:
        candidate = explicit
    elif os.environ.get("XDRIFT_PYTHON"):
        candidate = os.environ["XDRIFT_PYTHON"]
    else:
        venv = root / VENV_DIR
        if os.name == "nt":
            venv_python = venv / "Scripts" / "python.exe"
        else:
            venv_python = venv / "bin" / "python"
        candidate = str(venv_python) if venv_python.is_file() else sys.executable
    for path in (candidate, str(Path(candidate).expanduser())):
        if Path(path).is_file():
            return Path(path).resolve()
    # Bare command names (e.g. "python") are allowed; existence is checked at spawn.
    if Path(candidate).name == candidate:
        return Path(candidate)
    raise LaunchError(f"Python interpreter not found: {candidate}")


def resolve_entry(explicit: str, root: Path) -> Path:
    entry = Path(explicit).expanduser()
    if not entry.is_absolute():
        entry = root / entry
    if not entry.is_file():
        raise LaunchError(f"entry script not found: {entry}")
    if entry.suffix.lower() != ".py":
        raise LaunchError(f"entry script must be a .py file, got: {entry}")
    return entry.resolve()


def split_child_args(child_args: Sequence[str]) -> list[str]:
    """Strip the argparse separator so child processes get clean arguments."""
    args = list(child_args)
    while args and args[0] == "--":
        args.pop(0)
    return args


def check_cuda(python: Path) -> tuple[bool, str]:
    """Ask the child interpreter whether torch sees a CUDA device."""
    probe = "import torch;print('CUDA', torch.cuda.is_available())"
    try:
        completed = subprocess.run(
            [str(python), "-c", probe], capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"torch probe failed: {type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        tail = (completed.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        return False, f"torch import failed: {tail[0]}"
    ok = "CUDA True" in completed.stdout
    return ok, completed.stdout.strip()


def build_run_name(entry: Path, lane: str, explicit: str | None) -> str:
    if explicit:
        name = "".join(char if (char.isalnum() or char in "-_.") else "_" for char in explicit).strip("._-")
        if not name:
            raise LaunchError("--run-name must contain at least one alphanumeric character")
        return name
    stamp = utc_now().replace(":", "").replace("-", "").replace("+0000", "Z")
    return f"{entry.stem}_{lane}_{stamp}"


def parse_env_overrides(items: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise LaunchError(f"--env expects NAME=VALUE, got: {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name or not name.replace("_", "").isalnum():
            raise LaunchError(f"invalid environment variable name: {name!r}")
        parsed[name] = value
    return parsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        root = resolve_root(args.root)
        if args.list_lanes:
            return list_lanes(root)
        lane = resolve_lane(args.lane)
        lane_meta, _lane_raw = load_lane_metadata(root, lane)

        requirements = lane_meta.get("requirements")
        if requirements:
            requirements_path = Path(str(requirements))
            if not requirements_path.is_absolute():
                requirements_path = root / requirements_path
            if not requirements_path.is_file():
                raise LaunchError(f"lane requirements file missing: {requirements_path}")

        if not args.entry:
            raise LaunchError("--entry is required (or use --list-lanes)")
        entry = resolve_entry(args.entry, root)
        python = resolve_python(args.python, root)
        child_args = split_child_args(args.child_args)
        if child_args and child_args[0].startswith("--") and not args.entry:
            raise LaunchError("could not parse child arguments")

        config = load_config(
            args.config,
            overrides=args.overrides,
            project_root=root,
            lane=lane,
        )
        seed = config.get("seed")
        paths = load_paths(lane=lane, project_root=root, ensure=True)

        cuda_required = bool(lane_meta.get("cuda_required", False))
        cuda_ok: bool | None = None
        cuda_detail = "not checked"
        if cuda_required and not args.skip_cuda_check:
            cuda_ok, cuda_detail = check_cuda(python)
            if not cuda_ok:
                raise LaunchError(
                    f"lane {lane!r} requires CUDA but the torch probe failed/unavailable "
                    f"({cuda_detail}). Use --skip-cuda-check to bypass deliberately."
                )

        run_name = build_run_name(entry, lane, args.run_name)
        manifest_path = paths.manifests_root / f"{run_name}.json"
        manifest = RunManifest(
            "lane_launch",
            paths=paths,
            config=config,
            lane=lane,
            seed=seed if isinstance(seed, int) else None,
            out_path=manifest_path,
            extra={
                "entry": paths.relative(entry),
                "entry_abs": str(entry),
                "child_args": child_args,
                "lane_meta": lane_meta,
                "python_used": str(python),
                "python_source": "explicit"
                if args.python
                else ("XDRIFT_PYTHON" if os.environ.get("XDRIFT_PYTHON") else "auto"),
                "cuda_required": cuda_required,
                "cuda_check": {"checked": cuda_required and not args.skip_cuda_check, "ok": cuda_ok, "detail": cuda_detail},
                "env_overrides": parse_env_overrides(args.env_overrides),
                "config_override_count": len(args.overrides),
            },
        )

        child_env = os.environ.copy()
        child_env.update(paths.to_env())
        child_env["XDRIFT_LANE"] = lane
        child_env["XDRIFT_CONFIG"] = str(config.path) if config.path else ""
        child_env["XDRIFT_RUN_NAME"] = run_name
        child_env["XDRIFT_MANIFEST"] = str(manifest_path)
        child_env["PYTHONHASHSEED"] = child_env.get("PYTHONHASHSEED", "0")
        child_env.update(manifest.payload["extra"]["env_overrides"])
        # Make ``import src`` work in child scripts regardless of their location.
        existing_pythonpath = child_env.get("PYTHONPATH", "")
        root_text = str(root)
        if root_text not in existing_pythonpath.split(os.pathsep):
            child_env["PYTHONPATH"] = (
                root_text if not existing_pythonpath else root_text + os.pathsep + existing_pythonpath
            )
        if not child_env.get("XDRIFT_GIT"):
            found_git = git_executable()
            if found_git:
                child_env["XDRIFT_GIT"] = found_git

        command = [str(python), str(entry), *child_args]
        plan = {
            "root": str(root),
            "lane": lane,
            "entry": str(entry),
            "python": str(python),
            "config": str(config.path) if config.path else None,
            "config_sha256": config.sha256,
            "seed": seed,
            "command": command,
            "manifest": str(manifest_path),
            "cuda_required": cuda_required,
            "cuda_check": {"ok": cuda_ok, "detail": cuda_detail},
            "env": {key: value for key, value in child_env.items() if key.startswith("XDRIFT_")},
        }
        if args.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True))

        print(f"[run_lane] lane={lane} config={config.path} sha256={config.sha256[:12]}")
        print(f"[run_lane] manifest -> {manifest_path}")
        print(f"[run_lane] exec: {' '.join(command)}", flush=True)

        manifest.start()
        try:
            completed = subprocess.run(command, cwd=str(root), env=child_env)
        except OSError as exc:
            manifest.finish("failed", error=f"could not start entry script: {exc}")
            manifest.write()
            raise LaunchError(f"could not start entry script: {exc}") from exc
        rc = int(completed.returncode)
        manifest.finish(
            "success" if rc == 0 else "failed",
            error=None if rc == 0 else f"entry script exited with code {rc}",
            extra={"exit_code": rc},
        )
        manifest.write()
        print(f"[run_lane] exit={rc} duration={manifest.payload.get('duration_s')}s manifest={manifest_path}")
        return rc
    except (LaunchError, ConfigError, PathConfigError) as exc:
        print(f"[run_lane] ERROR: {exc}", file=sys.stderr)
        return _EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
