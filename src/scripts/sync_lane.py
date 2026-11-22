#!/usr/bin/env python
"""Move the repository and small artifacts between execution lanes.

The four lanes cannot share a filesystem (dev PC -> user's RTX 5060 PC -> Kaggle
notebooks), so this script is the portable transport layer:

    bundle   create a git bundle of all branches (full history, small)
    archive  create a ZIP snapshot of the working tree (no .git, no heavy dirs)
    collect  pack the small result artifacts of one lane for transfer back
    verify   verify a git bundle before trusting it
    clone    clone a project from a git bundle

No network access is used and no absolute paths are hard-coded. ``git`` is
resolved through ``XDRIFT_GIT`` or PATH, exactly as the run manifest does.

Examples
--------
    python src/scripts/sync_lane.py bundle --out ../xai_proj.bundle
    python src/scripts/sync_lane.py archive --out ../xai_proj.zip
    python src/scripts/sync_lane.py collect --out results_legacy.zip
    python src/scripts/sync_lane.py verify --bundle ../xai_proj.bundle
    python src/scripts/sync_lane.py clone --bundle ../xai_proj.bundle --dest ../xai_proj_copy
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

_LAUNCHER_ROOT = Path(__file__).resolve().parents[2]
if str(_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAUNCHER_ROOT))

from src.utils.manifest import git_executable  # noqa: E402
from src.utils.paths import find_project_root, load_paths  # noqa: E402

#: Directories never shipped in an archive (regenerated or machine-local).
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "data",
    "runs",
    "cache",
    "logs",
    "tools",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".dsh",
}

#: File suffixes never shipped (large model/tensor artifacts).
EXCLUDED_SUFFIXES = {".pt", ".pth", ".npz", ".npy", ".bundle", ".zip", ".tmp", ".log"}

#: Result subpaths shipped by ``collect`` (small, trackable artifacts).
COLLECT_ROOTS = ("results", "docs", "verification")


def _git() -> str:
    exe = git_executable()
    if exe is None:
        raise SystemExit("git not found: set XDRIFT_GIT or put git on PATH")
    return exe


def _run(cmd: Sequence[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        list(cmd), cwd=str(cwd) if cwd else None, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise SystemExit(f"command failed ({completed.returncode}): {' '.join(cmd)}\n{completed.stderr}")
    return completed


def _repo_root(explicit: str | None) -> Path:
    return Path(explicit).expanduser().resolve() if explicit else find_project_root()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_bundle(args: argparse.Namespace) -> int:
    root = _repo_root(args.root)
    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    refs = args.refs if args.refs else ["--all"]
    _run([_git(), "-C", str(root), "bundle", "create", str(out), *refs])
    print(f"[sync] bundle written: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("[sync] on the target lane: python src/scripts/sync_lane.py clone --bundle "
          f"{out.name} --dest <new_dir>")
    return 0


def _iter_files(root: Path, extra_excludes: Iterable[str] = ()) -> Iterable[Path]:
    excludes = EXCLUDED_DIRS | set(extra_excludes)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in excludes for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        yield path


def cmd_archive(args: argparse.Namespace) -> int:
    root = _repo_root(args.root)
    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in _iter_files(root):
            archive.write(path, arcname=str(path.relative_to(root)).replace(os.sep, "/"))
            count += 1
    print(f"[sync] archive written: {out} ({out.stat().st_size / 1e6:.1f} MB, {count} files)")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    """Pack the small artifacts of one lane (results, docs, verification)."""
    paths = load_paths(lane=args.lane, project_root=args.root)
    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    sources = [paths.project_root / name for name in COLLECT_ROOTS]
    sources.append(paths.manifests_root)
    sources.append(paths.logs_root)
    count = 0
    missing: list[str] = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source in sources:
            if not source.exists():
                missing.append(str(source))
                continue
            base_name = source.name if source.parent != paths.project_root else source.name
            for path in _iter_files(source):
                rel = path.relative_to(paths.project_root)
                archive.write(path, arcname=str(rel).replace(os.sep, "/"))
                count += 1
    if missing:
        print(f"[sync] note: not present on this lane: {', '.join(missing)}")
    print(f"[sync] collected {count} files -> {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle).expanduser()
    if not bundle.is_file():
        raise SystemExit(f"bundle not found: {bundle}")
    completed = _run([_git(), "bundle", "verify", str(bundle)])
    print("[sync] bundle verified")
    print(completed.stdout.strip() or completed.stderr.strip())
    return 0


def cmd_clone(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle).expanduser()
    dest = Path(args.dest).expanduser()
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"destination exists and is not empty: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run([_git(), "clone", str(bundle), str(dest)])
    branches = _run([_git(), "-C", str(dest), "branch", "-a"]).stdout.strip()
    print(f"[sync] cloned into {dest}\nbranches:\n{branches}")
    print("[sync] next: python -m venv .venv && run the lane setup script (see platforms/README.md)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sync_lane", description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None, help="project root override")
    sub = parser.add_subparsers(dest="command", required=True)

    bundle = sub.add_parser("bundle", help="git bundle of all refs")
    bundle.add_argument("--out", required=True)
    bundle.add_argument("--refs", nargs="*", default=None, help="refs to include (default: --all)")
    bundle.set_defaults(func=cmd_bundle)

    archive = sub.add_parser("archive", help="ZIP snapshot of the working tree")
    archive.add_argument("--out", required=True)
    archive.set_defaults(func=cmd_archive)

    collect = sub.add_parser("collect", help="pack small results for transfer back to the dev lane")
    collect.add_argument("--out", required=True)
    collect.add_argument("--lane", default=None, help="lane whose layout to use (default: auto)")
    collect.set_defaults(func=cmd_collect)

    verify = sub.add_parser("verify", help="verify a git bundle")
    verify.add_argument("--bundle", required=True)
    verify.set_defaults(func=cmd_verify)

    clone = sub.add_parser("clone", help="clone a project from a git bundle")
    clone.add_argument("--bundle", required=True)
    clone.add_argument("--dest", required=True)
    clone.set_defaults(func=cmd_clone)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - CLI surface
        print(f"[sync] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
