"""Push and run a script kernel on Kaggle GPU, then fetch its output.

Uses the official Kaggle CLI 2.x (which supports the new-style KAGGLE_API_TOKEN bearer token),
so the API plumbing stays Kaggle's. The lane is replication-only (docs/07 §5).

Usage (from the repository root, with the project venv):
    python src/kaggle_lane/run_kernel.py --code my_kernel.py --slug xdrift-pilot --gpu
    python src/kaggle_lane/run_kernel.py --code my_kernel.py --slug xdrift-pilot --gpu --wait

The kernel runs privately with the metadata written under
`src/kaggle_lane/runs/<slug>/`, and its output is downloaded to
`src/kaggle_lane/outputs/<slug>/`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = Path(__file__).resolve().parent / "runs"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
TOKEN_FILES = [Path.home() / ".kaggle" / "kgat.txt", Path.home() / ".kaggle" / "access_token"]


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    if not env.get("KAGGLE_API_TOKEN") and not env.get("KAGGLE_KEY"):
        for path in TOKEN_FILES:
            if path.exists():
                env["KAGGLE_API_TOKEN"] = path.read_text(encoding="utf-8").strip().splitlines()[0]
                break
    if not (env.get("KAGGLE_API_TOKEN") or env.get("KAGGLE_KEY")):
        raise RuntimeError("no Kaggle token: set KAGGLE_API_TOKEN or write ~/.kaggle/kgat.txt")
    return env


def kaggle_bin() -> list[str]:
    exe = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    if exe.exists():
        return [str(exe)]
    found = shutil.which("kaggle")
    if found:
        return [found]
    return [sys.executable, "-m", "kaggle"]


def run_cli(args: list[str], env: dict[str, str], timeout: int = 300) -> subprocess.CompletedProcess:
    cmd = kaggle_bin() + args
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    return result


def discover_username(env: dict[str, str]) -> str:
    try:
        result = run_cli(["config", "view"], env, timeout=60)
        for line in (result.stdout or "").splitlines():
            if line.strip().startswith("- username:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if username:
        return username
    raise RuntimeError("cannot determine Kaggle username (kaggle config view failed)")


def write_kernel_dir(slug: str, code_path: Path, title: str, gpu: bool, internet: bool,
                     dataset_sources: list[str], kernel_sources: list[str]) -> Path:
    kernel_dir = RUNS_DIR / slug
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True)
    shutil.copy2(code_path, kernel_dir / "script.py")
    metadata = {
        "id": f"__OWNER__/{slug}",
        "title": title,
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": bool(gpu),
        "enable_tpu": False,
        "enable_internet": bool(internet),
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": kernel_sources,
        "model_sources": [],
        "docker_image": None,
    }
    # A null docker_image is rejected by the CLI schema; drop it.
    metadata.pop("docker_image", None)
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return kernel_dir


def push_and_wait(slug: str, code_path: Path, gpu: bool = True, internet: bool = False,
                  title: str | None = None, dataset_sources: list[str] | None = None,
                  kernel_sources: list[str] | None = None, timeout_s: int = 3600,
                  poll_s: int = 20, fetch_output: bool = True) -> dict:
    env = build_env()
    owner = discover_username(env)
    kernel_dir = write_kernel_dir(
        slug, code_path, title or slug, gpu, internet,
        dataset_sources or [], kernel_sources or [],
    )
    meta_path = kernel_dir / "kernel-metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["id"] = f"{owner}/{slug}"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[kaggle] pushing {owner}/{slug} (gpu={gpu}) ...")
    result = run_cli(["kernels", "push", "-p", str(kernel_dir)], env, timeout=900)
    print((result.stdout or "").strip()[-1200:])
    if result.returncode != 0:
        raise RuntimeError(f"kaggle kernels push failed: {(result.stderr or result.stdout) [-1500:]}")

    ref = f"{owner}/{slug}"
    deadline = time.time() + timeout_s
    status = "running"
    while time.time() < deadline:
        check = run_cli(["kernels", "status", ref], env, timeout=120)
        text = ((check.stdout or "") + (check.stderr or "")).strip()
        print(f"[kaggle] status: {text[:200]}")
        lowered = text.lower()
        if "complete" in lowered:
            status = "complete"
            break
        if "error" in lowered or "failed" in lowered or "cancel" in lowered:
            status = "error"
            break
        time.sleep(poll_s)

    output_dir = OUTPUTS_DIR / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    if fetch_output and status == "complete":
        print(f"[kaggle] downloading output to {output_dir}")
        out = run_cli(["kernels", "output", ref, "-p", str(output_dir)], env, timeout=900)
        print((out.stdout or "").strip()[-1000:])
        downloaded = [str(p) for p in output_dir.rglob("*") if p.is_file()]

    summary = {
        "ref": ref,
        "slug": slug,
        "status": status,
        "gpu": gpu,
        "internet": internet,
        "dataset_sources": dataset_sources or [],
        "kernel_sources": kernel_sources or [],
        "kernel_dir": str(kernel_dir),
        "output_dir": str(output_dir),
        "downloaded": downloaded,
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[kaggle] summary:", json.dumps(summary, indent=2)[:1500])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--code", required=True, help="path to the Python script to run as the kernel")
    parser.add_argument("--slug", required=True, help="kernel slug (lowercase letters, digits, hyphens)")
    parser.add_argument("--title", default=None)
    parser.add_argument("--gpu", action="store_true", help="enable a GPU (recommended)")
    parser.add_argument("--no-internet", action="store_true")
    parser.add_argument("--dataset", action="append", default=[], help="dataset source ref (repeatable)")
    parser.add_argument("--kernel-source", action="append", default=[], help="kernel source ref (repeatable)")
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args(argv)
    summary = push_and_wait(
        slug=args.slug,
        code_path=Path(args.code),
        gpu=args.gpu,
        internet=not args.no_internet,
        title=args.title,
        dataset_sources=args.dataset,
        kernel_sources=args.kernel_source,
        timeout_s=args.timeout_s,
        fetch_output=not args.no_fetch,
    )
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
