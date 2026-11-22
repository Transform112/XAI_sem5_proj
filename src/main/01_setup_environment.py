#!/usr/bin/env python
"""XDRIFT main pipeline - step 01: environment and dataset presence check.

This is a normal ``.py`` script, but it prints in notebook-cell style so its
stdout can be pasted into a Kaggle notebook/log.  It deliberately does NOT
download anything: dataset acquisition is a separate step.

What this step guarantees before the expensive stages run
----------------------------------------------------------
* Python, operating system and compute context are usable.
* The required scientific stack imports (and versions are recorded).
* CUDA is available when this is treated as the Kaggle GPU lane.
* The repository layout and path configuration are found.
* The primary raw CSV exists and has the expected header shape.
* Output directories are writable.

Exit codes
----------
0  ready (warnings are allowed)
1  required environment/configuration check failed
2  environment is usable, but the primary dataset CSV was not found

Usage (from the repository root)
--------------------------------
python src/main/01_setup_environment.py
python src/main/01_setup_environment.py --lane kaggle
python src/main/01_setup_environment.py --dataset NF-UNSW-NB15-v3 --allow-cpu
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Path contract -------------------------------------------------------------
# This file lives at <repo>/src/main/01_setup_environment.py, so parents[2] is
# the repository root. If the main scripts are moved deeper (for example to
# src/main/steps/), update parents[2] to the new depth; _discover_root() also
# searches cwd, cwd parents and XDRIFT_ROOT, so a wrong guess only makes the
# layout check noisier, not silently wrong. Keep path literals below relative
# to the discovered root instead of hard-coding an absolute machine path.

WIDTH = 78
REQUIRED_PACKAGES = (
    "torch",
    "torch_geometric",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "yaml",
    "networkx",
)
OPTIONAL_PACKAGES = ("tqdm", "seaborn", "requests", "httpx", "psutil")
EXPECTED_DATASET_COLUMNS = 55
KAGGLE_INPUT = Path("/kaggle/input")


class Reporter:
    """Collect checks and print notebook-like cells/lines.

    The reporter is deliberately pure stdout + in-memory state: it never writes
    the report itself. That keeps the checks reusable and makes the final JSON
    write a single, obvious place to change the report location.
    """

    def __init__(self) -> None:
        self.checks: list[dict[str, str]] = []
        self.started = time.perf_counter()

    def cell(self, number: int, total: int, title: str) -> None:
        print("\n" + "=" * WIDTH)
        print(f"CELL {number:02d}/{total:02d} | {title}")
        print("=" * WIDTH)

    def line(self, status: str, name: str, detail: str = "") -> None:
        status = status.upper()
        detail = f" {detail}" if detail else ""
        print(f"[{status:<4}] {name}{detail}")
        self.checks.append(
            {"name": name, "status": status.lower(), "detail": str(detail).strip()}
        )

    def ok(self, name: str, detail: str = "") -> None:
        self.line("OK", name, detail)

    def warn(self, name: str, detail: str = "") -> None:
        self.line("WARN", name, detail)

    def fail(self, name: str, detail: str = "") -> None:
        self.line("FAIL", name, detail)

    def info(self, name: str, detail: str = "") -> None:
        self.line("INFO", name, detail)

    def skip(self, name: str, detail: str = "") -> None:
        self.line("SKIP", name, detail)

    @property
    def errors(self) -> list[dict[str, str]]:
        return [c for c in self.checks if c["status"] == "fail"]

    @property
    def warnings(self) -> list[dict[str, str]]:
        return [c for c in self.checks if c["status"] == "warn"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _version(module: Any) -> str | None:
    value = getattr(module, "__version__", None)
    return str(value) if value is not None else None


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _ram_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / 1024**3, 2)
    except Exception:
        pass
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text(errors="ignore").splitlines():
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024**2, 2)
        except Exception:
            pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.ullTotalPhys / 1024**3, 2)
        except Exception:
            pass
    return None


def _is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or KAGGLE_INPUT.is_dir()


def _discover_root() -> Path:
    candidates = [ROOT, Path.cwd(), *Path.cwd().parents]
    if os.environ.get("XDRIFT_ROOT"):
        candidates.insert(0, Path(os.environ["XDRIFT_ROOT"]))
    seen: set[str] = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if str(candidate) in seen:
            continue
        seen.add(str(candidate))
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
    return ROOT


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
# Each check_* function owns one notebook cell. Checks only read the machine or
# the repository and return plain dictionaries; they never download, train or
# mutate the dataset. This is what makes step 01 safe to rerun on every machine.
def check_runtime(reporter: Reporter) -> dict[str, Any]:
    reporter.cell(1, 7, "RUNTIME AND COMPUTE CONTEXT")
    version = platform.python_version()
    if sys.version_info >= (3, 10):
        reporter.ok("python", f"{version} ({sys.executable})")
    else:
        reporter.fail("python", f"{version}; Python >= 3.10 is required")

    reporter.info("platform", f"{platform.system()} {platform.release()} ({platform.machine()})")
    if _is_kaggle():
        reporter.ok("kaggle", "Kaggle kernel environment detected")
    else:
        reporter.warn("kaggle", "not a Kaggle kernel; local/CPU execution")

    reporter.info("cpu", f"{os.cpu_count() or 'unknown'} logical cores")
    ram = _ram_gb()
    if ram is None:
        reporter.warn("ram", "could not determine system memory")
    else:
        reporter.info("ram", f"{ram:.2f} GiB")

    return {
        "python": version,
        "executable": sys.executable,
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_count": os.cpu_count(),
        "ram_gb": ram,
        "is_kaggle": _is_kaggle(),
        "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
    }


def check_packages(reporter: Reporter) -> dict[str, Any]:
    # Import each package in isolation so one missing dependency is reported as
    # one FAIL line instead of aborting the whole step before the summary.
    reporter.cell(2, 7, "SCIENTIFIC PYTHON STACK")
    required: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            reporter.fail(f"package:{name}", f"{type(exc).__name__}: {exc}")
            required[name] = {"ok": False, "version": None, "error": str(exc)}
        else:
            version = _version(module)
            reporter.ok(f"package:{name}", version or "(version unknown)")
            required[name] = {"ok": True, "version": version, "error": None}

    optional: dict[str, dict[str, Any]] = {}
    for name in OPTIONAL_PACKAGES:
        try:
            module = importlib.import_module(name)
        except Exception:
            optional[name] = {"ok": False, "version": None}
        else:
            optional[name] = {"ok": True, "version": _version(module)}
            reporter.info(f"optional:{name}", optional[name]["version"] or "(version unknown)")

    if "torch_geometric" in sys.modules:
        try:
            from torch_geometric.explain import Explainer, GNNExplainer  # noqa: F401

            reporter.ok("pyg_api", "torch_geometric.explain.Explainer/GNNExplainer importable")
        except Exception as exc:
            reporter.fail("pyg_api", f"{type(exc).__name__}: {exc}")
    else:
        reporter.warn("pyg_api", "not checked because torch_geometric failed to import")

    return {"required": required, "optional": optional}


def check_gpu(reporter: Reporter, packages: dict[str, Any], require_cuda: bool) -> dict[str, Any]:
    # Kaggle is the canonical GPU lane, so a Kaggle run requires CUDA unless the
    # caller passes --allow-cpu. Local runs only warn, which keeps debugging on a
    # laptop possible without pretending it is the headline lane.
    reporter.cell(3, 7, "GPU / CUDA")
    info: dict[str, Any] = {"required": require_cuda, "available": False}
    torch_ok = packages["required"].get("torch", {}).get("ok", False)
    if not torch_ok:
        if require_cuda:
            reporter.fail("cuda", "torch is unavailable, so CUDA cannot be used")
        else:
            reporter.warn("cuda", "torch is unavailable; GPU check skipped")
        return info

    torch = sys.modules["torch"]
    info["torch_cuda_build"] = getattr(torch.version, "cuda", None)
    try:
        cudnn_version = torch.backends.cudnn.version()
        info["cudnn"] = str(cudnn_version) if cudnn_version is not None else None
    except Exception:
        info["cudnn"] = None
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:
        info["error"] = str(exc)
        if require_cuda:
            reporter.fail("cuda", f"torch.cuda.is_available() raised {type(exc).__name__}: {exc}")
        else:
            reporter.warn("cuda", f"torch.cuda.is_available() raised {type(exc).__name__}: {exc}")
        return info

    info["available"] = available
    if not available:
        message = "torch was built without a usable CUDA device"
        if require_cuda:
            reporter.fail("cuda", message)
        else:
            reporter.warn("cuda", f"{message}; CPU fallback is allowed here")
        return info

    try:
        count = int(torch.cuda.device_count())
        name = torch.cuda.get_device_name(0)
        capability = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
        props = torch.cuda.get_device_properties(0)
        memory_gb = round(props.total_memory / 1024**3, 2)
        device = torch.device("cuda:0")
        left = torch.ones((64, 64), device=device)
        right = torch.ones((64, 64), device=device)
        result = left @ right
        torch.cuda.synchronize()
        smoke = bool(result[0, 0].item() == 64.0)
        info.update(
            {
                "device_count": count,
                "device_name": name,
                "capability": capability,
                "memory_gb": memory_gb,
                "matmul_smoke": smoke,
            }
        )
        reporter.ok("cuda", f"{name} ({memory_gb} GiB, compute {capability}, devices={count})")
        if not smoke:
            reporter.fail("cuda:matmul", "64x64 GPU matrix multiply returned an unexpected value")
    except Exception as exc:
        reporter.fail("cuda", f"GPU probes failed: {type(exc).__name__}: {exc}")
        info["error"] = str(exc)
    return info


def check_project(
    reporter: Reporter,
    root: Path,
    dataset: str,
    lane: str,
) -> tuple[dict[str, Any], Any, str | None, list[Path]]:
    # Path resolution is delegated to src.utils.paths when pyyaml is available.
    # A missing optional dependency is a warning here, not a hard failure: the
    # dataset check below still works from the conservative datasets/ fallback.
    reporter.cell(4, 7, "PROJECT LAYOUT AND PATHS")
    info: dict[str, Any] = {"root": str(root), "lane": lane}

    for required in ("src", "configs", "platforms"):
        path = root / required
        if path.is_dir():
            reporter.ok(f"layout:{required}", str(path))
        else:
            reporter.fail(f"layout:{required}", f"missing directory: {path}")

    for label, path in (
        ("config:audit", root / "configs" / "audit.yaml"),
        ("config:dataset", root / "configs" / "datasets" / f"{dataset}.yaml"),
    ):
        if path.is_file():
            reporter.ok(label, str(path))
        else:
            reporter.warn(label, f"missing file: {path}")

    paths = None
    path_error = None
    try:
        from src.utils.paths import load_paths

        paths = load_paths(lane=lane, project_root=root)
        reporter.ok("path_resolver", f"lane={paths.lane} ({paths.lane_source})")
        reporter.info("data_root", str(paths.data_root))
        reporter.info("results_root", str(paths.results_root))
    except Exception as exc:
        path_error = f"{type(exc).__name__}: {exc}"
        reporter.warn("path_resolver", f"{path_error}; using conservative fallbacks")
        fallback = root / "datasets"
        reporter.info("data_root", str(fallback))

    data_roots: list[Path] = []
    if paths is not None:
        data_roots.append(Path(paths.data_root))
    if os.environ.get("XDRIFT_DATA"):
        data_roots.append(Path(os.environ["XDRIFT_DATA"]))
    data_roots.append(root / "datasets")

    return info, paths, path_error, _unique_paths(data_roots)


def _dataset_candidates(dataset: str, data_roots: Iterable[Path]) -> list[Path]:
    # Candidate order is a deterministic search path, not a preference: local
    # datasets/raw first, then the path lane's data root, then the common Kaggle
    # input mount. If the repository layout changes, add the new location here
    # (and keep it relative to a data root) rather than hard-coding an absolute
    # path in the check.
    csv_name = f"{dataset}.csv"
    candidates: list[Path] = []
    for data_root in data_roots:
        root = Path(data_root)
        candidates.extend(
            [
                root / "raw" / dataset / csv_name,
                root / "raw" / csv_name,
                root / dataset / csv_name,
                root / csv_name,
            ]
        )

    if KAGGLE_INPUT.is_dir():
        for child in sorted(KAGGLE_INPUT.iterdir()):
            if child.is_file() and child.suffix.lower() == ".csv":
                if dataset.lower() in child.name.lower():
                    candidates.append(child)
                continue
            if not child.is_dir():
                continue
            candidates.extend(
                [
                    child / csv_name,
                    child / dataset / csv_name,
                    child / "raw" / dataset / csv_name,
                ]
            )
            try:
                candidates.extend(child.glob(f"*/{csv_name}"))
                candidates.extend(child.glob(f"*/*/{csv_name}"))
            except OSError:
                pass
    return _unique_paths(candidates)


def _inspect_csv(path: Path) -> dict[str, Any]:
    # Header-only inspection: this step must stay cheap even for the 550 MiB
    # raw CSV. Full row/column profiling belongs to the data-audit stage.
    info: dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [column.strip() for column in next(reader)]
        except StopIteration:
            header = []
    info["columns"] = len(header)
    info["header"] = header
    lowered = {column.lower() for column in header}
    info["has_label"] = "label" in lowered
    info["has_attack"] = "attack" in lowered
    return info


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_dataset(
    reporter: Reporter,
    dataset: str,
    data_roots: list[Path],
    explicit_csv: str | None,
    hash_dataset: bool,
) -> dict[str, Any]:
    # Presence only: no download, no parsing beyond the header. Processed
    # period tensors and split masks are reported when present but are not
    # required here, because this step is expected to run before the build step.
    reporter.cell(5, 7, f"DATASET PRESENCE: {dataset}")
    info: dict[str, Any] = {"name": dataset, "found": False, "csv": None, "searched": []}

    if explicit_csv:
        candidates = [Path(explicit_csv)]
        if not candidates[0].is_absolute():
            candidates[0] = ROOT / candidates[0]
    else:
        candidates = _dataset_candidates(dataset, data_roots)
    info["searched"] = [str(path) for path in candidates[:12]]

    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        reporter.fail("dataset:csv", f"not found for {dataset}")
        reporter.info("dataset:searched", "; ".join(str(path) for path in candidates[:6]))
    else:
        info["found"] = True
        info["csv"] = str(found)
        reporter.ok("dataset:csv", str(found))
        try:
            details = _inspect_csv(found)
            info.update(details)
            reporter.info(
                "dataset:size",
                f"{details['size_bytes'] / 1024**2:.1f} MiB; {details['columns']} columns",
            )
            if details["columns"] != EXPECTED_DATASET_COLUMNS:
                reporter.warn(
                    "dataset:columns",
                    f"expected {EXPECTED_DATASET_COLUMNS}, found {details['columns']}",
                )
            else:
                reporter.ok("dataset:columns", f"{details['columns']} columns")
            if not (details["has_label"] and details["has_attack"]):
                reporter.fail(
                    "dataset:schema",
                    "required Label/Attack columns are missing from the CSV header",
                )
            if hash_dataset:
                digest = _sha256(found)
                info["sha256"] = digest
                reporter.ok("dataset:sha256", digest)
        except Exception as exc:
            reporter.fail("dataset:read", f"{type(exc).__name__}: {exc}")

    processed = [data_root / "processed" / dataset / f"period_{index}.pt" for data_root in data_roots for index in range(1, 6)]
    split_dir = data_roots[0] / "splits" / dataset
    info["processed_periods_found"] = sum(1 for path in processed if path.is_file())
    info["splits_dir"] = str(split_dir)
    info["splits_found"] = split_dir.is_dir() and any(split_dir.iterdir())
    if info["processed_periods_found"] >= 5:
        reporter.ok("processed:periods", "5 period tensors found")
    else:
        reporter.warn(
            "processed:periods",
            f"{info['processed_periods_found']}/5 period tensors found (data build step not yet run)",
        )
    if info["splits_found"]:
        reporter.ok("splits", str(split_dir))
    else:
        reporter.warn("splits", f"no split files yet under {split_dir}")

    return info


def check_outputs(
    reporter: Reporter,
    root: Path,
    paths: Any,
    json_out: Path,
) -> dict[str, Any]:
    reporter.cell(6, 7, "OUTPUT DIRECTORIES AND DISK")
    results_root = Path(paths.results_root) if paths is not None else root / "results"
    env_dir = json_out.parent
    info: dict[str, Any] = {
        "results_root": str(results_root),
        "json_report": str(json_out),
        "env_dir": str(env_dir),
    }
    try:
        env_dir.mkdir(parents=True, exist_ok=True)
        probe = env_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        reporter.ok("writable:report", str(env_dir))
    except Exception as exc:
        reporter.fail("writable:report", f"{type(exc).__name__}: {exc}")

    try:
        usage = shutil.disk_usage(root)
        free_gb = round(usage.free / 1024**3, 2)
        info["free_disk_gb"] = free_gb
        if free_gb < 5:
            reporter.warn("disk", f"{free_gb:.2f} GiB free")
        else:
            reporter.ok("disk", f"{free_gb:.2f} GiB free")
    except Exception as exc:
        reporter.warn("disk", f"could not read disk usage: {type(exc).__name__}: {exc}")

    for label, path in (("results", results_root), ("datasets", root / "datasets")):
        if path.exists():
            reporter.info(label, str(path))
        else:
            reporter.warn(label, f"does not exist yet: {path}")
    return info


def resolve_json_out(value: str | None, root: Path, paths: Any) -> Path:
    # Precedence: explicit --json-out, then the resolved lane's results root,
    # then the conservative repo-relative results/text/env_checks fallback.
    # Change the default here if env reports move; nothing else assumes a path.
    if value:
        path = Path(value)
        return path if path.is_absolute() else root / path
    if paths is not None:
        return Path(paths.results_root) / "text" / "env_checks" / "env_check.json"
    return root / "results" / "text" / "env_checks" / "env_check.json"


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XDRIFT step 01: environment and dataset-presence check (no download).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes: 0 ready, 1 environment/configuration failure, "
            "2 primary dataset CSV missing."
        ),
    )
    parser.add_argument("--dataset", default="NF-UNSW-NB15-v3", help="dataset name")
    parser.add_argument("--data-root", default=None, help="override the datasets root")
    parser.add_argument("--csv", default=None, help="explicit raw CSV path")
    parser.add_argument("--lane", default=None, help="path lane (default: kaggle if detected, else cpu)")
    parser.add_argument("--json-out", default=None, help="JSON report path, relative to repo root")
    parser.add_argument("--require-cuda", action="store_true", help="fail if CUDA is unavailable")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="on Kaggle, allow CPU-only execution instead of requiring CUDA",
    )
    parser.add_argument(
        "--hash-dataset",
        action="store_true",
        help="also compute the raw CSV SHA-256 (slower; useful for provenance)",
    )
    args = parser.parse_args(argv)
    if args.require_cuda and args.allow_cpu:
        parser.error("--require-cuda and --allow-cpu are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = datetime.now(timezone.utc)
    reporter = Reporter()
    total = 7

    print("=" * WIDTH)
    print("XDRIFT MAIN PIPELINE | STEP 01")
    print("Environment setup and dataset presence check")
    print("=" * WIDTH)
    print(f"[INFO] cwd: {Path.cwd()}")
    print(f"[INFO] command: {' '.join(sys.argv)}")

    root = _discover_root()
    kaggle = _is_kaggle()
    lane = args.lane or os.environ.get("XDRIFT_LANE") or ("kaggle" if kaggle else "cpu")
    require_cuda = bool(args.require_cuda or (kaggle and not args.allow_cpu))
    if kaggle and not args.allow_cpu:
        print("[INFO] Kaggle lane: CUDA is required unless --allow-cpu is passed")
    elif not kaggle:
        print("[INFO] local lane: CUDA is optional (use --require-cuda to enforce)")

    if not (root / "src").is_dir():
        reporter.fail("project_root", f"src/ not found from {root}")
    else:
        reporter.ok("project_root", str(root))

    runtime = check_runtime(reporter)
    packages = check_packages(reporter)
    gpu = check_gpu(reporter, packages, require_cuda)
    project, paths, path_error, data_roots = check_project(reporter, root, args.dataset, lane)
    if args.data_root:
        explicit_root = Path(args.data_root)
        if not explicit_root.is_absolute():
            explicit_root = root / explicit_root
        data_roots = _unique_paths([explicit_root, *data_roots])
    json_out = resolve_json_out(args.json_out, root, paths)
    dataset = check_dataset(
        reporter, args.dataset, data_roots, args.csv, args.hash_dataset
    )
    outputs = check_outputs(reporter, root, paths, json_out)

    reporter.cell(total, total, "RESULT")
    # Exit-code aggregation: environment/config errors dominate (1); a usable
    # environment with a missing CSV is a distinct acquisition problem (2); a
    # dataset that exists but has wrong schema/read errors is a hard failure (1).
    env_errors = [c for c in reporter.errors if not c["name"].startswith("dataset:")]
    dataset_errors = [
        c
        for c in reporter.errors
        if c["name"].startswith("dataset:") and c["name"] not in {"dataset:csv"}
    ]
    if env_errors or dataset_errors:
        exit_code = 1
        status = "fail"
        verdict = "NOT READY: required environment/configuration/data checks failed"
    elif not dataset["found"]:
        exit_code = 2
        status = "fail"
        verdict = "DATASET MISSING: environment is usable, but the raw CSV was not found"
    elif reporter.warnings:
        exit_code = 0
        status = "warn"
        verdict = "READY WITH WARNINGS"
    else:
        exit_code = 0
        status = "pass"
        verdict = "READY"
    print(f"[{status.upper():<4}] {verdict}")
    print(
        f"[INFO] checks: {len(reporter.checks)} total, "
        f"{len(reporter.errors)} failed, {len(reporter.warnings)} warnings"
    )
    print(f"[INFO] JSON report: {json_out}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "step": "01_setup_environment",
        "status": status,
        "exit_code": exit_code,
        "started_at_utc": started.isoformat(),
        "duration_s": round(time.perf_counter() - reporter.started, 2),
        "command": " ".join(sys.argv),
        "runtime": runtime,
        "packages": packages,
        "gpu": gpu,
        "project": project,
        "paths": {
            "lane": lane,
            "data_roots": [str(path) for path in data_roots],
            "path_error": path_error,
            "json_report": str(json_out),
        },
        "dataset": dataset,
        "outputs": outputs,
        "checks": reporter.checks,
        "warnings": reporter.warnings,
        "errors": reporter.errors,
        "next_step": "Acquire/download the dataset separately, then run the next main script.",
    }
    try:
        write_report(json_out, report)
        reporter.ok("report:written", str(json_out))
    except Exception as exc:
        reporter.fail("report:written", f"{type(exc).__name__}: {exc}")
        exit_code = 1

    print("[INFO] Dataset download is intentionally NOT part of this step.")
    print("[INFO] Next: run the separate dataset-acquisition step, then the data audit.")
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[WARN] interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[FAIL] unhandled {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
