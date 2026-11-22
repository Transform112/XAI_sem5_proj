"""Run manifests: the reproducibility record for every executed stage.

Why this module exists
----------------------
The study is only defensible if every number can be traced to the exact code,
configuration, data and environment that produced it. Each stage therefore
writes a JSON manifest containing at least:

* git commit SHA (and whether the worktree was dirty) via the project's git
  executable (``XDRIFT_GIT`` or ``git`` on ``PATH``),
* the full resolved configuration and its SHA-256,
* the dataset identity (file paths relative to the data root, sizes, SHA-256),
* Python / library versions, host and GPU description,
* the seed, the command line, timestamps and duration,
* the output artifacts with sizes and hashes.

Usage::

    from src.utils.paths import load_paths
    from src.utils.manifest import RunManifest

    paths = load_paths(lane="legacy")
    with RunManifest("data_audit", paths=paths, config=cfg,
                     out_path=paths.audit_root / "manifest.json") as manifest:
        manifest.set_dataset("NF-UNSW-NB15-v3", paths.raw_dataset("NF-UNSW-NB15-v3"))
        ... work ...
        manifest.add_artifact(paths.audit_root / "scorecard.json")

The context manager writes the manifest on success *and* on failure (with
``status: failed`` and the exception text), so a crashed run is still auditable.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.utils.config import Config, canonical_json

SCHEMA_VERSION = 1

#: Packages whose versions matter for reproducibility (missing ones are null).
DEFAULT_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "pyyaml",
    "torch",
    "torch-geometric",
    "networkx",
    "matplotlib",
    "pytest",
)


# ---------------------------------------------------------------------------
# Small primitives
# ---------------------------------------------------------------------------
def utc_now() -> str:
    """Current UTC time as RFC 3339 with second precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Atomically write a JSON document (UTF-8, sorted keys, indented)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
    )
    try:
        with handle:
            handle.write(text)
        os.replace(handle.name, target)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return target


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _run(cmd: Sequence[str], cwd: str | os.PathLike[str] | None = None) -> tuple[int, str, str]:
    """Run a short diagnostic command; never raises on a non-zero exit."""
    try:
        completed = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return completed.returncode, completed.stdout.strip(), completed.stderr.strip()
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
        return 1, "", f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Environment discovery
# ---------------------------------------------------------------------------
def package_versions(names: Iterable[str] | None = None) -> dict[str, str | None]:
    """Installed versions of the packages listed in *names* (unknown -> null)."""
    result: dict[str, str | None] = {}
    for name in names or DEFAULT_PACKAGES:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
        except Exception:  # pragma: no cover - exotic package metadata
            result[name] = None
    return result


def _git_candidates() -> list[str]:
    """Candidate git executables, most explicit first.

    These are *probe* paths, never output paths: the manifest only needs one git
    binary to pin the commit SHA. ``XDRIFT_GIT`` and PATH always win; the
    platform well-known locations are a last resort so a gate artifact cannot
    end up with a null SHA just because git was missing from PATH (D-030).
    """
    candidates: list[str] = []
    explicit = os.environ.get("XDRIFT_GIT")
    if explicit:
        candidates.append(explicit)
    found = shutil.which("git")
    if found:
        candidates.append(found)
    if os.name == "nt":
        program_files = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]
        for base in program_files:
            if not base:
                continue
            candidates.extend(
                [
                    os.path.join(base, "Git", "bin", "git.exe"),
                    os.path.join(base, "Git", "cmd", "git.exe"),
                ]
            )
        local_app_data = os.environ.get("LocalAppData")
        if local_app_data:
            candidates.append(os.path.join(local_app_data, "Programs", "Git", "bin", "git.exe"))
    else:
        candidates.extend(["/usr/bin/git", "/usr/local/bin/git", "/opt/homebrew/bin/git"])
    return candidates


def git_executable() -> str | None:
    """Locate git: ``XDRIFT_GIT``, then PATH, then platform well-known locations."""
    for candidate in _git_candidates():
        if not candidate:
            continue
        if Path(candidate).is_file():
            return candidate
        if Path(candidate).name == candidate:  # bare command name
            code, _, _ = _run([candidate, "--version"])
            if code == 0:
                return candidate
    code, _, _ = _run(["git", "--version"])
    return "git" if code == 0 else None


def git_info(project_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Commit SHA / branch / dirty flag for the repository."""
    exe = git_executable()
    info: dict[str, Any] = {"available": exe is not None, "executable": exe, "sha": None}
    if exe is None:
        info["reason"] = "git executable not found; set XDRIFT_GIT to the git binary"
        return info
    cwd = str(project_root) if project_root is not None else None
    code, sha, err = _run([exe, "-C", cwd, "rev-parse", "HEAD"] if cwd else [exe, "rev-parse", "HEAD"])
    if code != 0:
        info["reason"] = f"rev-parse failed: {err}"
        return info
    info["sha"] = sha
    code, branch, _ = _run(
        [exe, "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"] if cwd else [exe, "rev-parse", "--abbrev-ref", "HEAD"]
    )
    info["branch"] = branch if code == 0 else None
    code, status, _ = _run(
        [exe, "-C", cwd, "status", "--porcelain"] if cwd else [exe, "status", "--porcelain"]
    )
    info["dirty"] = bool(status) if code == 0 else None
    code, commit_date, _ = _run(
        [exe, "-C", cwd, "log", "-1", "--format=%cI"] if cwd else [exe, "log", "-1", "--format=%cI"]
    )
    info["commit_date"] = commit_date if code == 0 else None
    return info


def gpu_info() -> list[dict[str, Any]]:
    """CUDA GPUs visible to torch; empty list when torch/CUDA is unavailable."""
    try:
        import torch  # noqa: PLC0415 - optional dependency, probed at runtime
    except Exception:
        return []
    try:
        if not torch.cuda.is_available():
            return []
        devices: list[dict[str, Any]] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": ".".join(str(part) for part in torch.cuda.get_device_capability(index)),
                    "total_memory_mb": int(props.total_memory / (1024 * 1024)),
                }
            )
        return devices
    except Exception:  # pragma: no cover - driver edge cases
        return []


def _total_ram_bytes() -> int | None:
    try:  # POSIX
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        pass
    try:  # Windows
        import ctypes

        class _MemoryStatus(ctypes.Structure):
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

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except Exception:  # pragma: no cover - non-Windows fallbacks
        pass
    return None


#: Source roots included in the code fingerprint (used when git is unavailable
#: or as an additional pin next to the commit SHA).
CODE_FINGERPRINT_ROOTS: tuple[str, ...] = ("src", "scripts", "configs", "platforms", "tests")
_CODE_SUFFIXES = {".py", ".yaml", ".yml", ".sh", ".ps1", ".txt"}


def code_fingerprint(project_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """SHA-256 over the tracked source/config files.

    A gate artifact must pin the exact code and config even when the checkout has
    no git metadata (ZIP transfer, Kaggle snapshot) or git cannot run. The
    fingerprint is stable across platforms because it hashes relative paths and
    file bytes in sorted order.
    """
    if project_root is None:
        from src.utils.paths import find_project_root

        root = find_project_root()
    else:
        root = Path(project_root)
    files: list[Path] = []
    for name in CODE_FINGERPRINT_ROOTS:
        base = root / name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in _CODE_SUFFIXES and "__pycache__" not in path.parts:
                files.append(path)
    files.sort(key=lambda path: str(path.relative_to(root)).replace("\\", "/"))
    digest = hashlib.sha256()
    for path in files:
        relative = str(path.relative_to(root)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "roots": list(CODE_FINGERPRINT_ROOTS),
        "note": "SHA-256 over sorted relative paths + file hashes; pins code/config when git is unavailable",
    }


def host_info() -> dict[str, Any]:
    """Portable host description (no personal identifiers beyond the hostname)."""
    ram_bytes = _total_ram_bytes()
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "total_ram_gb": None if ram_bytes is None else round(ram_bytes / (1024**3), 1),
        "hostname": platform.node(),
    }


def python_info() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "executable_basename": Path(sys.executable).name,
    }


def environment_snapshot(prefix: str = "XDRIFT_", environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Relevant environment variables (paths only, no secrets)."""
    env = os.environ if environ is None else environ
    return {key: str(value) for key, value in sorted(env.items()) if key.startswith(prefix)}


# ---------------------------------------------------------------------------
# Dataset fingerprints
# ---------------------------------------------------------------------------
def file_record(path: str | os.PathLike[str], *, base: str | os.PathLike[str] | None = None, hash_file: bool = True) -> dict[str, Any]:
    """Describe one file: path (relative to *base* when given), size, sha256."""
    file_path = Path(path)
    record: dict[str, Any] = {
        "path": str(file_path),
        "bytes": file_path.stat().st_size if file_path.is_file() else None,
        "sha256": sha256_file(file_path) if (hash_file and file_path.is_file()) else None,
    }
    if base is not None:
        try:
            record["relative_path"] = str(file_path.resolve().relative_to(Path(base).resolve()))
        except ValueError:
            record["relative_path"] = str(file_path)
    return record


def directory_fingerprint(
    directory: str | os.PathLike[str],
    *,
    patterns: Sequence[str] = ("*",),
    hash_files: bool = True,
) -> dict[str, Any]:
    """Fingerprint every regular file in *directory* (recursive) and combine it."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {root}")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    files = sorted({path for path in files if path.is_file()})
    records = [file_record(path, base=root, hash_file=hash_files) for path in files]
    combined = hashlib.sha256(
        canonical_json(
            [{key: record[key] for key in ("relative_path", "bytes", "sha256")} for record in records]
        ).encode("utf-8")
    ).hexdigest()
    return {
        "directory": str(root),
        "file_count": len(records),
        "total_bytes": sum(int(record["bytes"] or 0) for record in records),
        "files": records,
        "fingerprint_sha256": combined,
    }


# ---------------------------------------------------------------------------
# Manifest object
# ---------------------------------------------------------------------------
def _slugify(text: str, max_length: int = 60) -> str:
    cleaned = "".join(char if (char.isalnum() or char in "-_.") else "_" for char in str(text))
    cleaned = cleaned.strip("._-") or "run"
    return cleaned[:max_length]


def default_manifest_path(
    kind: str,
    *,
    paths: Any | None = None,
    config: Config | None = None,
    out_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the conventional manifest path for a stage."""
    if paths is None:
        from src.utils.paths import load_paths

        paths = load_paths()
    directory = Path(out_dir) if out_dir is not None else paths.manifests_root
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = config.path.stem if config is not None and config.path is not None else kind
    return directory / f"{stamp}_{_slugify(kind)}_{_slugify(suffix)}.json"


class RunManifest:
    """Build, maintains and writes the reproducibility record of one stage run."""

    schema_version = SCHEMA_VERSION

    def __init__(
        self,
        kind: str,
        *,
        paths: Any | None = None,
        config: Config | None = None,
        lane: str | None = None,
        seed: int | None = None,
        dataset: str | None = None,
        notes: str | None = None,
        out_path: str | os.PathLike[str] | None = None,
        include_env: bool = True,
        include_gpu: bool = True,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self.kind = kind
        self.paths = paths
        self.config = config
        self.out_path: Path | None = Path(out_path) if out_path is not None else None
        self._start_monotonic: float | None = None
        self._closed = False
        self.payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": kind,
            "status": "created",
            "created_at_utc": utc_now(),
            "started_at_utc": None,
            "finished_at_utc": None,
            "duration_s": None,
            "lane": lane or (paths.lane if paths is not None else None),
            "lane_source": getattr(paths, "lane_source", None) if paths is not None else None,
            "python": python_info(),
            "versions": package_versions(),
            "git": git_info(paths.project_root if paths is not None else None),
            "code": code_fingerprint(paths.project_root if paths is not None else None),
            "host": host_info(),
            "gpu": gpu_info() if include_gpu else [],
            "seed": seed,
            "dataset": {"name": dataset, "files": [], "fingerprint_sha256": None},
            "command": {"argv": list(sys.argv), "cwd": str(Path.cwd())},
            "artifacts": [],
            "notes": notes,
        }
        if paths is not None:
            self.payload["paths"] = paths.as_dict()
            self.payload["paths_config"] = str(paths.config_path) if paths.config_path else None
        if config is not None:
            self.payload["config"] = config.provenance()
        if include_env:
            self.payload["env"] = environment_snapshot()
            # Gate-grade env block (T17): always record the resolved XDRIFT_* layout
            # even when the launcher did not export it into the environment.
            if paths is not None:
                self.payload["env"].update(paths.to_env())
                self.payload["env"] = dict(sorted(self.payload["env"].items()))
        git = self.payload["git"]
        git["clean_commit"] = (not bool(git.get("dirty"))) if git.get("dirty") is not None else None
        code = self.payload["code"]
        code["from_dirty_tree"] = bool(git.get("dirty")) if git.get("dirty") is not None else None
        code["clean_commit"] = git.get("clean_commit")
        # Gate-grade lane precondition (T17/T18): refuse to write a gate manifest
        # without a real lane tag unless the explicit dev escape hatch is set.
        from src.utils.gates import enforce_gate_lane, gate_record

        enforce_gate_lane(kind, paths)
        self.payload["gate"] = gate_record(kind, paths)
        if extra:
            self.payload["extra"] = dict(extra)

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "RunManifest":
        self._start_monotonic = time.monotonic()
        self.payload["status"] = "running"
        self.payload["started_at_utc"] = utc_now()
        return self

    def finish(self, status: str = "success", *, error: str | None = None, extra: Mapping[str, Any] | None = None) -> "RunManifest":
        elapsed = None if self._start_monotonic is None else round(time.monotonic() - self._start_monotonic, 3)
        self.payload["status"] = status
        self.payload["finished_at_utc"] = utc_now()
        self.payload["duration_s"] = elapsed
        if error is not None:
            self.payload["error"] = error
        if extra:
            self.payload.setdefault("extra", {}).update(dict(extra))
        return self

    # -- recording ----------------------------------------------------------
    def set_lane(self, lane: str) -> "RunManifest":
        self.payload["lane"] = lane
        return self

    def set_seed(self, seed: int) -> "RunManifest":
        self.payload["seed"] = int(seed)
        return self

    def set_notes(self, notes: str) -> "RunManifest":
        self.payload["notes"] = notes
        return self

    def set_dataset(
        self,
        name: str,
        directory: str | os.PathLike[str] | None = None,
        *,
        patterns: Sequence[str] = ("*",),
        hash_files: bool = True,
    ) -> "RunManifest":
        """Record the dataset identity (recursive file hashes)."""
        block: dict[str, Any] = {"name": name, "files": [], "fingerprint_sha256": None}
        if directory is not None:
            fingerprint = directory_fingerprint(directory, patterns=patterns, hash_files=hash_files)
            block.update(
                {
                    "directory": fingerprint["directory"],
                    "file_count": fingerprint["file_count"],
                    "total_bytes": fingerprint["total_bytes"],
                    "files": fingerprint["files"],
                    "fingerprint_sha256": fingerprint["fingerprint_sha256"],
                }
            )
        self.payload["dataset"] = block
        return self

    def add_dataset_file(self, path: str | os.PathLike[str], *, hash_file: bool = True) -> "RunManifest":
        record = file_record(path, hash_file=hash_file)
        self.payload["dataset"].setdefault("files", []).append(record)
        return self

    def add_artifact(
        self,
        path: str | os.PathLike[str],
        *,
        role: str | None = None,
        hash_file: bool = True,
    ) -> "RunManifest":
        """Record one output file (or directory of outputs)."""
        target = Path(path)
        if target.is_dir():
            for child in sorted(child for child in target.rglob("*") if child.is_file()):
                self.add_artifact(child, role=role, hash_file=hash_file)
            return self
        record = file_record(target, hash_file=hash_file)
        if role is not None:
            record["role"] = role
        self.payload["artifacts"].append(record)
        return self

    def note(self, message: str) -> "RunManifest":
        self.payload.setdefault("messages", []).append({"at_utc": utc_now(), "message": str(message)})
        return self

    # -- writing ------------------------------------------------------------
    def resolve_out_path(self) -> Path:
        if self.out_path is not None:
            return self.out_path
        if self.paths is None:
            raise RuntimeError(
                "manifest has no out_path and no paths object; pass out_path=... explicitly"
            )
        self.out_path = default_manifest_path(self.kind, paths=self.paths, config=self.config)
        return self.out_path

    def write(self) -> Path:
        if self._closed:
            return self.resolve_out_path()
        self._closed = True
        return write_json(self.resolve_out_path(), self.payload)

    # -- context manager ----------------------------------------------------
    def __enter__(self) -> "RunManifest":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.finish(
                "failed",
                error="".join(traceback.format_exception_only(exc_type, exc)).strip(),
                extra={"traceback_tail": "".join(traceback.format_tb(tb))[-4000:]},
            )
        elif self.payload.get("status") == "running":
            self.finish("success")
        self.write()
        return False


__all__ = [
    "CODE_FINGERPRINT_ROOTS",
    "DEFAULT_PACKAGES",
    "RunManifest",
    "SCHEMA_VERSION",
    "code_fingerprint",
    "default_manifest_path",
    "directory_fingerprint",
    "environment_snapshot",
    "file_record",
    "git_executable",
    "git_info",
    "gpu_info",
    "host_info",
    "package_versions",
    "python_info",
    "read_json",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "utc_now",
    "write_json",
]
