"""Portable path resolution for every XDRIFT entry point.

Why this module exists
----------------------
The study runs on at least four very different machines: the user's Windows PC
with an RTX 5060, a legacy Windows PC with a GTX 1050, Kaggle notebooks
(Linux), and CPU-only CI/analysis boxes. Hard-coded paths would make every
stage brittle, so every path is resolved here from:

1. a lane file ``configs/paths/<lane>.yaml`` (lane = ``default``, ``legacy``,
   ``rtx5060``, ``kaggle``, ``cpu``),
2. environment variables (``XDRIFT_DATA``, ``XDRIFT_RUNS``, ``XDRIFT_CACHE``,
   ``XDRIFT_RESULTS`` and ``XDRIFT_ROOT``), and
3. repo-relative defaults declared in the lane file's ``defaults`` mapping.

Resolution rules
----------------
* ``project_root: auto`` means the directory that contains ``platforms/`` and
  ``src/`` (and normally ``.git``), found by walking up from this file, the
  working directory or ``XDRIFT_ROOT``.
* A value of the form ``env:NAME`` reads environment variable ``NAME``; when it
  is unset the key's entry in ``defaults`` is used instead.
* Relative values are resolved against the project root, never the current
  working directory, so scripts behave identically after ``cd``.
* All derived locations (``raw/``, ``processed/``, ``splits/``,
  ``manifests/``, ...) are computed here and nowhere else.

The module intentionally performs no I/O beyond reading the lane YAML and
creating directories when :meth:`Paths.ensure_all` is called, so it is safe to
import from tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.utils.config import ConfigError, deep_merge, load_yaml

#: Lanes the project supports. ``default`` is the shared base, the others are
#: concrete execution environments. Adding a lane = adding a YAML file (no code).
LANES: tuple[str, ...] = ("default", "legacy", "rtx5060", "kaggle", "cpu")

#: Root keys that are resolved from ``env:NAME`` + ``defaults``.
ROOT_KEYS: tuple[str, ...] = (
    "data_root",
    "runs_root",
    "cache_root",
    "results_root",
)

#: Derived roots that may be overridden explicitly in the lane file.
OPTIONAL_ROOT_KEYS: tuple[str, ...] = (
    "configs_root",
    "audit_root",
    "explanations_root",
    "figures_root",
    "tables_root",
)

PATHS_DIR = Path("configs") / "paths"

#: Legacy location kept so an old checkout still resolves until the Lead's
#: ``platforms/configs/paths.default.yaml`` seed is retired.
LEGACY_PATH_FILES: tuple[Path, ...] = (Path("platforms") / "configs" / "paths.default.yaml",)

PROJECT_MARKERS: tuple[Path, ...] = (Path("platforms"), Path("src"))


class PathConfigError(RuntimeError):
    """Raised when a path configuration cannot be resolved unambiguously."""


# ---------------------------------------------------------------------------
# Project root discovery
# ---------------------------------------------------------------------------
def _is_project_root(candidate: Path) -> bool:
    if not candidate.is_dir():
        return False
    if not all((candidate / marker).is_dir() for marker in PROJECT_MARKERS):
        return False
    return (candidate / ".git").exists() or (candidate / PATHS_DIR).is_dir()


def _walk_up(start: Path) -> Path | None:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if _is_project_root(candidate):
            return candidate
    return None


def find_project_root(start: str | os.PathLike[str] | None = None) -> Path:
    """Locate the repository root, honouring ``XDRIFT_ROOT`` first.

    Raises :class:`PathConfigError` with actionable text when the layout cannot
    be found, instead of silently using a wrong directory.
    """
    env_root = os.environ.get("XDRIFT_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser()
        if not _is_project_root(candidate):
            raise PathConfigError(
                f"XDRIFT_ROOT={env_root!r} is not a project root "
                f"(expected directories: {', '.join(str(m) for m in PROJECT_MARKERS)})"
            )
        return candidate.resolve()

    starts: list[Path] = []
    if start is not None:
        starts.append(Path(start))
    starts.append(Path(__file__))
    starts.append(Path.cwd())
    for candidate_start in starts:
        found = _walk_up(candidate_start)
        if found is not None:
            return found
    raise PathConfigError(
        "cannot locate the project root; run from inside the repository or set XDRIFT_ROOT"
    )


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------
def _normalise(path: Path) -> Path:
    """Absolute, normalised path without requiring the target to exist."""
    return Path(os.path.normpath(str(path.expanduser())))


def _resolve_root_value(
    key: str,
    value: Any,
    defaults: Mapping[str, Any],
    project_root: Path,
    environ: Mapping[str, str],
) -> Path:
    """Resolve one root key following the ``env:NAME`` -> ``defaults`` chain."""
    if value is None:
        value = defaults.get(key)
    if value is None:
        raise PathConfigError(
            f"path key {key!r} is not configured and has no default in the lane file"
        )
    text = str(value)
    if text.startswith("env:"):
        env_name = text[len("env:") :].strip()
        env_value = environ.get(env_name)
        if env_value is None or env_value == "":
            env_value = defaults.get(key)
            if env_value is None:
                raise PathConfigError(
                    f"environment variable {env_name} is unset and {key!r} has no default"
                )
        text = str(env_value)
    path = Path(text)
    if not path.is_absolute():
        path = project_root / path
    return _normalise(path)


def _safe_name(name: str) -> str:
    """Reject dataset/lane names that could escape the configured roots."""
    cleaned = str(name).strip()
    if not cleaned or cleaned in {".", ".."}:
        raise PathConfigError(f"invalid path name: {name!r}")
    if any(sep in cleaned for sep in (os.sep, "/", "\\")) or ":" in cleaned:
        raise PathConfigError(f"invalid path name (separators not allowed): {name!r}")
    return cleaned


# ---------------------------------------------------------------------------
# Paths object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    """Resolved directory layout for one lane.

    Instances are immutable; use :func:`load_paths` to build one. The
    ``project_root`` plus the four configurable roots are the only state; every
    other location is derived, which keeps lane files minimal.
    """

    project_root: Path
    data_root: Path
    runs_root: Path
    cache_root: Path
    results_root: Path
    configs_root: Path
    audit_root: Path
    explanations_root: Path
    figures_root: Path
    tables_root: Path
    lane: str = "default"
    lane_source: str = "default"
    config_path: Path | None = None
    lane_env: dict[str, str] = field(default_factory=dict, hash=False)

    # Derived roots ---------------------------------------------------------
    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw"

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed"

    @property
    def splits_root(self) -> Path:
        return self.data_root / "splits"

    @property
    def graphs_root(self) -> Path:
        return self.data_root / "graphs"

    @property
    def manifests_root(self) -> Path:
        return self.runs_root / "manifests"

    @property
    def logs_root(self) -> Path:
        return self.runs_root / "logs"

    # Dataset-scoped helpers ------------------------------------------------
    def raw_dataset(self, dataset: str) -> Path:
        """Directory holding the extracted raw files of *dataset*."""
        return self.raw_root / _safe_name(dataset)

    def processed_dataset(self, dataset: str) -> Path:
        return self.processed_root / _safe_name(dataset)

    def dataset_splits(self, dataset: str) -> Path:
        return self.splits_root / _safe_name(dataset)

    def dataset_graphs(self, dataset: str) -> Path:
        return self.graphs_root / _safe_name(dataset)

    def audit_dataset(self, dataset: str) -> Path:
        return self.audit_root / _safe_name(dataset)

    def run_dir(self, *, strategy: str = "base", model: str = "model", seed: int = 0) -> Path:
        """Canonical checkpoint directory ``runs/<strategy>/<model>/seed<seed>``."""
        return self.runs_root / _safe_name(strategy) / _safe_name(model) / f"seed{int(seed)}"

    # Side effects ----------------------------------------------------------
    def ensure_all(self) -> "Paths":
        """Create every standard directory (idempotent)."""
        for directory in (
            self.data_root,
            self.raw_root,
            self.processed_root,
            self.splits_root,
            self.graphs_root,
            self.runs_root,
            self.manifests_root,
            self.logs_root,
            self.cache_root,
            self.results_root,
            self.audit_root,
            self.explanations_root,
            self.figures_root,
            self.tables_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def ensure(self, *directories: Path) -> tuple[Path, ...]:
        """Create the given directories (usually derived ones) and return them."""
        created: list[Path] = []
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            created.append(Path(directory))
        return tuple(created)

    # Serialisation ---------------------------------------------------------
    def as_dict(self) -> dict[str, str]:
        """Flat JSON-ready view used in manifests and logs."""
        return {
            "project_root": str(self.project_root),
            "data_root": str(self.data_root),
            "raw_root": str(self.raw_root),
            "processed_root": str(self.processed_root),
            "splits_root": str(self.splits_root),
            "graphs_root": str(self.graphs_root),
            "runs_root": str(self.runs_root),
            "manifests_root": str(self.manifests_root),
            "logs_root": str(self.logs_root),
            "cache_root": str(self.cache_root),
            "results_root": str(self.results_root),
            "audit_root": str(self.audit_root),
            "explanations_root": str(self.explanations_root),
            "figures_root": str(self.figures_root),
            "tables_root": str(self.tables_root),
            "configs_root": str(self.configs_root),
        }

    def to_env(self) -> dict[str, str]:
        """Environment variables that reproduce this layout in child processes."""
        env = {
            "XDRIFT_ROOT": str(self.project_root),
            "XDRIFT_LANE": self.lane,
            "XDRIFT_DATA": str(self.data_root),
            "XDRIFT_RUNS": str(self.runs_root),
            "XDRIFT_CACHE": str(self.cache_root),
            "XDRIFT_RESULTS": str(self.results_root),
        }
        if self.config_path is not None:
            env["XDRIFT_PATHS_CONFIG"] = str(self.config_path)
        env.update({str(key): str(value) for key, value in self.lane_env.items()})
        return env

    def relative(self, path: str | os.PathLike[str]) -> str:
        """Render *path* relative to the project root when possible (for logs)."""
        try:
            return str(Path(path).resolve().relative_to(self.project_root))
        except ValueError:
            return str(path)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Paths(lane={self.lane!r}, data_root={str(self.data_root)!r})"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _lane_file(project_root: Path, lane: str) -> Path:
    return project_root / PATHS_DIR / f"{lane}.yaml"


def _read_path_layers(
    project_root: Path,
    lane: str,
    config_path: str | os.PathLike[str] | None,
    environ: Mapping[str, str],
) -> tuple[dict[str, Any], Path | None]:
    """Return merged lane data and the file it came from."""
    explicit = config_path or environ.get("XDRIFT_PATHS_CONFIG")
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if not explicit_path.is_absolute():
            explicit_path = project_root / explicit_path
        if not explicit_path.is_file():
            raise PathConfigError(f"explicit paths config not found: {explicit_path}")
        return load_yaml(explicit_path), explicit_path

    lane_file = _lane_file(project_root, lane)
    default_file = _lane_file(project_root, "default")
    if not default_file.is_file():
        for legacy in LEGACY_PATH_FILES:
            candidate = project_root / legacy
            if candidate.is_file():
                default_file = candidate
                break

    if lane not in LANES and not lane_file.is_file():
        available = sorted(path.stem for path in (project_root / PATHS_DIR).glob("*.yaml"))
        raise PathConfigError(
            f"unknown lane {lane!r}; available lanes: {', '.join(available) or 'none'} "
            f"(expected {lane_file})"
        )
    if not lane_file.is_file():
        if lane == "default":
            raise PathConfigError(f"no paths configuration found under {project_root / PATHS_DIR}")
        raise PathConfigError(f"lane file not found: {lane_file}")

    data: dict[str, Any] = {}
    if default_file.is_file():
        data = load_yaml(default_file)
    if lane_file.resolve() != default_file.resolve():
        data = deep_merge(data, load_yaml(lane_file))
    return data, lane_file


def load_paths(
    lane: str | None = None,
    *,
    config_path: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    ensure: bool = False,
) -> Paths:
    """Resolve the directory layout for *lane*.

    Parameters
    ----------
    lane:
        Lane name; defaults to ``XDRIFT_LANE`` then ``default``.
    config_path:
        Explicit paths YAML (overrides the lane lookup).
    project_root:
        Repository root; auto-detected when omitted.
    environ:
        Environment mapping (tests pass a fake one; defaults to ``os.environ``).
    ensure:
        Create the standard directories before returning.
    """
    env = dict(os.environ if environ is None else environ)
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else find_project_root()
    )
    if not root.is_dir():
        raise PathConfigError(f"project root does not exist: {root}")

    resolved_lane = lane or env.get("XDRIFT_LANE") or "default"
    if lane:
        lane_source = "cli"
    elif env.get("XDRIFT_LANE"):
        lane_source = "env"
    else:
        lane_source = "default"
    data, used_file = _read_path_layers(root, resolved_lane, config_path, env)

    if str(data.get("project_root", "auto")) != "auto":
        declared = Path(str(data["project_root"]))
        root = _normalise(declared if declared.is_absolute() else root / declared)

    defaults = data.get("defaults")
    if defaults is not None and not isinstance(defaults, Mapping):
        raise PathConfigError("'defaults' in paths config must be a mapping")
    defaults = dict(defaults or {})

    data_root = _resolve_root_value("data_root", data.get("data_root"), defaults, root, env)
    runs_root = _resolve_root_value("runs_root", data.get("runs_root"), defaults, root, env)
    cache_root = _resolve_root_value("cache_root", data.get("cache_root"), defaults, root, env)
    results_root = _resolve_root_value(
        "results_root", data.get("results_root"), defaults, root, env
    )

    configs_root = root / "configs"
    if data.get("configs_root") is not None:
        configs_root = _resolve_root_value(
            "configs_root", data.get("configs_root"), defaults, root, env
        )

    def derived(key: str, default: Path) -> Path:
        if data.get(key) is None:
            return _normalise(default)
        return _resolve_root_value(key, data.get(key), defaults, root, env)

    lane_env_raw = data.get("env") or {}
    if not isinstance(lane_env_raw, Mapping):
        raise PathConfigError("'env' in paths config must be a mapping")
    lane_env = {str(k): str(v) for k, v in lane_env_raw.items()}

    paths = Paths(
        project_root=root,
        data_root=data_root,
        runs_root=runs_root,
        cache_root=cache_root,
        results_root=results_root,
        configs_root=configs_root,
        audit_root=derived("audit_root", results_root / "text" / "data_audit"),
        explanations_root=derived("explanations_root", results_root / "text" / "explanations"),
        figures_root=derived("figures_root", results_root / "images"),
        tables_root=derived("tables_root", results_root / "text" / "tables"),
        lane=resolved_lane,
        lane_source=lane_source,
        config_path=used_file,
        lane_env=lane_env,
    )
    if ensure:
        paths.ensure_all()
    return paths


def default_paths(**kwargs: Any) -> Paths:
    """Convenience wrapper used by tests and interactive sessions."""
    return load_paths(**kwargs)


class GateLaneError(RuntimeError):
    """Raised when a gate artifact would be produced without an explicit lane."""


def require_gate_lane(paths: Paths, *, allow_default_lane: bool = False) -> None:
    """T17/T18: gate runs must carry a real lane tag (cli or env), never 'default'."""
    if paths.lane_source == "default" and not allow_default_lane:
        raise GateLaneError(
            "lane is 'default': gate artifacts require an explicit lane. "
            "Launch via src/scripts/run_lane.ps1 -Lane <legacy|rtx5060|kaggle|cpu>, pass --lane, "
            "or set XDRIFT_LANE. Use --allow-default-lane only for local development."
        )


def gate_preflight(paths: Paths) -> dict[str, Any]:
    """Record lane + working-tree cleanliness for a gate artifact."""
    from src.utils.manifest import git_info

    git = git_info(paths.project_root)
    dirty = git.get("dirty")
    warnings: list[str] = []
    if dirty:
        warnings.append(
            "working tree is dirty: gate runs must start from a clean commit for a "
            "reconstructible code fingerprint (recorded as code.from_dirty_tree=true)"
        )
    return {
        "lane": paths.lane,
        "lane_source": paths.lane_source,
        "git_sha": git.get("sha"),
        "git_dirty": dirty,
        "git_clean_commit": (not bool(dirty)) if dirty is not None else None,
        "warnings": warnings,
    }


__all__ = [
    "GateLaneError",
    "LANES",
    "PathConfigError",
    "Paths",
    "default_paths",
    "find_project_root",
    "gate_preflight",
    "load_paths",
    "require_gate_lane",
]
