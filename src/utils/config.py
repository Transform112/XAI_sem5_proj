"""YAML configuration loading, merging and hashing.

Design contract
---------------
* Every stage loads its configuration through :func:`load_config`.
* A stage config is layered on top of ``configs/base.yaml`` so global defaults
  (seed, dataset, determinism) live in exactly one place.
* CLI-style overrides use dotted keys (``--set data.max_rows=1000``) and are
  applied last, which makes ad-hoc runs reproducible because the resolved
  configuration is written verbatim into the run manifest.
* The configuration hash is computed over the *resolved* mapping (canonical
  JSON, sorted keys), so two runs with the same effective config share a hash
  regardless of file formatting or layer order.

No absolute paths are used here; callers resolve files with
:mod:`src.utils.paths` and pass a project root when loading relative paths.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

try:  # pyyaml is a hard dependency of every stage; fail with a clear message.
    import yaml
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "pyyaml is required (pip install -r platforms/requirements/common.txt)"
    ) from exc


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing, invalid or inconsistent."""


# ---------------------------------------------------------------------------
# Low level YAML helpers
# ---------------------------------------------------------------------------
def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML mapping from *path*; an empty file yields an empty mapping."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"configuration file not found: {file_path}")
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:  # pragma: no cover - malformed user file
        raise ConfigError(f"invalid YAML in {file_path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"top level of {file_path} must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* without mutating either input.

    Mappings merge key by key; every other value (including lists) is replaced
    by the override. This keeps lane files short: they only list what differs.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_override_value(raw: str) -> Any:
    """Parse a CLI override value with YAML semantics (``1``, ``true``, ``[a]``)."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(data: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Return a copy of *data* with dotted ``key.path=value`` overrides applied."""
    result = _copy_mapping(data)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must have the form key.path=value")
        dotted, raw_value = item.split("=", 1)
        keys = [part for part in dotted.strip().split(".") if part]
        if not keys:
            raise ConfigError(f"override {item!r} has an empty key")
        cursor: MutableMapping[str, Any] = result
        for key in keys[:-1]:
            existing = cursor.get(key)
            if existing is None:
                existing = {}
                cursor[key] = existing
            if not isinstance(existing, MutableMapping):
                raise ConfigError(
                    f"override {item!r} would replace non-mapping value at {key!r}"
                )
            cursor = existing
        cursor[keys[-1]] = _parse_override_value(raw_value)
    return result


def _copy_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            copied[key] = _copy_mapping(value)
        elif isinstance(value, list):
            copied[key] = list(value)
        else:
            copied[key] = value
    return copied


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def canonical_json(data: Mapping[str, Any]) -> str:
    """Deterministic JSON text for hashing and embedding in manifests."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def hash_config(data: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON rendering of *data*."""
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------
def resolve_config_path(
    config: str | os.PathLike[str],
    project_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a config reference against the project root (absolute allowed)."""
    path = Path(config).expanduser()
    if path.is_absolute():
        return path
    root = Path(project_root).resolve() if project_root is not None else _project_root()
    return (root / path).resolve()


def _project_root() -> Path:
    """Lazily import the path resolver to avoid a circular import."""
    from src.utils.paths import find_project_root

    return find_project_root()


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Resolved configuration plus provenance needed for a run manifest."""

    data: dict[str, Any]
    path: Path | None = None
    base_path: Path | None = None
    overrides: tuple[str, ...] = ()
    lane: str | None = None

    # Mapping conveniences --------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: object) -> bool:
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def keys(self):  # noqa: D401 - thin proxy
        return self.data.keys()

    def items(self):  # noqa: D401 - thin proxy
        return self.data.items()

    def section(self, name: str) -> dict[str, Any]:
        """Return a sub-mapping, raising a helpful error when it is missing."""
        value = self.data.get(name)
        if not isinstance(value, Mapping):
            raise ConfigError(f"config section {name!r} missing or not a mapping")
        return dict(value)

    def to_dict(self) -> dict[str, Any]:
        return _copy_mapping(self.data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config(path={self.path}, sha256={self.sha256[:12]})"

    # Provenance -------------------------------------------------------------
    @property
    def sha256(self) -> str:
        return hash_config(self.data)

    def provenance(self) -> dict[str, Any]:
        """Small JSON-ready description embedded in manifests."""
        return {
            "path": None if self.path is None else str(self.path),
            "base_path": None if self.base_path is None else str(self.base_path),
            "sha256": self.sha256,
            "overrides": list(self.overrides),
            "lane": self.lane,
            "content": _copy_mapping(self.data),
        }


def load_config(
    config: str | os.PathLike[str] | Mapping[str, Any] | Config | None = None,
    *,
    base: str | os.PathLike[str] | None = "configs/base.yaml",
    overrides: Sequence[str] = (),
    project_root: str | os.PathLike[str] | None = None,
    lane: str | None = None,
) -> Config:
    """Load, layer, override and hash a stage configuration.

    Parameters
    ----------
    config:
        Stage config file (relative to *project_root*), a mapping, ``None`` for
        base only, or an already loaded :class:`Config` (returned unchanged
        apart from overrides).
    base:
        Global defaults file merged *under* the stage config. ``None`` skips it.
    overrides:
        Dotted ``key.path=value`` strings applied last.
    project_root:
        Root used to resolve relative config paths (auto-detected otherwise).
    lane:
        Optional lane name, recorded in provenance only.
    """
    if isinstance(config, Config):
        if overrides:
            merged = apply_overrides(config.data, overrides)
            return Config(
                data=merged,
                path=config.path,
                base_path=config.base_path,
                overrides=tuple(config.overrides) + tuple(overrides),
                lane=lane or config.lane,
            )
        return config

    root = Path(project_root).resolve() if project_root is not None else _project_root()
    merged: dict[str, Any] = {}
    base_path: Path | None = None
    stage_path: Path | None = None

    if base is not None:
        candidate = resolve_config_path(base, root)
        if candidate.is_file():
            merged = load_yaml(candidate)
            base_path = candidate
        elif isinstance(base, str) and base == "configs/base.yaml":
            # A stage config can be loaded outside the repository (tests,
            # Kaggle notebooks); base defaults are optional in that case.
            base_path = None
        else:
            raise ConfigError(f"base config not found: {candidate}")

    if isinstance(config, Mapping):
        merged = deep_merge(merged, config)
        stage_path = None
    elif config is not None:
        candidate = resolve_config_path(config, root)
        stage_data = load_yaml(candidate)
        merged = deep_merge(merged, stage_data)
        stage_path = candidate

    if overrides:
        merged = apply_overrides(merged, overrides)

    return Config(
        data=merged,
        path=stage_path,
        base_path=base_path,
        overrides=tuple(overrides),
        lane=lane,
    )


__all__ = [
    "Config",
    "ConfigError",
    "apply_overrides",
    "canonical_json",
    "deep_merge",
    "hash_config",
    "load_config",
    "load_yaml",
    "resolve_config_path",
]
