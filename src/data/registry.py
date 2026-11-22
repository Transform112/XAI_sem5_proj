"""Dataset registry: load and validate per-dataset YAML specifications.

Why this module exists
----------------------
Every data stage needs the same facts about a dataset: which column is the
multiclass target, which columns are identities that must be dropped, which are
categorical, which heavy-tailed columns get ``log1p``, how many flows the
publisher reports. Duplicating those facts in code would guarantee drift between
the audit, preprocessing and training stages. They live in
``configs/datasets/`` instead:

* ``_common_<schema>.yaml`` holds the shared column contract of a dataset
  family (all four NF-v3 datasets share one 55-column schema),
* ``<Dataset>.yaml`` holds identity, provenance, published counts and the class
  order for one dataset, merged on top of the common file.

This module only reads and validates those files; it never touches the raw data,
so it is cheap enough to import everywhere.

Validated invariants
--------------------
* required column roles exist (``label_multiclass``; ``timestamp_start``,
  ``host_src``, ``host_dst`` for the nf_v3 schema),
* the leakage/drop list, categorical list, ``log1p`` list and numeric list are
  pairwise disjoint and, together with the role columns, cover the schema,
* every declared dataset file exists and parses as a mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.config import ConfigError, deep_merge, load_yaml

DATASETS_DIR = Path("configs") / "datasets"

#: Column roles every dataset spec must define (values are column names).
REQUIRED_ROLES_COMMON: tuple[str, ...] = ("label_multiclass",)

#: Additional roles required for the NF-v3 family.
REQUIRED_ROLES_NF_V3: tuple[str, ...] = (
    "timestamp_start",
    "host_src",
    "host_dst",
    "label_binary",
)

#: Feature lists that must exist for an edge-classification dataset.
FEATURE_LISTS: tuple[str, ...] = ("categorical", "log1p", "numeric")


class DatasetSpecError(RuntimeError):
    """Raised when a dataset specification is missing or internally inconsistent."""


# ---------------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------------
def _resolve_project_root(project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).expanduser().resolve()
    from src.utils.paths import find_project_root  # local import: avoids a cycle

    return find_project_root()


def _configs_root(
    project_root: str | Path | None,
    configs_root: str | Path | None,
) -> Path:
    if configs_root is not None:
        candidate = Path(configs_root).expanduser()
        if not candidate.is_absolute():
            candidate = _resolve_project_root(project_root) / candidate
        return candidate
    return _resolve_project_root(project_root) / "configs"


# ---------------------------------------------------------------------------
# Registry listing
# ---------------------------------------------------------------------------
def available_datasets(
    *,
    project_root: str | Path | None = None,
    configs_root: str | Path | None = None,
) -> list[str]:
    """Return the sorted dataset names declared under ``configs/datasets``."""
    directory = _configs_root(project_root, configs_root) / "datasets"
    if not directory.is_dir():
        raise DatasetSpecError(f"dataset registry directory not found: {directory}")
    names = sorted(
        path.stem
        for path in directory.glob("*.yaml")
        if not path.stem.startswith("_")
    )
    if not names:
        raise DatasetSpecError(f"no dataset specs found in {directory}")
    return names


# ---------------------------------------------------------------------------
# Spec object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    """Validated, merged view of one dataset specification.

    The raw merged mapping is kept in :attr:`data`; typed properties expose the
    fields stages use. Instances are immutable so stages cannot mutate shared
    registry state; use :func:`dataclasses.replace` to build variants in tests.
    """

    name: str
    role: str
    schema: str
    data: Mapping[str, Any] = field(repr=False, compare=False)
    source_path: Path | None = None
    common_path: Path | None = None
    published: Mapping[str, Any] = field(default_factory=dict)
    class_order: tuple[str, ...] | None = None

    # -- Generic access -----------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        """Read a nested key with ``a.b.c`` notation."""
        cursor: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cursor, Mapping) or part not in cursor:
                return default
            cursor = cursor[part]
        return cursor

    def column(self, role: str) -> str:
        """Return the column name for *role*; raises when undefined."""
        value = self.get(f"columns.{role}")
        if not isinstance(value, str) or not value:
            raise DatasetSpecError(
                f"dataset {self.name!r} does not declare the {role!r} column role"
            )
        return value

    def optional_column(self, role: str) -> str | None:
        value = self.get(f"columns.{role}")
        return value if isinstance(value, str) and value else None

    # -- Column groups ------------------------------------------------------
    def _list(self, key: str) -> tuple[str, ...]:
        value = self.get(f"columns.{key}", self.get(key, []))
        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise DatasetSpecError(f"columns.{key} in dataset {self.name!r} must be a list")
        return tuple(str(item) for item in value)

    @property
    def drop_features(self) -> tuple[str, ...]:
        return self._list("drop_features")

    @property
    def categorical(self) -> tuple[str, ...]:
        return self._list("categorical")

    @property
    def log1p(self) -> tuple[str, ...]:
        return self._list("log1p")

    @property
    def numeric(self) -> tuple[str, ...]:
        return self._list("numeric")

    @property
    def non_negative(self) -> tuple[str, ...]:
        return self._list("non_negative")

    @property
    def nf_v2_features(self) -> tuple[str, ...]:
        """The 43 original NetFlow-v2 feature columns (v3 schema reference)."""
        return self._list("nf_v2_features")

    @property
    def temporal_raw(self) -> tuple[str, ...]:
        """Raw timestamp columns of the v3 temporal group (sidecar, not features)."""
        return self._list("temporal_raw")

    @property
    def temporal_iat(self) -> tuple[str, ...]:
        """Inter-arrival-time columns of the v3 temporal group (legitimate features)."""
        return self._list("temporal_iat")

    @property
    def drop_reasons(self) -> dict[str, str]:
        """Machine-readable reason for every dropped column (feature manifest)."""
        value = self.get("columns.drop_reasons") or {}
        if not isinstance(value, Mapping):
            raise DatasetSpecError(f"columns.drop_reasons in dataset {self.name!r} must be a mapping")
        return {str(key): str(reason).strip() for key, reason in value.items()}

    @property
    def label_taxonomy(self) -> dict[str, Any] | None:
        """Documented fine->coarse label aggregation config (D-031), or None."""
        value = self.data.get("label_taxonomy")
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise DatasetSpecError(f"label_taxonomy in dataset {self.name!r} must be a mapping")
        return dict(value)

    @property
    def class_aliases(self) -> dict[str, str]:
        """Canonical (paper) class name -> raw CSV value, for documented aliasing."""
        value = self.data.get("class_aliases")
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise DatasetSpecError(f"class_aliases in dataset {self.name!r} must be a mapping")
        return {str(key): str(alias) for key, alias in value.items()}

    def canonical_class_map(self) -> dict[str, str]:
        """Raw CSV class value -> canonical paper class name (aliases applied)."""
        return {alias: canonical for canonical, alias in self.class_aliases.items()}

    @property
    def published_classes_count(self) -> int | None:
        value = self.published.get("classes_count") if isinstance(self.published, Mapping) else None
        return None if value is None else int(value)

    def feature_columns(self) -> tuple[str, ...]:
        """Ordered feature list declared by the spec (stable across runs)."""
        return tuple(dict.fromkeys([*self.categorical, *self.log1p, *self.numeric]))

    def all_columns(self) -> tuple[str, ...]:
        """Every physical column the spec expects, including role columns."""
        roles = [
            self.optional_column(role)
            for role in (
                "timestamp_start",
                "timestamp_end",
                "host_src",
                "port_src",
                "host_dst",
                "port_dst",
                "label_binary",
                "label_multiclass",
            )
        ]
        ordered = [column for column in roles if column]
        ordered.extend(self.feature_columns())
        ordered.extend(self.drop_features)
        return tuple(dict.fromkeys(ordered))

    @property
    def expected_csv_count(self) -> int:
        value = self.get("raw.expected_csv_count", 1)
        return int(value)

    @property
    def csv_glob(self) -> str:
        return str(self.get("raw.csv_glob", "**/*.csv"))

    @property
    def published_flows(self) -> int | None:
        value = self.published.get("flows") if isinstance(self.published, Mapping) else None
        return None if value is None else int(value)

    @property
    def published_classes(self) -> dict[str, int | None]:
        classes = self.published.get("classes") if isinstance(self.published, Mapping) else None
        if not isinstance(classes, Mapping):
            return {}
        return {str(name): (None if count is None else int(count)) for name, count in classes.items()}

    # -- Raw data location --------------------------------------------------
    def csv_files(self, raw_dir: str | Path) -> list[Path]:
        """Locate the CSV file(s) of this dataset under *raw_dir*."""
        directory = Path(raw_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"raw dataset directory not found: {directory}")
        files = sorted(path for path in directory.glob(self.csv_glob) if path.is_file())
        return files


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
def _validate_spec(merged: Mapping[str, Any], name: str, source: Path) -> None:
    columns = merged.get("columns")
    schema = str(merged.get("schema", ""))
    # Alternative-domain specs (e.g. Elliptic, node classification) may declare
    # no columns at all: they are only listed/verified by the identity stage.
    if not columns:
        if schema and schema != "nf_v3":
            return
        raise DatasetSpecError(f"dataset {name!r} ({source}) has no 'columns' mapping")
    if not isinstance(columns, Mapping):
        raise DatasetSpecError(f"dataset {name!r} ({source}): 'columns' must be a mapping")
    required = list(REQUIRED_ROLES_COMMON)
    if schema == "nf_v3":
        required.extend(REQUIRED_ROLES_NF_V3)
    for role in required:
        value = columns.get(role)
        if not isinstance(value, str) or not value:
            raise DatasetSpecError(
                f"dataset {name!r} ({source}) is missing required column role {role!r}"
            )
    # EDGE-classification datasets need feature lists; alternative domains may not.
    if schema == "nf_v3":
        groups = {key: tuple(merged.get("columns", {}).get(key) or ()) for key in FEATURE_LISTS}
        for key, values in groups.items():
            if not values:
                raise DatasetSpecError(f"dataset {name!r} has an empty columns.{key} list")
        overlap = set()
        seen: set[str] = set()
        for key, values in groups.items():
            duplicates = seen & set(values)
            if duplicates:
                raise DatasetSpecError(
                    f"dataset {name!r}: columns appear in more than one feature list: {sorted(duplicates)}"
                )
            seen.update(values)
            overlap.update(values)
        dropped = set(merged.get("columns", {}).get("drop_features") or ())
        collision = overlap & dropped
        if collision:
            raise DatasetSpecError(
                f"dataset {name!r}: columns both dropped and used as features: {sorted(collision)}"
            )

        # The published v3 feature contract: 43 NetFlow-v2 + 10 temporal = 53
        # feature columns, plus 2 label columns = 55 physical columns.
        columns = merged.get("columns", {})
        nf_v2 = tuple(columns.get("nf_v2_features") or ())
        temporal_raw = tuple(columns.get("temporal_raw") or ())
        temporal_iat = tuple(columns.get("temporal_iat") or ())
        if nf_v2 or temporal_raw or temporal_iat:
            published_features = [*nf_v2, *temporal_raw, *temporal_iat]
            if len(nf_v2) != 43:
                raise DatasetSpecError(
                    f"dataset {name!r}: expected 43 nf_v2_features, found {len(nf_v2)}"
                )
            if len(temporal_raw) + len(temporal_iat) != 10:
                raise DatasetSpecError(
                    f"dataset {name!r}: expected 10 temporal columns, found "
                    f"{len(temporal_raw) + len(temporal_iat)}"
                )
            if len(set(published_features)) != 53:
                raise DatasetSpecError(
                    f"dataset {name!r}: published feature columns are not 53 unique names"
                )
            if set(overlap) - set(published_features):
                raise DatasetSpecError(
                    f"dataset {name!r}: feature lists contain columns outside the published schema: "
                    f"{sorted(set(overlap) - set(published_features))}"
                )
            label_columns = {
                columns.get("label_binary"),
                columns.get("label_multiclass"),
            }
            outside = dropped - set(published_features) - label_columns
            if outside:
                raise DatasetSpecError(
                    f"dataset {name!r}: drop_features contains columns outside the schema: {sorted(outside)}"
                )
            physical = set(published_features) | {column for column in label_columns if column}
            declared = set(overlap) | dropped
            if physical - declared:
                raise DatasetSpecError(
                    f"dataset {name!r}: schema columns not classified as feature or dropped: "
                    f"{sorted(physical - declared)}"
                )


def load_dataset_spec(
    name: str,
    *,
    project_root: str | Path | None = None,
    configs_root: str | Path | None = None,
) -> DatasetSpec:
    """Load, merge and validate the specification for *name*.

    The dataset file is merged on top of ``_common_<schema>.yaml`` when the
    dataset declares a ``schema`` and that common file exists.
    """
    directory = _configs_root(project_root, configs_root) / "datasets"
    dataset_file = directory / f"{name}.yaml"
    if not dataset_file.is_file():
        try:
            available = ", ".join(available_datasets(project_root=project_root, configs_root=configs_root))
        except DatasetSpecError:
            available = "none"
        raise DatasetSpecError(
            f"unknown dataset {name!r}: {dataset_file} not found (available: {available})"
        ) from None

    try:
        data = load_yaml(dataset_file)
    except ConfigError as exc:
        raise DatasetSpecError(str(exc)) from exc

    schema = str(data.get("schema", ""))
    common_path: Path | None = None
    merged: dict[str, Any] = {}
    if schema:
        common_file = directory / f"_common_{schema}.yaml"
        if common_file.is_file():
            merged = dict(load_yaml(common_file))
            common_path = common_file
    merged = deep_merge(merged, data)
    merged.setdefault("name", name)
    merged["name"] = str(merged["name"])

    _validate_spec(merged, name, dataset_file)

    published = merged.get("published") or {}
    if not isinstance(published, Mapping):
        raise DatasetSpecError(f"dataset {name!r}: 'published' must be a mapping")
    class_order_raw = merged.get("class_order")
    if class_order_raw is not None:
        if not isinstance(class_order_raw, Sequence) or isinstance(class_order_raw, (str, bytes)):
            raise DatasetSpecError(f"dataset {name!r}: 'class_order' must be a list or null")
        class_order: tuple[str, ...] | None = tuple(str(item) for item in class_order_raw)
    else:
        class_order = None

    return DatasetSpec(
        name=merged["name"],
        role=str(merged.get("role", "unknown")),
        schema=schema,
        data=merged,
        source_path=dataset_file,
        common_path=common_path,
        published=dict(published),
        class_order=class_order,
    )


def resolve_dataset_names(
    requested: Sequence[str] | str | None,
    *,
    project_root: str | Path | None = None,
    configs_root: str | Path | None = None,
) -> list[str]:
    """Expand ``all``/``None``/comma-separated names into a validated list."""
    available = available_datasets(project_root=project_root, configs_root=configs_root)
    if requested is None or requested == "all" or requested == ["all"]:
        return available
    if isinstance(requested, str):
        names = [part.strip() for part in requested.split(",") if part.strip()]
    else:
        names = [str(part).strip() for part in requested if str(part).strip()]
    unknown = [name for name in names if name not in available]
    if unknown:
        raise DatasetSpecError(f"unknown dataset(s) {unknown}; available: {', '.join(available)}")
    return names


__all__ = [
    "DatasetSpec",
    "DatasetSpecError",
    "available_datasets",
    "load_dataset_spec",
    "resolve_dataset_names",
]
