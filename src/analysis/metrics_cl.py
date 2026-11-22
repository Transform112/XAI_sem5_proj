"""Continual-learning evaluation metrics (stage S5).

Authority
---------
* ``docs/source_documents/Doc 2`` §S5 "Continual-learning evaluation" (binding):
  ``R[i, j]`` = test macro-F1 on period *j* after training through period *i*;
  final average F1 = ``mean_j R[T, j]``;
  forgetting of period j = ``F_j = max_{i<T} R[i, j] - R[T, j]`` (mean over
  ``j < T``); ``BWT = mean_{j<T} (R[T, j] - R[j, j])``; per-class F1 stored for
  every (checkpoint, period) pair for RQ3; macro-F1 only (not accuracy);
  the upper triangle of R (future periods) is never used for forgetting, and
  classes absent from a period are excluded from that period's macro-F1 with an
  explicit list.
* ``docs/05_EXPERIMENT_PROTOCOL.md`` §11: per-class later-forgetting
  ``F_{c,j}(i) = max_{k<=i} R[k,j,c] - R[i,j,c]`` (past checkpoints only) and
  the **pre-registered** binary later-drop rule (``drop >= delta``, default
  ``delta = 0.05``, "within the next update or final"; no post-hoc variants).
* ``docs/DECISIONS.md`` D-033: dual macro-F1 reporting — over all 10 classes AND
  over the supported set — with an explicit excluded-class list, never silently
  averaged.
* ``research/lit/claims_evidence_skeleton.md`` gap #3: the per-class F1 tensor
  format (npz keys + shapes) is fixed here and committed with a tiny fixture.

Conventions
-----------
* Periods are **1-based externally** (``M_i``, ``P_j``, as in the protocol) and
  **0-based internally** (array indices).
* Cells with ``j > i`` (a checkpoint evaluated on a not-yet-seen period) are
  legitimate **forward-evaluation** cells (Doc 2 §S5: ``R[i,j]`` "for all i and
  j"); they are recorded but never used by the forgetting statistic (Doc 2 §S5
  check). NaN means "not evaluated".
* ``R[i,j]`` is a *macro* over classes. When per-class values are present the
  macro is recomputed and cross-checked against the recorded value; a mismatch
  is reported by :meth:`RMetricTensor.validate` instead of being averaged away.
* The later-drop horizon is interpreted as the *next update or the final
  checkpoint* with the **largest** of those two drops triggering the label.
  The protocol phrase is ambiguous ("within the next update or final"); this
  module exposes ``horizon="next_or_final" | "any_later"`` and flags the
  ambiguity to the Lead (A3) rather than choosing silently.

No function here trains a model, touches data state, or writes to the real
results tree.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Frozen stream length (docs/05 §3: T = 5 chronological periods).
PROTOCOL_N_PERIODS: int = 5

#: Pre-registered later-drop threshold (docs/05 §11 default delta = 0.05).
DEFAULT_FORGETTING_DELTA: float = 0.05

#: epsilon used only to absorb floating-point representation error at the
#: delta boundary (``0.65 - 0.60`` must count as a 0.05 drop).
_DELTA_ATOL: float = 1e-12

_STRATEGY_KEYS = ("strategy", "update_strategy", "cl_strategy")
_MODEL_KEYS = ("model", "architecture", "arch", "model_name")
_SEED_KEYS = ("seed", "random_seed")
_CHECKPOINT_KEYS = ("checkpoint", "checkpoint_period", "checkpoint_idx", "i", "period_trained", "M_t", "trained_through")
_EVAL_PERIOD_KEYS = ("eval_period", "eval_period_idx", "period", "j", "test_period", "P_j")
_MACRO_KEYS = ("macro_f1", "test_macro_f1", "f1_macro", "macro_f1_test")
_PER_CLASS_KEYS = ("per_class_f1", "f1_per_class", "class_f1", "per_class_f1_test")
_N_TEST_KEYS = ("n_test", "n_test_edges", "n")
_EXCLUDED_KEYS = ("excluded_classes", "excluded", "classes_excluded")


class CLMetricsError(ValueError):
    """Raised on malformed S5 inputs (missing keys, duplicate cells, bad shapes)."""


class DuplicateCellError(CLMetricsError):
    """Raised when two records disagree for the same (strategy, model, seed, i, j)."""


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------
def _first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or value is None:
        raise CLMetricsError(f"{name} must be an integer, got {value!r}")
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise CLMetricsError(f"{name} must be an integer, got {value!r}") from exc
    if not as_float.is_integer():
        raise CLMetricsError(f"{name} must be an integer, got {value!r}")
    return int(as_float)


def _period_key_to_int(value: Any) -> int:
    """Parse a ``per_period`` key (``"1"``, ``1``, ``"P1"``, ``"period_1"``)."""
    try:
        return _as_int(value, "per_period key")
    except CLMetricsError:
        digits = "".join(character for character in str(value) if character.isdigit())
        if not digits:
            raise CLMetricsError(f"cannot parse per_period key {value!r} as a 1-based period index") from None
        return int(digits)


def _normalise_per_class(value: Any) -> dict[str, float] | None:
    """Accept ``{class: f1}`` or ``[{"class": ..., "f1": ...}, ...]``."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        result: dict[str, float] = {}
        for key, raw in value.items():
            try:
                result[str(key)] = float(raw)
            except (TypeError, ValueError) as exc:
                raise CLMetricsError(f"per_class_f1[{key!r}] is not numeric: {raw!r}") from exc
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for item in value:
            if not isinstance(item, Mapping):
                raise CLMetricsError(f"per-class F1 list entries must be mappings, got {type(item).__name__}")
            name = _first(item, ("class", "class_name", "label", "name"))
            score = _first(item, ("f1", "f1_score", "value"))
            if name is None or score is None:
                raise CLMetricsError(f"per-class F1 entry lacks class/f1 keys: {dict(item)}")
            result[str(name)] = float(score)
        return result
    raise CLMetricsError(f"per_class_f1 must be a mapping or a list, got {type(value).__name__}")


@dataclass(frozen=True)
class CLEvalRecord:
    """One cell of the CL evaluation: checkpoint ``M_checkpoint`` on period ``eval_period``.

    ``strategy`` / ``model`` / ``seed`` identify the run; ``checkpoint`` and
    ``eval_period`` are 1-based protocol indices (``i``, ``j``). ``macro_f1`` may
    be ``None`` when only per-class values were logged; the tensor then derives
    it from ``per_class_f1`` (minus ``excluded_classes``) per D-033.
    """

    strategy: str
    model: str
    seed: int
    checkpoint: int
    eval_period: int
    macro_f1: float | None = None
    per_class_f1: Mapping[str, float] | None = None
    n_test: int | None = None
    excluded_classes: tuple[str, ...] = ()
    manifest: str | None = None
    lane: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.strategy:
            raise CLMetricsError("strategy must not be empty")
        if not self.model:
            raise CLMetricsError("model must not be empty")
        if self.checkpoint < 1 or self.eval_period < 1:
            raise CLMetricsError(
                f"checkpoint/eval_period are 1-based protocol indices; got {self.checkpoint}/{self.eval_period}"
            )
        # NOTE (T19 cross-check): Doc 2 S5 defines R[i, j] "for all i and j", so a
        # checkpoint may legitimately be evaluated on a later period (forward /
        # zero-shot evaluation). Such upper-triangle cells are recorded but are
        # never used by the forgetting formula (see forgetting_per_period).
        if self.macro_f1 is not None and not np.isfinite(self.macro_f1):
            raise CLMetricsError(f"macro_f1 must be finite or None, got {self.macro_f1!r}")
        if self.per_class_f1 is not None:
            bad = {key: value for key, value in self.per_class_f1.items() if not np.isfinite(value)}
            if bad:
                raise CLMetricsError(f"per_class_f1 values must be finite; non-finite: {bad}")


def record_from_mapping(
    mapping: Mapping[str, Any],
    *,
    manifest: str | None = None,
    source: str | None = None,
    strategy: str | None = None,
    model: str | None = None,
    seed: int | None = None,
    checkpoint: int | None = None,
    eval_period: int | None = None,
) -> CLEvalRecord:
    """Build a :class:`CLEvalRecord` from a tolerant mapping (JSON result row).

    ``strategy``/``model``/``seed``/``checkpoint`` may be inherited from an
    enclosing payload via the keyword arguments (needed for ``per_period`` blocks).
    """
    merged = dict(mapping)
    resolved_strategy = strategy or _first(merged, _STRATEGY_KEYS)
    resolved_model = model or _first(merged, _MODEL_KEYS)
    resolved_seed = seed if seed is not None else _first(merged, _SEED_KEYS)
    resolved_checkpoint = checkpoint if checkpoint is not None else _first(merged, _CHECKPOINT_KEYS)
    resolved_eval_period = eval_period if eval_period is not None else _first(merged, _EVAL_PERIOD_KEYS)
    missing = [
        name
        for name, value in (
            ("strategy", resolved_strategy),
            ("model", resolved_model),
            ("seed", resolved_seed),
            ("checkpoint", resolved_checkpoint),
            ("eval_period", resolved_eval_period),
        )
        if value is None
    ]
    if missing:
        raise CLMetricsError(
            f"S5 result row is missing required field(s) {missing}; found keys: {sorted(merged)}"
        )
    macro_raw = _first(merged, _MACRO_KEYS)
    macro = None if macro_raw is None else float(macro_raw)
    n_test_raw = _first(merged, _N_TEST_KEYS)
    excluded_raw = _first(merged, _EXCLUDED_KEYS) or ()
    if isinstance(excluded_raw, str):
        excluded_raw = (excluded_raw,)
    return CLEvalRecord(
        strategy=str(resolved_strategy),
        model=str(resolved_model),
        seed=_as_int(resolved_seed, "seed"),
        checkpoint=_as_int(resolved_checkpoint, "checkpoint"),
        eval_period=_as_int(resolved_eval_period, "eval_period"),
        macro_f1=macro,
        per_class_f1=_normalise_per_class(_first(merged, _PER_CLASS_KEYS)),
        n_test=None if n_test_raw is None else _as_int(n_test_raw, "n_test"),
        excluded_classes=tuple(str(name) for name in excluded_raw),
        manifest=manifest,
        lane=_first(merged, ("lane",)),
        source=source,
    )


def records_from_payload(
    payload: Any,
    *,
    manifest: str | None = None,
    source: str | None = None,
    _inherited: Mapping[str, Any] | None = None,
) -> list[CLEvalRecord]:
    """Extract every S5 evaluation record from one JSON payload.

    Supported layouts (documented because no S4/S5 writer format is frozen yet —
    ambiguity A4 reported to the Lead):

    1. a list of result rows;
    2. ``{"records": [...]}`` / ``{"results": [...]}``;
    3. a per-checkpoint file with ``strategy/model/seed/checkpoint`` at the top and
       ``{"per_period": {"1": {...}, "2": {...}}}`` (also accepted as
       ``eval`` / ``evaluations`` / ``metrics.per_period``);
    4. a single-cell result with ``checkpoint`` and ``eval_period``;
    5. a run manifest whose ``metrics`` block contains any of the above.
    """
    inherited = dict(_inherited or {})
    if isinstance(payload, (list, tuple)):
        records: list[CLEvalRecord] = []
        for item in payload:
            records.extend(
                records_from_payload(item, manifest=manifest, source=source, _inherited=inherited)
            )
        return records
    if not isinstance(payload, Mapping):
        raise CLMetricsError(f"S5 payload must be a mapping or a list, got {type(payload).__name__}")

    for container_key in ("records", "results"):
        block = payload.get(container_key)
        if isinstance(block, (list, tuple)):
            return records_from_payload(block, manifest=manifest, source=source, _inherited=inherited)

    meta = dict(inherited)
    for key in _STRATEGY_KEYS + _MODEL_KEYS + _SEED_KEYS + ("lane",):
        if payload.get(key) is not None:
            meta[key] = payload[key]

    per_period = None
    for key in ("per_period", "evaluations", "eval", "per_period_metrics"):
        if isinstance(payload.get(key), Mapping):
            per_period = payload[key]
            break
    metrics = payload.get("metrics")
    if per_period is None and isinstance(metrics, Mapping):
        for key in ("per_period", "evaluations", "eval", "per_period_metrics"):
            if isinstance(metrics.get(key), Mapping):
                per_period = metrics[key]
                break
        if per_period is None and any(_first(metrics, keys) is not None for keys in (_MACRO_KEYS, _PER_CLASS_KEYS)):
            # manifest whose metrics block is one evaluation cell
            merged = dict(payload)
            merged.update(metrics)
            payload = merged

    checkpoint_raw = _first(payload, _CHECKPOINT_KEYS)
    if checkpoint_raw is not None and per_period is not None:
        checkpoint = _as_int(checkpoint_raw, "checkpoint")
        records = []
        for period_key, cell in per_period.items():
            if not isinstance(cell, Mapping):
                raise CLMetricsError(f"per_period[{period_key!r}] must be a mapping")
            records.append(
                record_from_mapping(
                    cell,
                    manifest=manifest,
                    source=source,
                    strategy=_first(meta, _STRATEGY_KEYS),
                    model=_first(meta, _MODEL_KEYS),
                    seed=_first(meta, _SEED_KEYS),
                    checkpoint=checkpoint,
                    eval_period=_period_key_to_int(period_key),
                )
            )
        return records

    if _first(payload, _CHECKPOINT_KEYS) is not None and _first(payload, _EVAL_PERIOD_KEYS) is not None:
        return [
            record_from_mapping(
                payload,
                manifest=manifest,
                source=source,
                strategy=_first(meta, _STRATEGY_KEYS),
                model=_first(meta, _MODEL_KEYS),
                seed=_first(meta, _SEED_KEYS),
            )
        ]

    if per_period is not None and _first(payload, _CHECKPOINT_KEYS) is None:
        raise CLMetricsError(
            "payload has per-period results but no 'checkpoint' field; cannot place R[i, j] "
            f"(keys: {sorted(payload)})"
        )
    raise CLMetricsError(
        "payload contains no recognisable S5 evaluation records; expected 'records'/'results', "
        f"a 'per_period' block with a 'checkpoint', or a single checkpoint/eval_period row (keys: {sorted(payload)})"
    )


def collect_records(source: Any, *, patterns: Sequence[str] | None = None) -> list[CLEvalRecord]:
    """Load S5 records from a record, a mapping, a JSON file or a run directory.

    A directory is scanned for run manifests / evaluation JSONs with the default
    patterns ``("**/manifest.json", "**/*eval*.json", "**/*result*.json")``;
    files that contain no S5 records are skipped (run manifests from other
    stages legitimately live in the same tree). A file passed explicitly must
    contain valid records.
    """
    if isinstance(source, CLEvalRecord):
        return [source]
    if isinstance(source, Mapping):
        return records_from_payload(source)
    if isinstance(source, (list, tuple)):
        records = []
        for item in source:
            records.extend(collect_records(item, patterns=patterns))
        return records
    path = Path(source)
    if path.is_file():
        return records_from_payload(_read_json(path), manifest=str(path), source=str(path))
    if path.is_dir():
        file_patterns = patterns or ("**/manifest.json", "**/*eval*.json", "**/*result*.json")
        files = sorted({candidate for pattern in file_patterns for candidate in path.glob(pattern) if candidate.is_file()})
        records: list[CLEvalRecord] = []
        for file_path in files:
            try:
                payload = _read_json(file_path)
            except CLMetricsError:
                continue
            try:
                records.extend(records_from_payload(payload, manifest=str(file_path), source=str(file_path)))
            except CLMetricsError:
                continue
        return records
    raise CLMetricsError(f"S5 source not found: {source}")


def _read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CLMetricsError(f"cannot read JSON {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Per-class helpers
# ---------------------------------------------------------------------------
def macro_f1_from_per_class(
    per_class_f1: Mapping[str, float],
    *,
    exclude_classes: Iterable[str] = (),
    nan_policy: str = "omit",
) -> float:
    """Macro-F1 over an explicit class list (D-033 dual reporting).

    ``nan_policy='raise'`` refuses to average when a class value is missing
    (safer for headline tables); ``'omit'`` skips NaN entries but never silently
    drops a *named* exclusion — exclusions are always explicit.
    """
    excluded = {str(name) for name in exclude_classes}
    values = np.array(
        [float(score) for name, score in per_class_f1.items() if name not in excluded],
        dtype=float,
    )
    if values.size == 0:
        return float("nan")
    if np.all(np.isnan(values)):
        return float("nan")
    if nan_policy == "raise" and np.any(np.isnan(values)):
        raise CLMetricsError("macro-F1 cannot be averaged: some class F1 values are NaN")
    if nan_policy not in {"omit", "raise"}:
        raise ValueError(f"nan_policy must be 'omit' or 'raise', got {nan_policy!r}")
    return float(np.nanmean(values))


# ---------------------------------------------------------------------------
# R matrix
# ---------------------------------------------------------------------------
class RMetricTensor:
    """``R[i, j]`` matrix + per-class tensor ``R_class[i, j, c]`` for one run.

    Parameters
    ----------
    R:
        ``(T, T)`` float array; ``R[i, j]`` = macro-F1 on period ``j`` after
        training through period ``i`` (0-based here). A cell is NaN only when it
        was not evaluated; ``j > i`` cells (a checkpoint evaluated on a later,
        unseen period) are allowed and recorded, but never enter the forgetting
        statistic (Doc 2 S5 check: "the upper triangle of R (future periods) is
        not used for forgetting").
    R_class:
        optional ``(T, T, C)`` float array of per-class F1, same indexing.
    classes:
        class-name tuple of length ``C`` (required when ``R_class`` is given).
    """

    def __init__(
        self,
        R: Any,
        *,
        R_class: Any | None = None,
        classes: Sequence[str] | None = None,
        strategy: str = "unknown",
        model: str = "unknown",
        seed: int | None = None,
        n_periods: int | None = None,
        lane: str | None = None,
    ) -> None:
        array = np.array(R, dtype=float, copy=True)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise CLMetricsError(f"R must be a square 2-D array, got shape {array.shape}")
        n = array.shape[0]
        if n_periods is not None and n != int(n_periods):
            raise CLMetricsError(f"R is {n}x{n} but n_periods={n_periods}")
        upper = np.triu(np.ones((n, n), dtype=bool), k=1)
        # Future-period evaluations are legitimate R cells (Doc 2 S5 "for all i and j");
        # they are excluded only from the forgetting formula below.
        self.R = array
        self.strategy = str(strategy)
        self.model = str(model)
        self.seed = seed
        self.lane = lane

        self.R_class: np.ndarray | None = None
        self.classes: tuple[str, ...] = tuple(classes or ())
        if R_class is not None:
            class_array = np.array(R_class, dtype=float, copy=True)
            if class_array.ndim != 3:
                raise CLMetricsError(f"R_class must be 3-D (T, T, C), got shape {class_array.shape}")
            if class_array.shape[:2] != (n, n):
                raise CLMetricsError(f"R_class leading shape {class_array.shape[:2]} != R shape {(n, n)}")
            if not self.classes:
                raise CLMetricsError("classes= is required when R_class is provided")
            if class_array.shape[2] != len(self.classes):
                raise CLMetricsError(
                    f"R_class has {class_array.shape[2]} classes but classes= lists {len(self.classes)}"
                )
            self.R_class = class_array  # future evaluations allowed; see class docstring

    # -- construction -------------------------------------------------------
    @classmethod
    def from_records(
        cls,
        records: Sequence[CLEvalRecord],
        *,
        strategy: str | None = None,
        model: str | None = None,
        seed: int | None = None,
        classes: Sequence[str] | None = None,
        n_periods: int | None = None,
        on_conflict: str = "raise",
    ) -> "RMetricTensor":
        """Build one run's tensor from records (one strategy/model/seed).

        ``on_conflict``: ``'raise'`` (default — duplicate cells must agree),
        ``'mean'`` (average seeds when ``seed`` is not fixed / aggregation) or
        ``'first'``.
        """
        if not records:
            raise CLMetricsError("cannot build an R matrix from zero records")
        combos = {(record.strategy, record.model, record.seed) for record in records}
        strategy, model, seed = _select_combo(combos, strategy=strategy, model=model, seed=seed)
        selected = [record for record in records if (record.strategy, record.model) == (strategy, model)]
        if seed is None:
            seed_values = {record.seed for record in selected}
            if len(seed_values) > 1 and on_conflict == "raise":
                raise CLMetricsError(
                    f"records span seeds {sorted(seed_values)}; pass seed= or use from_records_grouped()/aggregate_seeds()"
                )
        else:
            selected = [record for record in selected if record.seed == seed]

        n = int(n_periods or max(max(r.checkpoint, r.eval_period) for r in selected))
        if n < 1:
            raise CLMetricsError(f"n_periods must be >= 1, got {n}")
        if classes is None:
            class_names = sorted({name for record in selected if record.per_class_f1 for name in record.per_class_f1})
            classes = tuple(class_names)
        else:
            classes = tuple(classes)
        c = len(classes)
        class_index = {name: index for index, name in enumerate(classes)}

        R = np.full((n, n), np.nan, dtype=float)
        R_class = np.full((n, n, c), np.nan, dtype=float) if c else None
        seen: dict[tuple[int, int], list[float]] = {}
        seen_class: dict[tuple[int, int, str], list[float]] = {}
        for record in selected:
            i = record.checkpoint - 1
            j = record.eval_period - 1
            if i >= n or j >= n:
                raise CLMetricsError(
                    f"record M_{record.checkpoint}/P_{record.eval_period} exceeds n_periods={n}"
                )
            if record.macro_f1 is not None:
                seen.setdefault((i, j), []).append(float(record.macro_f1))
            if record.per_class_f1:
                for name, value in record.per_class_f1.items():
                    if name not in class_index:
                        if c:
                            raise CLMetricsError(
                                f"per-class F1 for unknown class {name!r}; classes={list(classes)}"
                            )
                        continue
                    seen_class.setdefault((i, j, name), []).append(float(value))

        for (i, j), values in seen.items():
            R[i, j] = _resolve_values(values, on_conflict, cell=f"M_{i + 1}/P_{j + 1}")
        if R_class is not None:
            for (i, j, name), values in seen_class.items():
                R_class[i, j, class_index[name]] = _resolve_values(
                    values, on_conflict, cell=f"M_{i + 1}/P_{j + 1}/{name}"
                )
        tensor = cls(
            R,
            R_class=R_class,
            classes=classes,
            strategy=strategy or "unknown",
            model=model or "unknown",
            seed=seed,
            n_periods=n,
        )
        # Fill missing macro values from the per-class tensor (never the other way round).
        if tensor.R_class is not None:
            for i in range(n):
                for j in range(i + 1):
                    if np.isnan(R[i, j]) and np.any(np.isfinite(R_class[i, j, :])):
                        R[i, j] = float(np.nanmean(R_class[i, j, :]))
        return tensor

    @classmethod
    def from_records_grouped(
        cls,
        records: Sequence[CLEvalRecord],
        **kwargs: Any,
    ) -> dict[tuple[str, str, int], "RMetricTensor"]:
        """Build one tensor per ``(strategy, model, seed)`` combination."""
        groups: dict[tuple[str, str, int], list[CLEvalRecord]] = {}
        for record in records:
            groups.setdefault((record.strategy, record.model, record.seed), []).append(record)
        return {
            key: cls.from_records(group, strategy=key[0], model=key[1], seed=key[2], **kwargs)
            for key, group in groups.items()
        }

    # -- properties ---------------------------------------------------------
    @property
    def n_periods(self) -> int:
        return self.R.shape[0]

    @property
    def n_seen_cells(self) -> int:
        """Number of defined ``(i, j)`` cells (lower triangle + diagonal)."""
        return int(np.isfinite(self.R).sum())

    # -- S5 metrics ---------------------------------------------------------
    def final_macro_f1(self, *, nan_policy: str = "raise") -> float:
        """Final average F1: ``mean_j R[T, j]`` (Doc 2 §S5)."""
        row = self.R[-1, :]
        if nan_policy == "raise" and np.any(np.isnan(row)):
            missing = [f"P{j + 1}" for j in range(self.n_periods) if np.isnan(row[j])]
            raise CLMetricsError(f"final mean F1 undefined: missing final-checkpoint evaluations on {missing}")
        return float(np.nanmean(row)) if np.any(np.isfinite(row)) else float("nan")

    def forgetting_per_period(self) -> np.ndarray:
        """``F_j = max_{i in [j, T-1)} R[i, j] - R[T, j]`` for ``j < T``; NaN on the last period.

        Only *past* checkpoints for period j (``i >= j``) enter the best-so-far
        term; forward-evaluation cells (``i < j``) are excluded (Doc 2 §S5 check:
        "the upper triangle of R (future periods) is not used for forgetting").
        """
        n = self.n_periods
        result = np.full(n, np.nan, dtype=float)
        for j in range(n - 1):
            # Only "past" checkpoints for period j (i >= j) enter the best-so-far
            # term: future-period cells R[i, j] with i < j are forward evaluations
            # and are excluded (Doc 2 S5 check).
            prior = self.R[j : n - 1, j]
            if not np.any(np.isfinite(prior)):
                continue
            if np.isnan(self.R[n - 1, j]):
                continue
            result[j] = float(np.nanmax(prior) - self.R[n - 1, j])
        return result

    def forgetting(self) -> float:
        """Mean forgetting over the first ``T-1`` periods (Doc 2 §S5)."""
        values = self.forgetting_per_period()[: self.n_periods - 1]
        if not np.any(np.isfinite(values)):
            return float("nan")
        return float(np.nanmean(values))

    def bwt_per_period(self) -> np.ndarray:
        """``BWT_j = R[T, j] - R[j, j]`` for ``j < T``; NaN on the last period."""
        n = self.n_periods
        result = np.full(n, np.nan, dtype=float)
        for j in range(n - 1):
            result[j] = self.R[n - 1, j] - self.R[j, j]
        return result

    def bwt(self) -> float:
        """Backward transfer ``mean_{j<T} (R[T,j] - R[j,j])`` (Doc 2 §S5)."""
        values = self.bwt_per_period()[: self.n_periods - 1]
        if not np.any(np.isfinite(values)):
            return float("nan")
        return float(np.nanmean(values))

    def forward_transfer_per_period(self) -> np.ndarray:  # pragma: no cover - convenience only
        """``R[j, j] - R[j-1, j]`` diagnostics (not part of the frozen S5 table)."""
        result = np.full(self.n_periods, np.nan, dtype=float)
        for j in range(1, self.n_periods):
            result[j] = self.R[j, j] - self.R[j - 1, j]
        return result

    # -- dual macro (D-033) -------------------------------------------------
    def class_macro_matrix(self, *, exclude_classes: Iterable[str] = (), nan_policy: str = "omit") -> np.ndarray:
        """Recompute the macro-F1 matrix from the per-class tensor with exclusions."""
        if self.R_class is None:
            raise CLMetricsError("class_macro_matrix needs the per-class F1 tensor (R_class)")
        excluded = {str(name) for name in exclude_classes}
        keep = [index for index, name in enumerate(self.classes) if name not in excluded]
        matrix = np.full((self.n_periods, self.n_periods), np.nan, dtype=float)
        if not keep:
            if nan_policy == "raise":
                raise CLMetricsError("class_macro_matrix excludes every class")
            return matrix
        for i in range(self.n_periods):
            for j in range(self.n_periods):
                values = self.R_class[i, j, keep]
                finite = values[np.isfinite(values)]
                if finite.size:
                    matrix[i, j] = float(np.mean(finite))
                elif nan_policy == "raise":
                    raise CLMetricsError(f"class macro matrix undefined at R[{i},{j}] (all class values NaN)")
        return matrix

    def final_macro_f1_by_class(self, *, exclude_classes: Iterable[str] = ()) -> float:
        """Final-checkpoint macro-F1 over a class subset (supported set, D-033)."""
        matrix = self.class_macro_matrix(exclude_classes=exclude_classes)
        return RMetricTensor(matrix, strategy=self.strategy, model=self.model, seed=self.seed).final_macro_f1()

    def final_f1_per_class(self) -> dict[str, float]:
        """Final-checkpoint per-class F1 averaged over the periods seen (RQ3 input)."""
        if self.R_class is None:
            raise CLMetricsError("final_f1_per_class needs the per-class F1 tensor (R_class)")
        axis_values = self.R_class[-1, :, :]  # (T, C)
        result: dict[str, float] = {}
        for index, name in enumerate(self.classes):
            column = axis_values[:, index]
            finite = column[np.isfinite(column)]
            result[name] = float(np.mean(finite)) if finite.size else float("nan")
        return result

    # -- validation ---------------------------------------------------------
    def validate(self, *, rtol: float = 1e-6, atol: float = 1e-9) -> list[str]:
        """Cross-check recorded macro-F1 against the per-class tensor.

        Returns a list of human-readable issues (empty = consistent). The convex
        bound check ``per-class min <= macro-F1 <= per-class max`` is part of it,
        because a macro average outside the class range signals a logging bug.
        """
        issues: list[str] = []
        n = self.n_periods
        if self.R_class is None:
            return issues
        for i in range(n):
            for j in range(n):
                per_class = self.R_class[i, j, :]
                finite = per_class[np.isfinite(per_class)]
                if finite.size == 0:
                    continue
                recomputed = float(np.mean(finite))
                recorded = self.R[i, j]
                if np.isnan(recorded):
                    issues.append(f"R[{i},{j}] recorded as NaN but {finite.size} per-class values exist")
                    continue
                if not np.isclose(recorded, recomputed, rtol=rtol, atol=atol):
                    issues.append(
                        f"R[{i},{j}]={recorded:.6f} != mean(per-class)={recomputed:.6f} "
                        f"(upper-triangle/NaN conventions or excluded classes may differ)"
                    )
                elif recorded < finite.min() - atol or recorded > finite.max() + atol:
                    issues.append(f"R[{i},{j}]={recorded:.6f} outside per-class range [{finite.min():.6f}, {finite.max():.6f}]")
        return issues

    # -- serialisation ------------------------------------------------------
    def to_long_rows(self) -> list[dict[str, Any]]:
        """One row per defined ``(i, j, class)`` for the CSV tensor representation."""
        if self.R_class is None:
            raise CLMetricsError("to_long_rows needs the per-class F1 tensor (R_class)")
        rows: list[dict[str, Any]] = []
        for i in range(self.n_periods):
            for j in range(self.n_periods):
                for index, class_name in enumerate(self.classes):
                    value = self.R_class[i, j, index]
                    if np.isnan(value):
                        continue
                    rows.append(
                        {
                            "strategy": self.strategy,
                            "model": self.model,
                            "seed": self.seed if self.seed is not None else "",
                            "checkpoint": i + 1,
                            "eval_period": j + 1,
                            "class": class_name,
                            "f1": float(value),
                        }
                    )
        return rows


def _select_combo(
    combos: set[tuple[str, str, Any]],
    *,
    strategy: str | None,
    model: str | None,
    seed: int | None,
) -> tuple[str, str, int | None]:
    if strategy is None and len({combo[0] for combo in combos}) == 1:
        strategy = next(iter(combos))[0]
    if model is None and len({combo[1] for combo in combos}) == 1:
        model = next(iter(combos))[1]
    if strategy is None or model is None:
        raise CLMetricsError(
            f"records span several runs; pass strategy=/model= (found {sorted(combos)[:8]})"
        )
    candidate_seeds = {combo[2] for combo in combos if combo[0] == strategy and combo[1] == model}
    if seed is None and len(candidate_seeds) > 1:
        # handled by the caller with an explicit error when on_conflict='raise'
        return strategy, model, None
    if seed is not None and seed not in candidate_seeds and candidate_seeds:
        raise CLMetricsError(f"seed={seed} not present for {strategy}/{model}; available: {sorted(candidate_seeds)}")
    return strategy, model, seed


def _resolve_values(values: Sequence[float], on_conflict: str, *, cell: str) -> float:
    unique = np.array(values, dtype=float)
    if unique.size == 1:
        return float(unique[0])
    if np.allclose(unique, unique[0], rtol=1e-9, atol=1e-12):
        return float(unique[0])
    if on_conflict == "raise":
        raise DuplicateCellError(
            f"conflicting values for cell {cell}: {sorted(set(float(v) for v in unique))}; "
            "duplicate S5 cells must agree (or aggregate seeds explicitly)"
        )
    if on_conflict == "mean":
        return float(np.mean(unique))
    if on_conflict == "first":
        return float(unique[0])
    raise ValueError(f"on_conflict must be 'raise', 'mean' or 'first', got {on_conflict!r}")


def aggregate_seeds(tensors: Sequence[RMetricTensor], *, require_same_run: bool = True) -> RMetricTensor:
    """Element-wise mean over several seeds (NaN-aware) for one strategy/model."""
    if not tensors:
        raise CLMetricsError("aggregate_seeds needs at least one tensor")
    first = tensors[0]
    if require_same_run:
        combos = {(tensor.strategy, tensor.model) for tensor in tensors}
        if len(combos) != 1:
            raise CLMetricsError(f"aggregate_seeds requires one strategy/model per call, got {sorted(combos)}")
    n = first.n_periods
    if any(tensor.n_periods != n for tensor in tensors):
        raise CLMetricsError("all tensors must share n_periods")
    R_stack = np.stack([tensor.R for tensor in tensors], axis=0)
    R = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(i + 1):
            column = R_stack[:, i, j]
            if np.any(np.isfinite(column)):
                R[i, j] = float(np.nanmean(column))
    R_class = None
    classes = first.classes
    if all(tensor.R_class is not None for tensor in tensors):
        if len({tensor.classes for tensor in tensors}) != 1:
            raise CLMetricsError("aggregate_seeds requires identical class orders")
        stack = np.stack([tensor.R_class for tensor in tensors], axis=0)  # (S, T, T, C)
        R_class = np.full((n, n, len(classes)), np.nan, dtype=float)
        for i in range(n):
            for j in range(i + 1):
                for c in range(len(classes)):
                    column = stack[:, i, j, c]
                    if np.any(np.isfinite(column)):
                        R_class[i, j, c] = float(np.nanmean(column))
    return RMetricTensor(
        R,
        R_class=R_class,
        classes=classes if R_class is not None else None,
        strategy=first.strategy,
        model=first.model,
        seed=None,
        n_periods=n,
    )


# ---------------------------------------------------------------------------
# Per-class F1 tensor persistence (skeleton gap #3 — format fixed here)
# ---------------------------------------------------------------------------
#: npz keys of ``results/text/tables/per_class_f1.npz`` (frozen format):
#: ``R_class`` (T,T,C float64, NaN above the diagonal), ``R_macro`` (T,T float64),
#: ``checkpoints`` (T int), ``eval_periods`` (T int), ``classes`` (T unicode),
#: ``meta`` (JSON string with strategy/model/seed/lane).
PER_CLASS_F1_NPZ_KEYS: tuple[str, ...] = ("R_class", "R_macro", "checkpoints", "eval_periods", "classes", "meta")

#: Columns of ``results/text/tables/per_class_f1.csv``.
PER_CLASS_F1_CSV_COLUMNS: tuple[str, ...] = (
    "strategy",
    "model",
    "seed",
    "checkpoint",
    "eval_period",
    "class",
    "f1",
)


def save_per_class_f1(tensor: RMetricTensor, npz_path: str | Path, csv_path: str | Path | None = None) -> list[Path]:
    """Write the frozen per-class F1 tensor format (npz) and optional long CSV."""
    if tensor.R_class is None:
        raise CLMetricsError("save_per_class_f1 needs the per-class F1 tensor (R_class)")
    npz_target = Path(npz_path)
    npz_target.parent.mkdir(parents=True, exist_ok=True)
    n = tensor.n_periods
    meta = {
        "format_version": 1,
        "strategy": tensor.strategy,
        "model": tensor.model,
        "seed": tensor.seed,
        "lane": tensor.lane,
        "n_periods": n,
        "columns_convention": "R[i,j,c] = F1 on period j after training through period i; NaN = not evaluated (future evaluations allowed, excluded from forgetting)",
        "protocol": "Doc 2 S5 / docs/05 §11; skeleton gap #3",
    }
    np.savez_compressed(
        npz_target,
        R_class=tensor.R_class,
        R_macro=tensor.R,
        checkpoints=np.arange(1, n + 1, dtype=np.int64),
        eval_periods=np.arange(1, n + 1, dtype=np.int64),
        classes=np.asarray(tensor.classes, dtype=str),
        meta=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    written = [npz_target]
    if csv_path is not None:
        csv_target = Path(csv_path)
        csv_target.parent.mkdir(parents=True, exist_ok=True)
        rows = tensor.to_long_rows()
        _write_csv(rows, PER_CLASS_F1_CSV_COLUMNS, csv_target)
        written.append(csv_target)
    return written


def load_per_class_f1(npz_path: str | Path) -> RMetricTensor:
    """Load a tensor written by :func:`save_per_class_f1`."""
    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in PER_CLASS_F1_NPZ_KEYS if key not in payload]
        if missing:
            raise CLMetricsError(f"{path} is missing npz keys {missing}")
        R_class = payload["R_class"]
        R_macro = payload["R_macro"]
        classes = tuple(str(name) for name in payload["classes"].tolist())
        meta = json.loads(str(payload["meta"]))
    return RMetricTensor(
        R_macro,
        R_class=R_class,
        classes=classes,
        strategy=meta.get("strategy", "unknown"),
        model=meta.get("model", "unknown"),
        seed=meta.get("seed"),
        n_periods=R_macro.shape[0],
        lane=meta.get("lane"),
    )


# ---------------------------------------------------------------------------
# Compatibility with the S5 writer (src/scripts/run_eval.py)
# ---------------------------------------------------------------------------
def load_cl_matrix_json(path: str | Path) -> RMetricTensor:
    """Load the engineer's ``results/text/tables/cl_matrix.json`` into an R tensor.

    The S5 writer (``src/scripts/run_eval.py``, T14) emits ``matrix.all_classes``
    (T x T; NaN = not evaluated, fully-populated forward cells for ``j > i`` when
    a checkpoint is evaluated on later periods), ``class_order`` and
    ``dual_macro_f1_per_cell[i][j].per_class_f1`` keyed by numeric class id.
    This adapter lets the analysis tables consume the real S5 artifact directly;
    the skeleton's ``per_class_f1.{npz,csv}`` files are then materialised from
    the returned tensor by :func:`analysis.tables.generate_per_class_f1`.
    """
    payload = _read_json(Path(path))
    if "matrix" not in payload or "all_classes" not in payload.get("matrix", {}):
        raise CLMetricsError(f"{path} is not an S5 cl_matrix.json payload (missing matrix.all_classes)")
    matrix = np.asarray(payload["matrix"]["all_classes"], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise CLMetricsError(f"matrix.all_classes must be square, got shape {matrix.shape}")
    n = matrix.shape[0]
    class_order = tuple(str(name) for name in payload.get("class_order") or ())
    R_class: np.ndarray | None = None
    if class_order:
        R_class = np.full((n, n, len(class_order)), np.nan, dtype=float)
        per_cell = payload.get("dual_macro_f1_per_cell") or {}
        for row_key, columns in per_cell.items():
            i = int(row_key) - 1
            if not 0 <= i < n:
                continue
            for column_key, cell in (columns or {}).items():
                j = int(column_key) - 1
                if not 0 <= j < n:
                    continue
                values = cell.get("per_class_f1") if isinstance(cell, Mapping) else None
                if not isinstance(values, Mapping):
                    continue
                for class_key, score in values.items():
                    index = int(class_key)
                    if 0 <= index < len(class_order):
                        R_class[i, j, index] = float(score)
    return RMetricTensor(
        matrix,
        R_class=R_class,
        classes=class_order if class_order else None,
        strategy=str(payload.get("strategy", "unknown")),
        model=str(payload.get("model", "unknown")),
        seed=payload.get("seed"),
        n_periods=n,
    )


def load_cl_metrics_csv(path: str | Path) -> pd.DataFrame:
    """Load the S5 writer's long-format ``cl_metrics.csv`` (all-classes rows).

    The S5 writer reports dual macro-F1 as two rows per run (``metric`` =
    ``all_classes`` / ``supported_set``); this helper keeps that schema intact
    and returns a DataFrame so callers can join it without re-deriving metrics.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    expected = {"strategy", "model", "seed", "metric", "final_average_macro_f1", "forgetting", "bwt"}
    missing = expected - set(frame.columns)
    if missing:
        raise CLMetricsError(f"{path} is not an S5 cl_metrics.csv (missing {sorted(missing)})")
    return frame


def _write_csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], path: Path) -> Path:
    import csv

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


# ---------------------------------------------------------------------------
# RQ3 forgetting labels (docs/05 §11 pre-registered rule)
# ---------------------------------------------------------------------------
def best_so_far_f1(R_class: Any) -> np.ndarray:
    """``max_{k<=i} R[k,j,c]`` for every checkpoint i (past checkpoints only)."""
    array = np.asarray(R_class, dtype=float)
    if array.ndim != 3:
        raise CLMetricsError(f"R_class must be 3-D (T,T,C), got shape {array.shape}")
    n = array.shape[0]
    result = np.full_like(array, np.nan)
    for i in range(n):
        block = array[: i + 1]  # (i+1, T, C)
        finite = np.isfinite(block)
        has_value = np.any(finite, axis=0)
        filled = np.where(finite, block, -np.inf)
        block_max = filled.max(axis=0)
        result[i] = np.where(has_value, block_max, np.nan)
    return result


def forgetting_fj(R_class: Any) -> np.ndarray:
    """``F_{c,j}(i) = max_{k<=i} R[k,j,c] - R[i,j,c]`` (docs/05 §11)."""
    array = np.asarray(R_class, dtype=float)
    return best_so_far_f1(array) - array


def later_drop_label_tensor(
    R_class: Any,
    *,
    delta: float = DEFAULT_FORGETTING_DELTA,
    horizon: str = "next_or_final",
) -> np.ndarray:
    """Binary "later drop" label per ``(i, j, class)`` (docs/05 §11).

    Pre-registered rule: a class at checkpoint *i* is labelled 1 when its F1 drops
    by at least ``delta`` at the next update **or** by the final checkpoint
    (``horizon='next_or_final'``, the default reading). ``horizon='any_later'``
    uses the maximum drop over every later checkpoint as a robustness variant.
    The last checkpoint has no future observation and yields NaN. No other
    variants are computed.
    """
    array = np.asarray(R_class, dtype=float)
    if array.ndim != 3:
        raise CLMetricsError(f"R_class must be 3-D (T,T,C), got shape {array.shape}")
    if not delta > 0:
        raise ValueError(f"delta must be positive, got {delta}")
    if horizon not in {"next_or_final", "any_later"}:
        raise ValueError(f"horizon must be 'next_or_final' or 'any_later', got {horizon!r}")
    n = array.shape[0]
    labels = np.full_like(array, np.nan)
    for i in range(n - 1):
        current = array[i]  # (T, C)
        if horizon == "next_or_final":
            candidates = [array[i + 1]]
            if i + 1 != n - 1:
                candidates.append(array[n - 1])
            drops = [current - future for future in candidates]
            drop = _nanmax_stack(drops)
        else:  # any_later
            drops = [current - array[k] for k in range(i + 1, n)]
            drop = _nanmax_stack(drops)
        valid = np.isfinite(current) & np.isfinite(drop)
        labels[i] = np.where(valid, (drop >= delta - _DELTA_ATOL).astype(float), np.nan)
    return labels


def _nanmax_stack(arrays: Sequence[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays, axis=0)
    finite = np.any(np.isfinite(stack), axis=0)
    result = np.full(stack.shape[1:], np.nan, dtype=float)
    if np.any(finite):
        result[finite] = np.nanmax(np.where(np.isfinite(stack), stack, -np.inf), axis=0)[finite]
    return result


def later_drop_rows(
    tensor: RMetricTensor,
    *,
    delta: float = DEFAULT_FORGETTING_DELTA,
    horizon: str = "next_or_final",
) -> list[dict[str, Any]]:
    """Long rows behind :func:`later_drop_label_tensor` (feeds ``rq3_forgetting_labels.csv``).

    ``future_checkpoint``/``future_f1`` identify the observation that triggered
    the label (the next update or the final checkpoint); ``drop`` is the signed
    F1 decrease that was compared against ``delta``.
    """
    if tensor.R_class is None:
        raise CLMetricsError("later_drop_rows needs the per-class F1 tensor (R_class)")
    array = tensor.R_class
    labels = later_drop_label_tensor(array, delta=delta, horizon=horizon)
    n = tensor.n_periods
    rows: list[dict[str, Any]] = []
    for i in range(n - 1):
        for j in range(i + 1):
            for c, class_name in enumerate(tensor.classes):
                label = labels[i, j, c]
                if np.isnan(label):
                    continue
                current = array[i, j, c]
                if horizon == "next_or_final":
                    future_indices = [i + 1]
                    if i + 1 != n - 1:
                        future_indices.append(n - 1)
                else:
                    future_indices = list(range(i + 1, n))
                drops = [current - array[k, j, c] for k in future_indices]
                best = int(np.nanargmax(drops)) if np.any(np.isfinite(drops)) else None
                if best is None:
                    continue
                rows.append(
                    {
                        "strategy": tensor.strategy,
                        "model": tensor.model,
                        "seed": tensor.seed if tensor.seed is not None else "",
                        "checkpoint": i + 1,
                        "eval_period": j + 1,
                        "class": class_name,
                        "f1": float(current),
                        "future_checkpoint": future_indices[best] + 1,
                        "future_f1": float(array[future_indices[best], j, c]),
                        "drop": float(drops[best]),
                        "label": int(label),
                        "delta": float(delta),
                        "horizon": horizon,
                    }
                )
    return rows


__all__ = [
    "CLMetricsError",
    "CLEvalRecord",
    "DEFAULT_FORGETTING_DELTA",
    "DuplicateCellError",
    "PER_CLASS_F1_CSV_COLUMNS",
    "PER_CLASS_F1_NPZ_KEYS",
    "PROTOCOL_N_PERIODS",
    "RMetricTensor",
    "aggregate_seeds",
    "best_so_far_f1",
    "collect_records",
    "forgetting_fj",
    "later_drop_label_tensor",
    "later_drop_rows",
    "load_cl_matrix_json",
    "load_cl_metrics_csv",
    "load_per_class_f1",
    "macro_f1_from_per_class",
    "record_from_mapping",
    "records_from_payload",
    "save_per_class_f1",
]
