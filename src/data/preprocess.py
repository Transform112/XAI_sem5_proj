"""Train-only preprocessing (Doc 2 S1).

Every transformation that learns statistics from data is fitted **only on the
period-1 training mask** and then applied unchanged to every later period, which
mimics deployment: the scaler, imputation values, categorical vocabularies and
missing indicators never see validation/test/future data.

Per column (classification comes from the feature manifest):
* categorical -> label encoding with a vocabulary fitted on period-1 train;
  unseen categories map to the reserved index ``unknown_category_index`` (0);
* ``log1p`` columns -> ``log1p(clip(x, clip_min, None))`` then median-impute
  then standardise (mean/std from the fitted training rows);
* ``numeric`` columns -> median-impute then standardise;
* if the fitted training missing rate exceeds ``missing_indicator_threshold``,
  a binary ``<column>_missing`` indicator is appended (also a train-fitted
  decision);
* standardized values are clipped to ``+/- clip_std`` for numerical safety (the
  clipping rule is recorded in the state).

The fitted state is a JSON document (``preprocessor_state.json``) so the scaler
is inspectable and reusable across lanes without pickling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.registry import DatasetSpec
from src.utils.logging import get_logger
from src.utils.manifest import write_json

LOGGER = get_logger("xdrift.preprocess")

DEFAULT_CONFIG: dict[str, Any] = {
    "categorical_encoding": "label",
    "unknown_category_index": 0,
    "impute_strategy": "median",
    "missing_indicator_threshold": 0.001,
    "log1p_clip_min": 0.0,
    "standardize": True,
    "clip_std": 10.0,
}

STATE_VERSION = 1


class Preprocessor:
    """Fit-once / transform-many feature pipeline with explicit state."""

    def __init__(self, spec: DatasetSpec, config: Mapping[str, Any] | None = None) -> None:
        self.spec = spec
        self.config = {**DEFAULT_CONFIG, **dict(config or {})}
        self._fitted = False
        self.state: dict[str, Any] = {}
        self._feature_columns: list[str] = []
        self._feature_names: list[str] = []

    # -- fitting -----------------------------------------------------------
    def fit(self, frame: pd.DataFrame) -> "Preprocessor":
        """Fit all learned statistics on *frame* (must be period-1 train rows)."""
        if len(frame) == 0:
            raise ValueError("cannot fit the preprocessor on an empty frame")
        feature_columns = list(self.spec.feature_columns())
        missing = set(feature_columns) - set(frame.columns)
        if missing:
            raise KeyError(f"fit frame is missing feature columns: {sorted(missing)}")

        columns: dict[str, Any] = {}
        categorical: dict[str, Any] = {}
        indicators: dict[str, Any] = {}
        for column in feature_columns:
            is_categorical = column in self.spec.categorical
            if is_categorical:
                values = frame[column].astype("string")
                vocab_values = sorted({str(value) for value in values.dropna().unique().tolist()})
                unknown_index = int(self.config["unknown_category_index"])
                vocab = {"<unknown>": unknown_index}
                for offset, value in enumerate(vocab_values, start=unknown_index + 1):
                    vocab[value] = offset
                categorical[column] = {
                    "vocab": vocab,
                    "n_categories": len(vocab_values),
                    "unknown_index": unknown_index,
                }
                columns[column] = {
                    "transform": "label_encoding",
                    "n_train": int(len(frame)),
                }
                continue

            raw = pd.to_numeric(frame[column], errors="coerce")
            if column in self.spec.log1p:
                transformed = np.log1p(np.clip(raw.to_numpy(dtype="float64"), float(self.config["log1p_clip_min"]), None))
                transform = "log1p+standardize"
            else:
                transformed = raw.to_numpy(dtype="float64")
                transform = "standardize"
            missing_count = int(np.isnan(transformed).sum())
            missing_rate = missing_count / len(frame) if len(frame) else 0.0
            impute_value = float(np.nanmedian(transformed)) if missing_count < len(transformed) else 0.0
            if not np.isfinite(impute_value):
                impute_value = 0.0
            filled = np.where(np.isnan(transformed), impute_value, transformed)
            if self.config.get("standardize", True):
                mean = float(np.mean(filled))
                std = float(np.std(filled, ddof=0))
                if not np.isfinite(std) or std == 0.0:
                    std = 1.0
            else:
                mean, std = 0.0, 1.0
            columns[column] = {
                "transform": transform,
                "n_train": int(len(frame)),
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "impute_value": impute_value,
                "mean": mean,
                "std": std,
            }
            if missing_rate > float(self.config["missing_indicator_threshold"]):
                indicators[column] = f"{column}_missing"
                columns[column]["missing_indicator"] = indicators[column]

        feature_names = [column for column in feature_columns]
        feature_names += [indicators[column] for column in feature_columns if column in indicators]
        self.state = {
            "schema_version": STATE_VERSION,
            "dataset": self.spec.name,
            "feature_columns": feature_columns,
            "feature_names": feature_names,
            "config": dict(self.config),
            "columns": columns,
            "categorical": categorical,
            "missing_indicators": indicators,
            "fit": {"n_rows": int(len(frame)), "note": "period-1 train mask only"},
        }
        self._feature_columns = feature_columns
        self._feature_names = feature_names
        self._fitted = True
        LOGGER.info(
            "preprocessor fitted on %s rows: %s features (%s missing indicators)",
            f"{len(frame):,}",
            len(feature_names),
            len(indicators),
        )
        return self

    # -- transforming ------------------------------------------------------
    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply the fitted state to *frame*; returns ``(n_rows, n_features)``."""
        if not self._fitted:
            raise RuntimeError("Preprocessor.transform called before fit")
        missing = set(self._feature_columns) - set(frame.columns)
        if missing:
            raise KeyError(f"transform frame is missing feature columns: {sorted(missing)}")
        n_rows = len(frame)
        matrix = np.empty((n_rows, len(self._feature_names)), dtype="float64")
        clip_std = float(self.config.get("clip_std", 0.0) or 0.0)

        for index, column in enumerate(self._feature_columns):
            column_state = self.state["columns"][column]
            if column_state["transform"] == "label_encoding":
                vocab = self.state["categorical"][column]["vocab"]
                unknown = int(self.state["categorical"][column]["unknown_index"])
                values = frame[column].astype("string")
                encoded = values.map(lambda value: vocab.get(str(value), unknown) if pd.notna(value) else unknown)
                matrix[:, index] = encoded.to_numpy(dtype="float64")
                continue
            raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
            if column_state["transform"] == "log1p+standardize":
                raw = np.log1p(np.clip(raw, float(self.config["log1p_clip_min"]), None))
            filled = np.where(np.isnan(raw), float(column_state["impute_value"]), raw)
            standardized = (filled - float(column_state["mean"])) / float(column_state["std"])
            if clip_std > 0:
                standardized = np.clip(standardized, -clip_std, clip_std)
            matrix[:, index] = standardized

        for column, indicator in self.state["missing_indicators"].items():
            position = self._feature_names.index(indicator)
            matrix[:, position] = frame[column].isna().to_numpy(dtype="float64")

        if not np.isfinite(matrix).all():  # pragma: no cover - defensive
            bad = int((~np.isfinite(matrix)).sum())
            raise RuntimeError(f"preprocessor produced {bad} non-finite values")
        return matrix

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    # -- accessors ---------------------------------------------------------
    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    @property
    def feature_names(self) -> list[str]:
        if not self._fitted:
            raise RuntimeError("Preprocessor.feature_names accessed before fit")
        return list(self._feature_names)

    def state_dict(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Preprocessor.state_dict accessed before fit")
        return json.loads(json.dumps(self.state, sort_keys=True))

    def used_missing_indicators(self) -> dict[str, str]:
        return dict(self.state.get("missing_indicators", {}))

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        if not self._fitted:
            raise RuntimeError("cannot save an unfitted Preprocessor")
        target = write_json(Path(path), self.state_dict())
        return target

    @classmethod
    def load(cls, spec: DatasetSpec, path: str | Path) -> "Preprocessor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        instance = cls(spec, payload.get("config"))
        if payload.get("schema_version") != STATE_VERSION:
            raise ValueError(
                f"unsupported preprocessor state version {payload.get('schema_version')!r}"
            )
        instance.state = payload
        instance._feature_columns = list(payload["feature_columns"])
        instance._feature_names = list(payload["feature_names"])
        instance._fitted = True
        return instance


__all__ = ["DEFAULT_CONFIG", "Preprocessor", "STATE_VERSION"]
