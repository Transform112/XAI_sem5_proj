"""Documented label taxonomies (fine -> coarse class aggregation).

Why this module exists
----------------------
Some NF-v3 datasets ship a finer label vocabulary than the paper reports. The
clearest case is NF-CSE-CIC-IDS2018-v3 (D-031): the raw CSV `Attack` column has
15 fine CSE-CIC-IDS2018 labels while the UQ paper reports 7 aggregated classes
(Benign, Bot, BruteForce, DDoS, DoS, Infiltration, Web Attacks). Aggregating
labels is a **documented, validated** transformation, not a silent relabel:

* the mapping and the published aggregated counts live in the dataset YAML,
* every raw label must appear in the mapping (an unmapped label raises),
* the aggregated counts must equal the published counts when
  ``strict_aggregate_check`` is enabled (default), otherwise the stage fails
  loudly with the exact per-class delta.

The same mechanism generalises to future datasets/papers without code changes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd


class TaxonomyError(ValueError):
    """Raised when labels cannot be mapped/validated against the published table."""


@dataclass(frozen=True)
class LabelTaxonomy:
    """Fine -> coarse label aggregation with published-count validation."""

    name: str | None
    aggregate_to: Mapping[str, str]
    published_aggregated_counts: Mapping[str, int] = field(default_factory=dict)
    strict_aggregate_check: bool = True
    source: str | None = None
    fine_class_count: int | None = None

    # -- construction ------------------------------------------------------
    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "LabelTaxonomy | None":
        if not config:
            return None
        if not isinstance(config, Mapping):
            raise TaxonomyError("label_taxonomy must be a mapping")
        aggregate = config.get("aggregate_to")
        if not isinstance(aggregate, Mapping) or not aggregate:
            raise TaxonomyError("label_taxonomy.aggregate_to must be a non-empty mapping")
        mapping: dict[str, str] = {}
        for fine, coarse in aggregate.items():
            if not isinstance(coarse, str) or not coarse:
                raise TaxonomyError(f"label_taxonomy.aggregate_to[{fine!r}] must be a class name")
            mapping[str(fine)] = coarse
        published = {
            str(key): int(value)
            for key, value in (config.get("published_aggregated_counts") or {}).items()
        }
        if published and sum(published.values()) <= 0:
            raise TaxonomyError("label_taxonomy.published_aggregated_counts sums to zero")
        return cls(
            name=config.get("name"),
            aggregate_to=mapping,
            published_aggregated_counts=published,
            strict_aggregate_check=bool(config.get("strict_aggregate_check", True)),
            source=config.get("source"),
            fine_class_count=(
                None if config.get("fine_class_count") is None else int(config["fine_class_count"])
            ),
        )

    @classmethod
    def from_spec(cls, spec: Any) -> "LabelTaxonomy | None":
        return cls.from_config(spec.get("label_taxonomy"))

    # -- helpers -----------------------------------------------------------
    def target_classes(self) -> list[str]:
        if self.published_aggregated_counts:
            return sorted(self.published_aggregated_counts)
        return sorted(set(self.aggregate_to.values()))

    def apply_series(self, values: pd.Series) -> pd.Series:
        """Map a fine-label series to coarse labels, raising on unknown labels."""
        mapped = values.map(self.aggregate_to)
        unknown = values.notna() & mapped.isna()
        if unknown.any():
            unknown_values = sorted({str(value) for value in values[unknown].unique()})
            raise TaxonomyError(
                f"label taxonomy {self.name or ''} does not cover raw label(s): {unknown_values}; "
                "add them to label_taxonomy.aggregate_to (never relabel silently)"
            )
        return mapped

    def aggregate_counts(self, fine_counts: Mapping[str, Any]) -> dict[str, int]:
        """Aggregate a fine-label count mapping and validate it if strict."""
        aggregated: Counter[str] = Counter()
        unmapped: list[str] = []
        for label, count in fine_counts.items():
            coarse = self.aggregate_to.get(str(label))
            if coarse is None:
                unmapped.append(str(label))
                continue
            aggregated[coarse] += int(count)
        if unmapped:
            raise TaxonomyError(
                f"label taxonomy {self.name or ''} does not cover label(s): {sorted(unmapped)}"
            )
        result = dict(aggregated)
        if self.strict_aggregate_check and self.published_aggregated_counts:
            self.validate_counts(result, measured_total=sum(int(v) for v in fine_counts.values()))
        return result

    def validate_counts(self, aggregated: Mapping[str, int], *, measured_total: int | None = None) -> None:
        """Fail loudly when aggregated counts disagree with the published table."""
        if not self.published_aggregated_counts:
            return
        problems: list[str] = []
        expected_total = sum(self.published_aggregated_counts.values())
        measured_sum = sum(int(value) for value in aggregated.values())
        if measured_total is not None and int(measured_total) != measured_sum:
            problems.append(
                f"raw labelled rows {measured_total} != aggregated sum {measured_sum} (unmapped labels?)"
            )
        if measured_sum != expected_total:
            problems.append(
                f"aggregated total {measured_sum} != published total {expected_total} "
                f"(delta {measured_sum - expected_total:+d})"
            )
        for label in sorted(set(self.published_aggregated_counts) | set(aggregated)):
            expected = int(self.published_aggregated_counts.get(label, 0))
            measured = int(aggregated.get(label, 0))
            if expected != measured:
                problems.append(f"{label}: measured {measured} vs published {expected} (delta {measured - expected:+d})")
        if problems:
            raise TaxonomyError(
                f"label taxonomy {self.name or ''} aggregated counts do not match the published table: "
                + "; ".join(problems)
            )


__all__ = ["LabelTaxonomy", "TaxonomyError"]
