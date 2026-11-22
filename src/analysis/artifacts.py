"""Exact artifact path registry (binding names from the claim→evidence skeleton).

Authority
---------
``research/lit/claims_evidence_skeleton.md`` §C1–C7 and "Outputs to generate at
S13/G5" fix the artifact names each claim needs; ``docs/source_documents/Doc 2b`` §S8–S13
and ``docs/05`` §8–§13 fix their content. This module is the single place where
those names are spelled out, so no generator hard-codes a path and the verifier
can diff the toolkit against the skeleton mechanically.

Conventions
-----------
* Paths are **workspace-relative POSIX strings** exactly as they appear in the
  skeleton (``results/text/tables/rq1_h1_tests.csv``), because those strings are the
  contract. :class:`ArtifactLayout` turns them into concrete paths for a given
  results root (``results/`` by default, a temporary directory in tests, and
  ``src/analysis/examples/results/`` for the synthetic demo).
* ``(proposed)`` names in the skeleton are kept as-is; they are marked
  ``PROPOSED`` so the Lead can confirm them before S7 actually writes them.
  Until confirmed they may be used by tests and the synthetic demo only.
* ``paper_writing/manuscript/numbers.json`` lives outside the analyst write scope
  (``src/analysis/**``, ``results/text/tables/**``, ``results/images/**``); the exporter
  can serialise a validated payload anywhere, and the S13 writer (Lead/verifier)
  points it at ``paper_writing/manuscript/numbers.json``. The synthetic demo writes a schema-
  identical file under ``src/analysis/examples/`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

#: Workspace-relative table paths copied verbatim from
#: ``research/lit/claims_evidence_skeleton.md`` (C1–C7 table + S13 outputs).
TABLE_ARTIFACTS: Mapping[str, str] = {
    # -- S5 CL evaluation ----------------------------------------------------
    "cl_metrics": "results/text/tables/cl_metrics.csv",
    "per_class_f1_npz": "results/text/tables/per_class_f1.npz",  # proposed in skeleton gap #3
    "per_class_f1_csv": "results/text/tables/per_class_f1.csv",
    # -- S6/S7 prerequisites -------------------------------------------------
    "anchor_retention": "results/text/tables/anchor_retention.csv",  # (proposed)
    "explain_index": "results/text/tables/explain_index.csv",
    # -- S8 noise floor + drift ---------------------------------------------
    "noise_floor": "results/text/tables/noise_floor.csv",
    # drift_metrics_{explainer}.csv handled by drift_metrics_table()
    # -- S9 RQ1 --------------------------------------------------------------
    "rq1_anchor_level": "results/text/tables/rq1_anchor_level.csv",  # (proposed)
    "rq1_aggregated": "results/text/tables/rq1_aggregated.csv",  # (proposed)
    "rq1_h1_tests": "results/text/tables/rq1_h1_tests.csv",  # (proposed)
    "rq1_strategy_comparison": "results/text/tables/rq1_strategy_comparison.csv",  # (proposed)
    # -- S10 RQ2 -------------------------------------------------------------
    "rq2_anchor_level": "results/text/tables/rq2_anchor_level.csv",  # (proposed)
    "rq2_correlations": "results/text/tables/rq2_correlations.csv",  # (proposed)
    # -- S11 RQ3 -------------------------------------------------------------
    "rq3_features": "results/text/tables/rq3_features.csv",  # (proposed)
    "rq3_auc": "results/text/tables/rq3_auc.csv",  # (proposed)
    "rq3_forgetting_labels": "results/text/tables/rq3_forgetting_labels.csv",  # (proposed)
    # -- S12 RQ4 -------------------------------------------------------------
    "rq4_comparison": "results/text/tables/rq4_comparison.csv",  # (proposed)
    "rq4_ablations": "results/text/tables/rq4_ablations.csv",  # (proposed)
    # -- S13 paper numbers ----------------------------------------------------
    "paper_numbers": "paper_writing/manuscript/numbers.json",
}

#: Artifact names the skeleton marks ``(proposed)``; confirm at S5/S7 freeze.
PROPOSED: frozenset[str] = frozenset(
    {
        "per_class_f1_npz",
        "per_class_f1_csv",
        "anchor_retention",
        "rq1_anchor_level",
        "rq1_aggregated",
        "rq1_h1_tests",
        "rq1_strategy_comparison",
        "rq2_anchor_level",
        "rq2_correlations",
        "rq3_features",
        "rq3_auc",
        "rq3_forgetting_labels",
        "rq4_comparison",
        "rq4_ablations",
    }
)

#: Supplementary tables this toolkit emits in addition to the skeleton contract.
#: They are *not* claimed by C1–C7; they carry the detail the skeleton asks the
#: RQ rows to report (e.g. RQ3 coefficients "with CIs"). Registered here so the
#: Lead can confirm them before S7/S11 write real artifacts.
EXTRA_TABLE_ARTIFACTS: Mapping[str, str] = {
    "rq3_coefficients": "results/text/tables/rq3_coefficients.csv",
}

#: Concrete figure names. The skeleton fixes the *prefixes* (``s8_*``, ``s9_*``,
#: ``s10_*``, ``s11_*``) and Doc 2b §S13 fixes the figure content; the concrete
#: filenames below are this toolkit's contract for that wildcard.
FIGURE_ARTIFACTS: Mapping[str, str] = {
    # Doc 2b §S13: "drift distributions (real vs null) per strategy"
    "s8_drift_real_vs_null": "results/images/s8_drift_real_vs_null.png",
    # Doc 2b §S8/S9: per-class q95 noise floor
    "s8_noise_floor_q95": "results/images/s8_noise_floor_q95.png",
    # Doc 2b §S13: "excess drift by period pair"
    "s9_excess_by_period_pair": "results/images/s9_excess_by_period_pair.png",
    # Doc 2b §S9: "fraction of anchors above the null floor vs 5%"
    "s9_above_floor_fraction": "results/images/s9_above_floor_fraction.png",
    # Doc 2b §S13: "scatter of neighbourhood change vs drift"
    "s10_neighbourhood_vs_drift": "results/images/s10_neighbourhood_change_vs_drift.png",
    # Doc 2b §S13: "per-class F1 trajectories with drift markers"
    "s11_f1_trajectories_drift": "results/images/s11_f1_trajectories_drift_markers.png",
    # Doc 2b §S13: "EAR trade-off plot (drift vs macro-F1)"
    "s12_ear_drift_vs_f1": "results/images/s12_ear_drift_vs_macro_f1.png",
}

#: Canonical explainer names from docs/05 §6 and Doc 2b §S7, mapped to the
#: filesystem slug used by ``drift_metrics_{explainer}.csv``.
EXPLAINER_SLUGS: Mapping[str, str] = {
    "IG": "ig",
    "GNNExplainer": "gnnexplainer",
}

#: Explanation channels from Doc 2b §S8 ("noise-floor components").
NOISE_CHANNELS: tuple[str, ...] = ("model_null", "explainer_repeat", "combined")

#: Canonical explainer order (docs/05 §6: IG then GNNExplainer).
EXPLAINERS: tuple[str, ...] = tuple(EXPLAINER_SLUGS)


def canonical_explainer(name: str) -> str:
    """Normalise an explainer name to ``"IG"`` / ``"GNNExplainer"``.

    Docs/05 §6 and Doc 2b §S7 use exactly these two names; the filesystem slug
    is derived from them (``drift_metrics_ig.csv``, ``drift_metrics_gnnexplainer.csv``).
    """
    key = " ".join(str(name).strip().split()).casefold()
    aliases = {
        "ig": "IG",
        "integrated gradients": "IG",
        "integratedgradients": "IG",
        "integrated_gradients": "IG",
        "gnnexplainer": "GNNExplainer",
        "gnn explainer": "GNNExplainer",
        "gnn_explainer": "GNNExplainer",
        "gnn-explainer": "GNNExplainer",
    }
    if key not in aliases:
        raise ValueError(
            f"unknown explainer {name!r}; expected one of {sorted(EXPLAINERS)} "
            f"(aliases: {sorted(set(aliases.values()))})"
        )
    return aliases[key]


def explainer_slug(name: str) -> str:
    """Filesystem slug for an explainer: ``IG -> ig``, ``GNNExplainer -> gnnexplainer``."""
    return EXPLAINER_SLUGS[canonical_explainer(name)]


def drift_metrics_table_path(explainer: str, root: str | Path | None = None) -> Path:
    """``results/text/tables/drift_metrics_{explainer}.csv`` per the claim skeleton C1."""
    return _root(root) / "text" / "tables" / f"drift_metrics_{explainer_slug(explainer)}.csv"


def _root(root: str | Path | None) -> Path:
    """Return the results root; default = the project's configured results root."""
    if root is not None:
        return Path(root)
    from src.utils.paths import load_paths  # local import: analysis must import without a repo

    return load_paths(lane="cpu" if _cpu_lane_available() else None).results_root


def _cpu_lane_available() -> bool:
    from src.utils.paths import find_project_root

    return (find_project_root() / "configs" / "paths" / "cpu.yaml").is_file()


@dataclass(frozen=True)
class ArtifactLayout:
    """Resolved table/figure roots for one analysis run.

    ``results_root`` is the directory that contains ``text/tables/`` and
    ``images/``. The real project layout resolves to ``<repo>/results``; the
    synthetic demo passes ``src/analysis/examples/results`` so no synthetic artefact
    can land in the real results tree.
    """

    results_root: Path

    @classmethod
    def default(cls) -> "ArtifactLayout":
        """Project default: ``load_paths`` from ``src/utils/paths.py``."""
        from src.utils.paths import load_paths

        paths = load_paths(ensure=False)
        return cls(results_root=paths.results_root)

    @property
    def tables_root(self) -> Path:
        return self.results_root / "text" / "tables"

    @property
    def figures_root(self) -> Path:
        return self.results_root / "images"

    # -- resolution ---------------------------------------------------------
    def table(self, name: str) -> Path:
        """Resolve a registered table name to its concrete path."""
        if name not in TABLE_ARTIFACTS:
            raise KeyError(f"unknown table artifact {name!r}; known: {sorted(TABLE_ARTIFACTS)}")
        return self._workspace_path(TABLE_ARTIFACTS[name])

    def figure(self, name: str) -> Path:
        """Resolve a registered figure name to its concrete path."""
        if name not in FIGURE_ARTIFACTS:
            raise KeyError(f"unknown figure artifact {name!r}; known: {sorted(FIGURE_ARTIFACTS)}")
        return self._workspace_path(FIGURE_ARTIFACTS[name])

    def drift_metrics(self, explainer: str) -> Path:
        return self.tables_root / f"drift_metrics_{explainer_slug(explainer)}.csv"

    def extra_table(self, name: str) -> Path:
        """Resolve a supplementary (registered, not skeleton-claimed) table."""
        if name not in EXTRA_TABLE_ARTIFACTS:
            raise KeyError(f"unknown extra table {name!r}; known: {sorted(EXTRA_TABLE_ARTIFACTS)}")
        return self._workspace_path(EXTRA_TABLE_ARTIFACTS[name])

    def paper_numbers(self, override: str | Path | None = None) -> Path:
        """``paper_writing/manuscript/numbers.json`` path convention.

        The real file is written by the S13 writer (Lead/verifier); this method
        only resolves the contract path. When ``override`` is given (tests /
        synthetic demo) that path is used instead.
        """
        if override is not None:
            return Path(override)
        # workspace-relative contract path, resolved next to the results root
        return self.results_root.parent / "paper_writing" / "manuscript" / "numbers.json"

    # -- internals ----------------------------------------------------------
    def _workspace_path(self, relative: str) -> Path:
        rel = Path(relative)
        # ``results/text/tables/x.csv`` -> <results_root>/tables/x.csv
        parts = rel.parts
        if len(parts) >= 2 and parts[0] == "results":
            return self.results_root.joinpath(*parts[1:])
        if len(parts) >= 1 and parts[0] == "paper_writing":
            return self.results_root.parent.joinpath(*parts)
        return self.results_root / rel

    def ensure(self) -> "ArtifactLayout":
        """Create the table/figure roots (and the numbers parent on demand)."""
        self.tables_root.mkdir(parents=True, exist_ok=True)
        self.figures_root.mkdir(parents=True, exist_ok=True)
        return self


__all__ = [
    "ArtifactLayout",
    "EXPLAINERS",
    "EXPLAINER_SLUGS",
    "EXTRA_TABLE_ARTIFACTS",
    "FIGURE_ARTIFACTS",
    "NOISE_CHANNELS",
    "PROPOSED",
    "TABLE_ARTIFACTS",
    "canonical_explainer",
    "drift_metrics_table_path",
    "explainer_slug",
]
