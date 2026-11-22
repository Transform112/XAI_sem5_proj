"""Deterministic synthetic fixtures with the real schemas (tests + demo only).

Why this module exists
----------------------
The S1–S7 pipeline is still being built by the engineer, so the analysis toolkit
cannot yet consume real run artifacts. To keep the toolkit verifiable *now* and
to prove the artifact contract end-to-end, this module synthesises inputs that
obey the same schemas:

* S5 evaluation records (``RMetricTensor``) for FT / replay / EWC x model x seed;
* S8 explanation caches → real-update drift rows, null-update noise rows,
  explainer-repeat noise rows (computed with the *real* drift functions, not
  hand-typed numbers);
* S10 neighbourhood-change rows, S6 retention rows, S12 EAR comparison and
  ablation rows.

The data is **obviously synthetic** (deterministic generators, no raw files) and
is written only under ``src/analysis/examples/`` by the demo driver — never into the
real ``results/`` tree. Tests may write it to ``tmp_path``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis import artifacts
from src.analysis import metrics_cl as mc
from src.analysis import metrics_drift as md
from src.analysis import noise_floor as nf

#: Stream length frozen by docs/05 §3.
N_PERIODS: int = 5

#: Class list from the frozen NF-UNSW-NB15-v3 label space (docs/05 §3, D-033).
CLASSES: tuple[str, ...] = (
    "Benign",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Backdoor",
    "Analysis",
    "Worms",
)

#: Supported class set from D-033 (>= 50 test edges in every period): Benign + 6 attacks.
SUPPORTED_CLASSES: tuple[str, ...] = CLASSES[:7]

#: Anchors per class (rare classes deliberately below the A7 pooling threshold).
ANCHOR_COUNTS: Mapping[str, int] = {
    "Benign": 10,
    "DoS": 10,
    "Exploits": 10,
    "Fuzzers": 10,
    "Generic": 10,
    "Reconnaissance": 10,
    "Shellcode": 10,
    "Backdoor": 3,
    "Analysis": 3,
    "Worms": 3,
}

STRATEGIES: tuple[str, ...] = ("ft", "replay", "ewc")
MODELS: tuple[str, ...] = ("egraphsage", "gcn")
EXPLAINERS: tuple[str, ...] = artifacts.EXPLAINERS
SEED: int = 42

#: Per-class propensity used to couple drift magnitude and later forgetting, so
#: the synthetic RQ3 analysis has a real (but synthetic) signal to detect.
_CLASS_PROPENSITY: Mapping[str, float] = {
    class_name: round(0.15 + 0.85 * ((index * 7 + 3) % 10) / 9.0, 4)
    for index, class_name in enumerate(CLASSES)
}

#: Strategy-specific drift strengths (FT highest, replay lowest) — protocol story.
_DRIFT_STRENGTH: Mapping[str, float] = {"ft": 0.55, "replay": 0.22, "ewc": 0.38}

#: Strategy-specific forgetting rates used by the synthetic CL matrix.
_FORGET_RATE: Mapping[str, float] = {"ft": 0.10, "replay": 0.03, "ewc": 0.055}

FEATURE_DIM: int = 8
EDGE_DIM: int = 6


@dataclass(frozen=True)
class AnchorSpec:
    """One synthetic anchor edge: id, class, source period."""

    anchor_id: str
    class_name: str
    period: int


_PROPENSITY_KEYS: tuple[str, ...] = tuple(CLASSES)
_PROPENSITY_VALUES: np.ndarray = np.array([_CLASS_PROPENSITY[name] for name in CLASSES], dtype=float)


def anchor_specs() -> list[AnchorSpec]:
    """Deterministic anchor list: 79 anchors spread over 5 periods and 10 classes."""
    specs: list[AnchorSpec] = []
    for class_name in CLASSES:
        for index in range(ANCHOR_COUNTS[class_name]):
            period = 1 + (index % N_PERIODS)
            specs.append(AnchorSpec(f"anchor_P{period}_{class_name}_{index:02d}", class_name, period))
    return specs


def class_propensity(class_name: str) -> float:
    """Synthetic class-level coupling factor (drift ↑ / forgetting ↑)."""
    return float(_CLASS_PROPENSITY[class_name])


# ---------------------------------------------------------------------------
# Explanation vectors / masks
# ---------------------------------------------------------------------------
def _anchor_vector(seed: int, model: str, explainer: str, anchor_id: str, dim: int, kind: str) -> np.ndarray:
    rng = np.random.default_rng(
        [seed, _text_code(model), _text_code(explainer), _text_code(anchor_id), _text_code(kind)]
    )
    vector = rng.normal(0.0, 1.0, size=dim)
    if kind == "edge":
        vector = np.abs(vector) + 0.1  # edge mask importance is non-negative
    return vector


def _text_code(text: str) -> int:
    """Deterministic string -> int (``hash()`` is salted per process; never use it for seeds)."""
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % 1_000_003


def _shift_vector(seed: int, model: str, explainer: str, anchor_id: str, strategy: str, update: int, dim: int, kind: str) -> np.ndarray:
    rng = np.random.default_rng(
        [seed + 7, update, _text_code(model + explainer + strategy + anchor_id + kind)]
    )
    return rng.normal(0.0, 1.0, size=dim)


def _blend(base: np.ndarray, shift: np.ndarray, strength: float) -> np.ndarray:
    strength = float(min(max(strength, 0.0), 0.95))
    blended = (1.0 - strength) * base + strength * shift
    return blended


def _drift_strength(strategy: str, class_name: str, distance: int) -> float:
    return float(min(0.9, _DRIFT_STRENGTH[strategy] * (0.45 + 0.75 * class_propensity(class_name)) * (1.0 + 0.05 * distance)))


# ---------------------------------------------------------------------------
# S8 real-update drift rows
# ---------------------------------------------------------------------------
def real_drift_frame(*, seed: int = SEED, strategies: Sequence[str] = STRATEGIES, models: Sequence[str] = MODELS) -> pd.DataFrame:
    """Real update drift ``M_j -> M_t`` for every strategy/model/explainer/anchor pair.

    Drift is computed by the real S8 functions (:mod:`analysis.metrics_drift`),
    so the table exercises the production code path; the underlying explanation
    vectors are synthetic but have the protocol's shape/keys.
    """
    specs = anchor_specs()
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        for model in models:
            for explainer in EXPLAINERS:
                primary = md.PRIMARY_EDGE_METRIC if explainer == "GNNExplainer" else md.PRIMARY_FEATURE_METRIC
                for spec in specs:
                    feature_base = _anchor_vector(seed, model, explainer, spec.anchor_id, FEATURE_DIM, "feature")
                    edge_base = _anchor_vector(seed, model, explainer, spec.anchor_id, EDGE_DIM, "edge")
                    for update in range(spec.period + 1, N_PERIODS + 1):
                        distance = update - spec.period
                        strength = _drift_strength(strategy, spec.class_name, distance)
                        feature_shifted = _blend(
                            feature_base,
                            _shift_vector(seed, model, explainer, spec.anchor_id, strategy, update, FEATURE_DIM, "feature"),
                            strength,
                        )
                        edge_shifted = _blend(
                            edge_base,
                            _shift_vector(seed, model, explainer, spec.anchor_id, strategy, update, EDGE_DIM, "edge"),
                            strength,
                        )
                        row: dict[str, Any] = {
                            "strategy": strategy,
                            "model": model,
                            "explainer": explainer,
                            "seed": seed,
                            "anchor_id": spec.anchor_id,
                            "class": spec.class_name,
                            "anchor_period": spec.period,
                            "checkpoint_j": spec.period,
                            "checkpoint_t": update,
                        }
                        # Every S8 metric is reported for every row (docs/05 §8:
                        # never only the primary); the primary differs by explainer.
                        for name, value in md.attribution_similarity(feature_base, feature_shifted).items():
                            row[f"sim_{name}"] = float(value)
                            row[f"drift_{name}"] = 1.0 - float(value)
                        for name, value in md.edge_mask_similarity(edge_base, edge_shifted).items():
                            row[f"sim_{name}"] = float(value)
                            row[f"drift_{name}"] = 1.0 - float(value)
                        row["sim_primary"] = row[f"sim_{primary}"]
                        row["drift_primary"] = row[f"drift_{primary}"]
                        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("synthetic real drift frame is empty")
    return frame


# ---------------------------------------------------------------------------
# S8 noise-floor frames (null update, explainer repeats, combined null)
# ---------------------------------------------------------------------------
def _class_map(specs: Sequence[AnchorSpec]) -> dict[str, str]:
    return {spec.anchor_id: spec.class_name for spec in specs}


def model_null_noise_frame(
    *,
    seed: int = SEED,
    models: Sequence[str] = MODELS,
    strategies: Sequence[str] = ("ft",),
    cfg: nf.NoiseFloorConfig | None = None,
) -> pd.DataFrame:
    """S8 component 2: drift between a checkpoint and R = 5 same-period reseeds."""
    config = cfg or nf.NoiseFloorConfig()
    specs = anchor_specs()
    class_by_anchor = _class_map(specs)
    frames: list[pd.DataFrame] = []
    for model in models:
        for explainer in EXPLAINERS:
            kind = "edge" if explainer == "GNNExplainer" else "feature"
            metric_set = "edge" if kind == "edge" else "feature"
            dim = EDGE_DIM if kind == "edge" else FEATURE_DIM
            reference: dict[str, np.ndarray] = {}
            reseeded: dict[str, dict[str, np.ndarray]] = {}
            for spec in specs:
                base = _anchor_vector(seed, model, explainer, spec.anchor_id, dim, kind)
                reference[spec.anchor_id] = base
                reseeded[spec.anchor_id] = {}
                for null_index in range(config.n_null_reseeds):
                    null_seed = 10_000 + null_index
                    shift = _shift_vector(seed + null_index + 1, model, explainer, spec.anchor_id, "null_update", null_index, dim, kind)
                    reseeded[spec.anchor_id][null_seed] = _blend(base, shift, 0.10)
            frame = nf.null_update_drift(reference, reseeded, class_by_anchor=class_by_anchor, metric_set=metric_set, cfg=config)
            frame["model"] = model
            frame["explainer"] = explainer
            frame["strategy"] = strategies[0]
            frame["seed"] = seed
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def explainer_repeat_noise_frame(
    *,
    seed: int = SEED,
    models: Sequence[str] = MODELS,
    explainers: Sequence[str] = ("GNNExplainer",),
    cfg: nf.NoiseFloorConfig | None = None,
) -> pd.DataFrame:
    """S8 component 1: R = 10 seeded explainer repetitions on the same checkpoint.

    IG repetitions are identical (deterministic explainer, docs/05 §6) so their
    drift must be exactly 0; GNNExplainer repetitions are jittered.
    """
    config = cfg or nf.NoiseFloorConfig()
    specs = anchor_specs()
    class_by_anchor = _class_map(specs)
    frames: list[pd.DataFrame] = []
    for model in models:
        for explainer in explainers:
            kind = "edge" if explainer == "GNNExplainer" else "feature"
            metric_set = "edge" if kind == "edge" else "feature"
            dim = EDGE_DIM if kind == "edge" else FEATURE_DIM
            explanations: dict[str, list[np.ndarray]] = {}
            for spec in specs:
                base = _anchor_vector(seed, model, explainer, spec.anchor_id, dim, kind)
                repeats = []
                for repeat in range(config.n_explainer_repeats):
                    if explainer == "IG":
                        repeats.append(base.copy())
                    else:
                        shift = _shift_vector(seed + repeat + 3, model, explainer, spec.anchor_id, "repeat", repeat, dim, kind)
                        repeats.append(_blend(base, shift, 0.015))
                explanations[spec.anchor_id] = repeats
            frame = nf.explainer_repeat_drift(explanations, class_by_anchor=class_by_anchor, metric_set=metric_set, cfg=config)
            frame["model"] = model
            frame["explainer"] = explainer
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def combined_null_noise_frame(
    *,
    seed: int = SEED,
    models: Sequence[str] = MODELS,
    cfg: nf.NoiseFloorConfig | None = None,
) -> pd.DataFrame:
    """S8 component 3: null-update pairs with repeated explainer seeds (IG only demo)."""
    config = cfg or nf.NoiseFloorConfig()
    specs = anchor_specs()
    class_by_anchor = _class_map(specs)
    frames: list[pd.DataFrame] = []
    for model in models:
        reference: dict[str, np.ndarray] = {}
        reseeded_repeats: dict[str, dict[str, list[np.ndarray]]] = {}
        for spec in specs:
            base = _anchor_vector(seed, model, "IG", spec.anchor_id, FEATURE_DIM, "feature")
            reference[spec.anchor_id] = base
            reseeded_repeats[spec.anchor_id] = {}
            for null_index in range(config.n_null_reseeds):
                null_seed = 10_000 + null_index
                repeats = []
                for repeat in range(config.n_explainer_repeats):
                    shift = _shift_vector(seed + null_index + repeat + 5, model, "IG", spec.anchor_id, "combined", repeat, FEATURE_DIM, "feature")
                    repeats.append(_blend(base, shift, 0.05))
                reseeded_repeats[spec.anchor_id][null_seed] = repeats
        frame = nf.combined_null_drift(reference, reseeded_repeats, class_by_anchor=class_by_anchor, metric_set="feature", cfg=config)
        frame["model"] = model
        frame["explainer"] = "IG"
        frame["strategy"] = "ft"
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# S5 CL records / tensors
# ---------------------------------------------------------------------------
def cl_records(
    *,
    seed: int = SEED,
    strategies: Sequence[str] = STRATEGIES,
    models: Sequence[str] = MODELS,
    classes: Sequence[str] = CLASSES,
) -> list[mc.CLEvalRecord]:
    """Synthetic ``R[i, j]`` + per-class F1 records for one seed per run.

    Classes with a high synthetic propensity drift more *and* forget faster, so
    the RQ3 analysis has a detectable (synthetic, not real) association; the
    forgetting numbers themselves follow the protocol's CL shape (FT forgets most,
    replay least).
    """
    records: list[mc.CLEvalRecord] = []
    for run_index, strategy in enumerate(strategies):
        for model_index, model in enumerate(models):
            rng = np.random.default_rng([seed, run_index, model_index])
            base_f1 = {class_name: float(np.clip(rng.normal(0.78, 0.10), 0.45, 0.95)) for class_name in classes}
            for i in range(1, N_PERIODS + 1):
                per_class: dict[str, float] = {}
                for j in range(1, i + 1):
                    for class_name in classes:
                        propensity = class_propensity(class_name)
                        distance = i - j
                        rate = _FORGET_RATE[strategy] * (0.4 + 1.2 * propensity)
                        if distance == 0:
                            value = base_f1[class_name] + 0.02 + rng.normal(0.0, 0.008)
                        else:
                            value = base_f1[class_name] - rate * distance + rng.normal(0.0, 0.01)
                        per_class[f"P{j}::{class_name}"] = float(np.clip(value, 0.02, 0.99))
                # split per-period per-class dicts into separate records (one per eval period)
                for j in range(1, i + 1):
                    values = {class_name: per_class[f"P{j}::{class_name}"] for class_name in classes}
                    macro = float(np.mean(list(values.values())))
                    records.append(
                        mc.CLEvalRecord(
                            strategy=strategy,
                            model=model,
                            seed=seed,
                            checkpoint=i,
                            eval_period=j,
                            macro_f1=macro,
                            per_class_f1=values,
                            lane="synthetic",
                            source="analysis.synthetic.cl_records",
                        )
                    )
    return records


def cl_tensors(
    *,
    seed: int = SEED,
    strategies: Sequence[str] = STRATEGIES,
    models: Sequence[str] = MODELS,
) -> dict[tuple[str, str, int], mc.RMetricTensor]:
    """One :class:`analysis.metrics_cl.RMetricTensor` per synthetic run."""
    records = cl_records(seed=seed, strategies=strategies, models=models)
    return mc.RMetricTensor.from_records_grouped(records, classes=CLASSES, n_periods=N_PERIODS)


# ---------------------------------------------------------------------------
# Retention, neighbourhood change, EAR comparison/ablations
# ---------------------------------------------------------------------------
def retention_frame(*, seed: int = SEED, strategies: Sequence[str] = STRATEGIES, models: Sequence[str] = MODELS) -> pd.DataFrame:
    """S6 retained-anchor fraction per (strategy, model, class, checkpoint)."""
    rng = np.random.default_rng([seed, 51])
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        for model in models:
            for checkpoint in range(1, N_PERIODS + 1):
                for class_name in CLASSES:
                    base = {"ft": 0.92, "replay": 0.97, "ewc": 0.95}[strategy]
                    value = base - 0.035 * (checkpoint - 1) * (1.0 + class_propensity(class_name)) + rng.normal(0.0, 0.004)
                    rows.append(
                        {
                            "strategy": strategy,
                            "model": model,
                            "class": class_name,
                            "checkpoint_t": checkpoint,
                            "retained_fraction": float(np.clip(value, 0.0, 1.0)),
                        }
                    )
    return pd.DataFrame(rows)


def neighbourhood_frame(real_drift: pd.DataFrame | None = None, *, seed: int = SEED) -> pd.DataFrame:
    """S10 per-anchor neighbourhood change ``1 - Jaccard(H_j, H_t)`` (synthetic).

    Weakly coupled to drift with additive noise, as the protocol hypothesises
    for real updates; the noise-floor / null-update comparison lives in the
    tables layer.
    """
    rng = np.random.default_rng([seed, 91])
    frame = real_drift if real_drift is not None else real_drift_frame(seed=seed)
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        propensity = class_propensity(str(row["class"]))
        change = float(np.clip(0.25 * row["drift_primary"] * (0.5 + propensity) + rng.normal(0.0, 0.04) + 0.02, 0.0, 0.95))
        rows.append(
            {
                "strategy": row["strategy"],
                "model": row["model"],
                "explainer": row["explainer"],
                "seed": row["seed"],
                "anchor_id": row["anchor_id"],
                "class": row["class"],
                "checkpoint_j": row["checkpoint_j"],
                "checkpoint_t": row["checkpoint_t"],
                "neighbourhood_change": change,
                "degree": float(rng.integers(1, 20)),
            }
        )
    return pd.DataFrame(rows)


def ear_comparison_frame(*, seed: int = SEED, models: Sequence[str] = MODELS) -> pd.DataFrame:
    """S12 RQ4 comparison table: replay vs EAR-A vs EAR-B under identical budget."""
    variants = {
        "replay": (0.300, 0.620, 0.070, 0.081),
        "ear_a": (0.222, 0.615, 0.052, 0.099),
        "ear_b": (0.245, 0.613, 0.058, 0.096),
    }
    rng = np.random.default_rng([seed, 121])
    rows: list[dict[str, Any]] = []
    for model in models:
        for variant, (drift, f1, forgetting, bwt) in variants.items():
            for run_seed in (42, 43, 44):
                rows.append(
                    {
                        "model": model,
                        "variant": variant,
                        "seed": run_seed,
                        "drift_mean": float(np.clip(rng.normal(drift, 0.012), 0.05, 0.9)),
                        "macro_f1": float(np.clip(rng.normal(f1, 0.008), 0.3, 0.95)),
                        "forgetting": float(np.clip(rng.normal(forgetting, 0.008), 0.0, 0.5)),
                        "bwt": float(np.clip(rng.normal(bwt, 0.012), -0.5, 0.5)),
                        "runtime_s": float(abs(rng.normal({"replay": 210.0, "ear_a": 268.0, "ear_b": 255.0}[variant], 12.0))),
                    }
                )
    return pd.DataFrame(rows)


def ear_ablation_frame(*, seed: int = SEED, models: Sequence[str] = MODELS) -> pd.DataFrame:
    """S12 RQ4 ablations: buffer size x beta grid x attribution source."""
    rng = np.random.default_rng([seed, 131])
    rows: list[dict[str, Any]] = []
    for model in models:
        for variant in ("ear_a", "ear_b"):
            for buffer_size in (500, 2000, 5000):
                for beta in (0.1, 0.5, 1.0):
                    for source in ("IG", "GNNExplainer"):
                        base_drift = 0.30 - 0.02 * np.log10(buffer_size / 500.0) - 0.025 * beta + (0.012 if source == "GNNExplainer" else 0.0)
                        rows.append(
                            {
                                "model": model,
                                "variant": variant,
                                "buffer_size": buffer_size,
                                "beta": beta,
                                "attribution_source": source,
                                "drift_mean": float(max(0.01, rng.normal(base_drift, 0.008))),
                                "macro_f1": float(np.clip(rng.normal(0.615 - 0.004 * (buffer_size == 500), 0.006), 0.3, 0.95)),
                                "runtime_s": float(abs(rng.normal(200.0 + 0.01 * buffer_size, 10.0))),
                            }
                        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SyntheticBundle:
    """Everything the S13 table/figure generators need, in one object.

    ``real_drift`` and the noise frames are mutually consistent: the noise floor
    is built from the same anchor/checkpoint vectors as the real updates so
    ``excess``/``above_floor`` are meaningful (the synthetic drift is engineered
    so fine-tuning sits above the floor and replay partly below it).
    """

    seed: int
    cl_records: list[mc.CLEvalRecord]
    cl_tensors: dict[tuple[str, str, int], mc.RMetricTensor]
    real_drift: pd.DataFrame
    model_null_noise: pd.DataFrame
    explainer_repeat_noise: pd.DataFrame
    combined_null_noise: pd.DataFrame
    retention: pd.DataFrame
    neighbourhood: pd.DataFrame
    ear_comparison: pd.DataFrame
    ear_ablations: pd.DataFrame

    @property
    def classes(self) -> tuple[str, ...]:
        return CLASSES

    @property
    def primary_tensor(self) -> mc.RMetricTensor:
        """FT / primary model tensor (the RQ3 reference run)."""
        return self.cl_tensors[("ft", "egraphsage", self.seed)]


def make_bundle(
    *,
    seed: int = SEED,
    include_combined: bool = False,
    models: Sequence[str] = MODELS,
    strategies: Sequence[str] = STRATEGIES,
) -> SyntheticBundle:
    """Build the deterministic synthetic bundle (seconds; options keep tests fast)."""
    model_list = tuple(models)
    strategy_list = tuple(strategies)
    real = real_drift_frame(seed=seed, models=model_list, strategies=strategy_list)
    null_noise = model_null_noise_frame(seed=seed, models=model_list, strategies=strategy_list[:1])
    repeat_noise = explainer_repeat_noise_frame(seed=seed, models=model_list)
    combined = combined_null_noise_frame(seed=seed, models=model_list) if include_combined else pd.DataFrame()
    return SyntheticBundle(
        seed=seed,
        cl_records=cl_records(seed=seed, models=model_list, strategies=strategy_list),
        cl_tensors=cl_tensors(seed=seed, models=model_list, strategies=strategy_list),
        real_drift=real,
        model_null_noise=null_noise,
        explainer_repeat_noise=repeat_noise,
        combined_null_noise=combined,
        retention=retention_frame(seed=seed, models=model_list, strategies=strategy_list),
        neighbourhood=neighbourhood_frame(real, seed=seed),
        ear_comparison=ear_comparison_frame(seed=seed, models=model_list),
        ear_ablations=ear_ablation_frame(seed=seed, models=model_list),
    )


__all__ = [
    "ANCHOR_COUNTS",
    "AnchorSpec",
    "CLASSES",
    "EDGE_DIM",
    "EXPLAINERS",
    "FEATURE_DIM",
    "MODELS",
    "N_PERIODS",
    "SEED",
    "STRATEGIES",
    "SUPPORTED_CLASSES",
    "SyntheticBundle",
    "anchor_specs",
    "class_propensity",
    "cl_records",
    "cl_tensors",
    "combined_null_noise_frame",
    "ear_ablation_frame",
    "ear_comparison_frame",
    "explainer_repeat_noise_frame",
    "make_bundle",
    "model_null_noise_frame",
    "neighbourhood_frame",
    "real_drift_frame",
    "retention_frame",
]
