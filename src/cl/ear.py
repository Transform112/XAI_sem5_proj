"""S12 explanation-anchored replay hooks: EAR-A and EAR-B (Doc 2b S12).

Definitions (authoritative text: Doc 2b S12; configs/ear.yaml):

* **EAR-A**: ``L_attr = mean(1 - cos(a_t, a_stored))`` where ``a_stored`` are the
  attributions stored at buffer insertion and ``a_t`` are *cheap differentiable*
  current attributions. Per Doc 2b the per-step surrogate is gradient x input
  (a documented 1-step IG approximation); stored attributions use IG by default.
* **EAR-B**: differentiable soft top-k / rank consistency on the stored top-k
  feature set, penalising changes of the top-k set.
* ``beta`` is selected on validation macro-F1 of old periods + validation drift
  (never test), from the pre-registered grid ``{0.1, 0.5, 1.0}`` (optionally 0.01).
* Ablations (buffer size, attribution source, beta, consistency on/off) are
  configured in ``configs/ear.yaml``; the full RQ4 matrix waits for G2.

The training hook is exposed through ``train_edge_classifier(penalty_fn=...)``:
:class:`EARStrategy` builds a penalty closure over a small number of buffer
anchors (``max_anchors_per_step``), so the cost stays bounded.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from src.cl.strategies import ReplayBuffer, ReplayStrategy
from src.train.trainer import TrainSettings, train_edge_classifier
from src.utils.logging import get_logger

LOGGER = get_logger("xdrift.ear")

BETA_GRID: tuple[float, ...] = (0.1, 0.5, 1.0)


# ---------------------------------------------------------------------------
# Pure losses (hand-testable)
# ---------------------------------------------------------------------------
def cosine_consistency_loss(current: torch.Tensor, stored: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """EAR-A: ``mean(1 - cosine_similarity(a_t, a_stored))``."""
    current = current.reshape(-1).float()
    stored = stored.reshape(-1).float()
    if current.numel() == 0:
        return torch.zeros((), dtype=torch.float32)
    similarity = torch.dot(current, stored) / (current.norm() * stored.norm() + eps)
    return 1.0 - similarity


def soft_rank_vector(attributions: torch.Tensor, *, temperature: float = 0.1) -> torch.Tensor:
    """Differentiable soft top-k/rank vector: softmax(|a| / temperature)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = torch.abs(attributions.reshape(-1).float())
    return torch.softmax(values / float(temperature), dim=0)


def ear_b_rank_loss(
    current: torch.Tensor,
    stored: torch.Tensor,
    *,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """EAR-B: ``1 - cos(soft_rank(a_t), soft_rank(a_stored))`` (soft top-k consistency)."""
    current_rank = soft_rank_vector(current, temperature=temperature)
    stored_rank = soft_rank_vector(stored, temperature=temperature)
    similarity = torch.dot(current_rank, stored_rank) / (current_rank.norm() * stored_rank.norm() + eps)
    return 1.0 - similarity


def grad_x_input_attributions(
    model: nn.Module,
    data: Any,
    target_edge: int,
    *,
    target_class: int | None = None,
) -> torch.Tensor:
    """Cheap differentiable attribution surrogate for EAR-A/EAR-B training.

    ``a = e_target * d logit/target_class / d e_target`` computed with
    ``create_graph=True`` so the consistency loss backpropagates into the model.
    """
    device = next(model.parameters()).device
    x = data.x.to(device) if hasattr(data.x, "to") else data.x
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.detach().clone().to(device).requires_grad_(True)
    target_edge = int(target_edge)
    if hasattr(model, "node_embeddings") and hasattr(model, "edge_logits"):
        # T32: head-only target-edge logits (no full-edge head / 1.3 GiB concat),
        # still differentiable through the shared node embeddings and edge features.
        target_tensor = torch.as_tensor([target_edge], device=device)
        hidden = model.node_embeddings(x, edge_index, edge_attr)
        logits = model.edge_logits(hidden, edge_index, edge_attr, target_tensor)
        row = 0
    else:
        logits = model(x, edge_index, edge_attr)
        row = target_edge
    if target_class is None:
        target_class = int(logits[row].argmax().detach())
    score = logits[row, int(target_class)]
    grads = torch.autograd.grad(score, edge_attr, create_graph=True, retain_graph=True)[0]
    return (edge_attr[target_edge] * grads[target_edge]).reshape(-1)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class EARStrategy(ReplayStrategy):
    """Replay + attribution-consistency penalty (EAR-A or EAR-B)."""

    def __init__(
        self,
        *,
        variant: str = "ear_a",
        beta: float = 0.5,
        buffer: ReplayBuffer | None = None,
        alpha: float = 1.0,
        replay_batch: int = 32,
        attribution_source: str = "ig",
        training_attribution: str = "grad_x_input",
        top_k: int = 5,
        temperature: float = 0.1,
        max_anchors_per_step: int = 2,
        max_attributions_per_period: int = 64,
        beta_grid: Sequence[float] = BETA_GRID,
    ) -> None:
        if variant not in ("ear_a", "ear_b"):
            raise ValueError("variant must be 'ear_a' or 'ear_b'")
        super().__init__(buffer=buffer or ReplayBuffer(), alpha=alpha, replay_batch=replay_batch)
        self.name = variant
        self.variant = variant
        self.beta = float(beta)
        self.attribution_source = str(attribution_source)
        self.training_attribution = str(training_attribution)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.max_anchors_per_step = int(max_anchors_per_step)
        self.max_attributions_per_period = int(max_attributions_per_period)
        if self.max_attributions_per_period < 1:
            raise ValueError("max_attributions_per_period must be positive")
        self.beta_grid = [float(value) for value in beta_grid]
        self.last_consistency: dict[str, float] = {}

    # -- hooks -------------------------------------------------------------
    def _entries_with_stored_attributions(self, seed: int) -> list[Any]:
        entries = [entry for entry in self.buffer.entries if getattr(entry, "attributions", None) is not None]
        if not entries:
            return []
        rng = np.random.default_rng(seed)
        take = min(self.max_anchors_per_step, len(entries))
        picks = rng.choice(len(entries), size=take, replace=False)
        return [entries[int(position)] for position in picks]

    def consistency_loss(self, model: nn.Module, entries: Sequence[Any]) -> torch.Tensor:
        """Attribution-consistency penalty over buffer anchors (differentiable)."""
        losses: list[torch.Tensor] = []
        for entry in entries:
            target_index = int(entry.target_index[0])
            current = grad_x_input_attributions(model, entry, target_index)
            stored = torch.as_tensor(entry.attributions, dtype=current.dtype, device=current.device)
            if self.variant == "ear_a":
                losses.append(cosine_consistency_loss(current, stored))
            else:
                losses.append(ear_b_rank_loss(current, stored, temperature=self.temperature))
        if not losses:
            return torch.zeros((), dtype=torch.float32, device=next(model.parameters()).device)
        return torch.stack(losses).mean()

    def _penalty_closure(self, model: nn.Module, entries: Sequence[Any]):
        if not entries:
            return None

        def penalty_fn(module: nn.Module) -> torch.Tensor:
            return self.beta * self.consistency_loss(module, entries)

        return penalty_fn

    # -- update ------------------------------------------------------------
    def update(self, model, data, *, settings: TrainSettings, seed: int, device: str, **kwargs) -> dict[str, Any]:
        import time

        start = time.monotonic()
        anchor_entries = self._entries_with_stored_attributions(seed)
        replay_entries = self.buffer.sample(self.replay_batch, seed=seed)
        penalty_fn = self._penalty_closure(model, anchor_entries)
        result = train_edge_classifier(
            model,
            data,
            settings=settings,
            seed=seed,
            device=device,
            replay_entries=replay_entries,
            replay_alpha=self.alpha,
            penalty_fn=penalty_fn,
            **kwargs,
        )
        # Buffer update AFTER training (Doc 2 S4.2), then store fresh attributions
        # for the S12 EAR buffer (stored offline with IG by default).
        buffer_stats = self.buffer.add_from_graph(data, seed=seed)
        stored = self._store_attributions(model, device)
        consistency_stats = self._measure_consistency(model, device)
        runtime = time.monotonic() - start
        self.last_consistency = consistency_stats
        return {
            "strategy": self.name,
            "variant": self.variant,
            "beta": self.beta,
            "attribution_source": self.attribution_source,
            "training_attribution": self.training_attribution,
            "buffer": buffer_stats,
            "buffer_class_balance": self.buffer.class_balance(),
            "replay_entries_used": len(replay_entries),
            "anchors_with_stored_attributions": len(anchor_entries),
            "stored_attributions": stored,
            "max_attributions_per_period": self.max_attributions_per_period,
            "consistency": consistency_stats,
            "ear_runtime_s": round(runtime, 3),
            **result,
        }

    def _store_attributions(self, model: nn.Module, device: str, *, refresh: bool = False) -> int:
        """Store detached IG attributions on buffer entries (EAR storage step).

        D-068/default EAR semantics: attributions target the model state **at
        insertion time**, so already-attributed entries are skipped (pass
        ``refresh=True`` only for an explicit later-model diagnostic). T34
        (bug-report #12): this avoids O(buffer Ã— steps) IG forwards every period.
        """
        from src.explain.ig import integrated_gradients

        stored = 0
        for entry in self.buffer.entries:
            if stored >= self.max_attributions_per_period:
                break
            if not refresh and getattr(entry, "attributions", None) is not None:
                continue
            try:
                record = integrated_gradients(
                    model, entry, int(entry.target_index[0]), device=device
                )
                entry.attributions = torch.as_tensor(record["attributions"], dtype=torch.float32)
                stored += 1
            except Exception:  # noqa: BLE001 - storage must not break training
                continue
        return stored

    def _measure_consistency(self, model: nn.Module, device: str) -> dict[str, float]:
        """Detached consistency stats (before/after) for reporting (no threshold)."""
        values: list[float] = []
        for entry in self.buffer.entries:
            if getattr(entry, "attributions", None) is None:
                continue
            try:
                current = grad_x_input_attributions(model, entry, int(entry.target_index[0])).detach()
            except Exception:  # noqa: BLE001
                continue
            stored = entry.attributions.detach().to(current.device)
            if self.variant == "ear_a":
                values.append(float(cosine_consistency_loss(current, stored)))
            else:
                values.append(float(ear_b_rank_loss(current, stored, temperature=self.temperature)))
        return {
            "n_measured": float(len(values)),
            "mean_consistency_loss": float(np.mean(values)) if values else float("nan"),
        }

    # -- beta selection ----------------------------------------------------
    def select_beta(self, scores: Mapping[float, float]) -> dict[str, Any]:
        """Select beta on validation macro-F1 (old periods) + validation drift."""
        candidates = {float(value): float(score) for value, score in scores.items()}
        if not candidates:
            raise ValueError("beta selection needs at least one candidate score")
        unknown = [value for value in candidates if value not in self.beta_grid]
        if unknown:
            raise ValueError(f"beta candidates {unknown} not in the configured grid {self.beta_grid}")
        best_beta = max(candidates, key=lambda value: (candidates[value], -value))
        self.beta = best_beta
        return {
            "beta_grid": sorted(self.beta_grid),
            "scores": {str(value): score for value, score in sorted(candidates.items())},
            "selected_beta": best_beta,
            "selected_score": candidates[best_beta],
            "selection_metric": "validation macro-F1 of old periods + validation drift (never test)",
        }


def build_ear_strategy(name: str, *, config: Mapping[str, Any] | None = None, seed: int = 42) -> EARStrategy:
    """Factory used by ``src/scripts/run_train.py --strategy ear_a|ear_b``.

    Reads the authoritative ``configs/ear.yaml`` (materialised from the analyst's
    ``src/analysis/config_defaults/ear.yaml`` proposal): ``replay`` (buffer/alpha/
    subgraph), ``training.beta_grid``, ``variants.ear_b.k`` and the attribution
    source. The older flat keys are still accepted for backward compatibility.
    """
    config = dict(config or {})
    ear_cfg = dict(config.get("ear", config))
    replay_cfg = dict(config.get("replay", {})) or dict(ear_cfg.get("replay", {}))
    training_cfg = dict(config.get("training", {})) or dict(ear_cfg.get("training", {}))
    variants_cfg = dict(config.get("variants", {})) or dict(ear_cfg.get("variants", {}))
    buffer_cfg = dict(config.get("buffer", {}))
    subgraph_cfg = dict(replay_cfg.get("subgraph", {}))
    variant = "ear_a" if name == "ear_a" else "ear_b"
    beta_grid = tuple(training_cfg.get("beta_grid", ear_cfg.get("beta_grid", BETA_GRID)))
    default_beta = float(
        training_cfg.get("default_beta", ear_cfg.get("default_beta", beta_grid[len(beta_grid) // 2]))
    )
    capacity = int(replay_cfg.get("buffer_edges", buffer_cfg.get("default", ear_cfg.get("buffer_edges", 2000))))
    num_hops = int(subgraph_cfg.get("num_hops", ear_cfg.get("num_hops", 2)))
    per_hop = subgraph_cfg.get("fanout_per_hop", ear_cfg.get("fanout", 10))
    if isinstance(per_hop, (list, tuple)):
        fanout = tuple(int(value) for value in per_hop)
    else:
        fanout = tuple(int(per_hop) for _ in range(max(num_hops, 1)))
    attribution_source = str(
        ear_cfg.get("stored_attribution_source", config.get("attribution_targets", {}).get("source", "ig"))
    )
    ear_b_cfg = dict(variants_cfg.get("ear_b", {}))
    return EARStrategy(
        variant=variant,
        beta=default_beta,
        buffer=ReplayBuffer(
            capacity=capacity,
            fanout=fanout,
            num_hops=num_hops,
            eviction=str(replay_cfg.get("eviction", "largest_class_random")),
        ),
        alpha=float(replay_cfg.get("alpha", ear_cfg.get("alpha", 1.0))),
        attribution_source=attribution_source,
        training_attribution=str(
            variants_cfg.get("ear_a", {}).get("pred_attribution", ear_cfg.get("training_attribution", "grad_x_input"))
        ),
        top_k=int(ear_b_cfg.get("k", ear_cfg.get("ear_b_top_k", 5))),
        temperature=float(ear_b_cfg.get("temperature", ear_cfg.get("ear_b_temperature", 0.1))),
        max_anchors_per_step=int(ear_cfg.get("max_anchors_per_step", training_cfg.get("max_anchors_per_step", 2))),
        beta_grid=beta_grid,
    )


__all__ = [
    "BETA_GRID",
    "EARStrategy",
    "build_ear_strategy",
    "cosine_consistency_loss",
    "ear_b_rank_loss",
    "grad_x_input_attributions",
    "soft_rank_vector",
]
