"""S4 continual-update strategy scaffolding (FT / Replay / EWC).

Only scaffolding at this stage (T14): the strategy classes are unit-tested on
toy graphs and CLI-ready, but the full (strategy x model x seed) matrix is not
run until the G2 compute plan is set (Doc 1 cut rule).

* :class:`FineTuneStrategy` - plain fine-tuning on the current period.
* :class:`ReplayBuffer` - class-balanced memory of 2-hop computation subgraphs
  around stored edges (Doc 2 S4.2), with a replay loss term.
* :class:`EWCStrategy` - diagonal Fisher information estimated on training
  edges and a quadratic penalty towards the previous parameters (Doc 2 S4.3).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.train.trainer import TrainSettings, train_edge_classifier
from src.utils.logging import get_logger
from src.utils.seed import seed_everything

LOGGER = get_logger("xdrift.cl")


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------
class FineTuneStrategy:
    name = "ft"

    def update(self, model, data, *, settings: TrainSettings, seed: int, device: str, **kwargs) -> dict[str, Any]:
        result = train_edge_classifier(model, data, settings=settings, seed=seed, device=device, **kwargs)
        return {"strategy": self.name, **result}


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
@dataclass
class EdgeReplayEntry:
    """D-072 replay entry: old edge, identified by global host ids.

    The replay logits are recomputed from the *current* period's full-graph node
    embeddings using this stored old edge feature vector (no stored subgraph, no
    full-edge head, no re-sampling of the anchor context).
    """

    period: int
    src_global: int
    dst_global: int
    edge_attr: torch.Tensor
    label: int
    source_edge: int = -1


@dataclass
class ReplayBuffer:
    """Class-balanced memory of sampled 2-hop subgraphs around stored edges."""

    capacity: int = 2000
    fanout: tuple[int, ...] = (10, 10)
    num_hops: int = 2
    eviction: str = "largest_class_random"
    # ``edge`` = D-072 full-graph-consistent replay (default for the matrix);
    # ``subgraph`` = legacy stored-subgraph mode, diagnostic only.
    representation: str = "subgraph"
    # Dense host graphs can make the induced 2-hop subgraph contain most of the
    # period's edges; bound the stored context deterministically (target always
    # kept) so the <=2000-entry buffer stays small. Toy/sparse graphs unaffected.
    max_subgraph_edges: int = 2000
    entries: list[Any] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.representation not in ("subgraph", "edge"):
            raise ValueError("representation must be 'subgraph' or 'edge'")

    def __len__(self) -> int:
        return len(self.entries)

    # -- helpers -----------------------------------------------------------
    def _fanout_for_hop(self, hop: int) -> int:
        if isinstance(self.fanout, (list, tuple)):
            if not self.fanout:
                return 0
            return int(self.fanout[min(hop, len(self.fanout) - 1)])
        return int(self.fanout)

    def _build_graph_index(self, data: Any) -> tuple[dict[int, list[int]], np.ndarray, np.ndarray, dict[int, list[int]]]:
        """Adjacency + endpoint arrays, computed once (vectorised for >1M edges).

        The 40-44 host graph has >1M edges, so Python per-edge adjacency/incidence
        construction is avoided; induced-edge membership is computed with boolean
        endpoint masks in ``_sampled_subgraph`` (incidence is therefore unused).
        """
        src = np.asarray(data.edge_index[0].detach().cpu().numpy(), dtype=np.int64)
        dst = np.asarray(data.edge_index[1].detach().cpu().numpy(), dtype=np.int64)
        nodes = np.concatenate([src, dst])
        neighbours = np.concatenate([dst, src])
        pairs = np.unique(np.stack([nodes, neighbours], axis=1), axis=0)
        adjacency: dict[int, list[int]] = {}
        for source, target in pairs:
            adjacency.setdefault(int(source), []).append(int(target))
        return {node: adjacency[node] for node in sorted(adjacency)}, src, dst, {}

    def _sampled_subgraph(
        self,
        data: Any,
        target_edge: int,
        *,
        adjacency: Mapping[int, Sequence[int]],
        src_list: np.ndarray,
        dst_list: np.ndarray,
        incidence: Mapping[int, Sequence[int]],
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Bounded BFS subgraph with <= fanout sampled neighbours per hop.

        T32: vectorised induced-edge selection + searchsorted local remap (no Python
        loops over edges); deterministic context cap keeps dense-graph entries small.
        """
        import torch

        src = int(data.edge_index[0, target_edge])
        dst = int(data.edge_index[1, target_edge])
        node_set = {src, dst}
        frontier = [src, dst]
        for hop in range(int(self.num_hops)):
            per_hop = self._fanout_for_hop(hop)
            next_frontier: list[int] = []
            for node in frontier:
                candidates = [value for value in adjacency.get(node, ()) if value not in node_set]
                if len(candidates) > per_hop:
                    pick = rng.choice(len(candidates), size=per_hop, replace=False)
                    candidates = [candidates[int(position)] for position in pick]
                for neighbour in candidates:
                    if neighbour not in node_set:
                        node_set.add(neighbour)
                        next_frontier.append(neighbour)
            frontier = next_frontier
            if not frontier:
                break
        node_list = np.asarray(sorted(node_set), dtype=np.int64)
        node_mask = np.zeros(int(data.x.shape[0]), dtype=bool)
        node_mask[node_list] = True
        src_np = np.asarray(src_list, dtype=np.int64)
        dst_np = np.asarray(dst_list, dtype=np.int64)
        edge_mask = node_mask[src_np] & node_mask[dst_np]
        sub_edges = np.flatnonzero(edge_mask)
        cap = int(self.max_subgraph_edges)
        if cap > 0 and len(sub_edges) > cap:
            # Deterministic target-preserving context cap (same policy as S6 anchors).
            target_position = int(np.searchsorted(sub_edges, target_edge))
            others = np.delete(sub_edges, target_position)
            n_take = min(max(cap - 1, 1), len(others))
            if n_take < len(others):
                picked = np.sort(rng.choice(len(others), size=n_take, replace=False))
                others = others[picked]
            sub_edges = np.sort(np.concatenate(([target_edge], others)))
        if not np.any(sub_edges == target_edge):  # defensive; endpoints are in node_set
            sub_edges = np.sort(np.concatenate((sub_edges, [target_edge])))
        sub_edges_tensor = torch.as_tensor(sub_edges, dtype=torch.long)
        node_tensor = torch.as_tensor(node_list, dtype=torch.long)
        local_src = np.searchsorted(node_list, src_np[sub_edges])
        local_dst = np.searchsorted(node_list, dst_np[sub_edges])
        local_edge_index = torch.as_tensor(np.stack([local_src, local_dst]), dtype=torch.long)
        target_local = int(np.searchsorted(sub_edges, target_edge))
        return {
            "sub_edge_index": local_edge_index,
            "sub_edge_attr": data.edge_attr[sub_edges_tensor],
            "sub_edge_ids": sub_edges_tensor,
            "sub_node_local_ids": node_tensor,
            "sub_node_global_ids": data.node_global_ids[node_tensor],
            "target_local_index": torch.tensor([target_local], dtype=torch.long),
            "target_edge": torch.tensor([target_edge], dtype=torch.long),
            "n_sub_nodes": int(node_tensor.numel()),
            "n_sub_edges": int(sub_edges_tensor.numel()),
        }

    # -- population --------------------------------------------------------
    def add_from_graph(self, data: Any, *, split: str = "train", seed: int = 42) -> dict[str, Any]:
        torch_mod = torch
        rng = np.random.default_rng(seed)
        mask = getattr(data, f"{split}_mask")
        indices = torch_mod.nonzero(mask, as_tuple=False).flatten()
        labels = data.y[indices]
        present = torch_mod.unique(labels)
        per_class = max(1, int(self.capacity) // max(len(present), 1))
        chosen: list[int] = []
        for class_id in present.tolist():
            class_indices = indices[labels == class_id]
            take = min(per_class, len(class_indices))
            pick = rng.choice(len(class_indices), size=take, replace=False)
            chosen.extend(class_indices[torch_mod.as_tensor(pick, dtype=torch_mod.long)].tolist())
        rng.shuffle(chosen)
        if self.representation == "edge":
            # D-072: store old edge features + global host ids; replay logits are
            # computed later on the current period's full-graph node embeddings.
            added = 0
            period_id = int(getattr(data, "period", -1)) + 1
            for edge_index_value in chosen[: int(self.capacity)]:
                label = int(data.y[edge_index_value])
                self.entries.append(
                    EdgeReplayEntry(
                        period=period_id,
                        src_global=int(data.edge_host_src_global[edge_index_value]),
                        dst_global=int(data.edge_host_dst_global[edge_index_value]),
                        edge_attr=data.edge_attr[edge_index_value].detach().cpu().clone(),
                        label=label,
                        source_edge=int(edge_index_value),
                    )
                )
                added += 1
            evicted = self._enforce_capacity(seed=seed)
            return {
                "capacity": self.capacity,
                "entries": len(self.entries),
                "added": added,
                "evicted": evicted,
                "eviction_rule": self.eviction,
                "classes_requested_per_class": per_class,
                "representation": "edge",
            }
        adjacency, src_list, dst_list, incidence = self._build_graph_index(data)
        added = 0
        for edge_index_value in chosen[: int(self.capacity)]:
            sampled = self._sampled_subgraph(
                data,
                int(edge_index_value),
                adjacency=adjacency,
                src_list=src_list,
                dst_list=dst_list,
                incidence=incidence,
                rng=np.random.default_rng(seed + int(edge_index_value)),
            )
            entry = type(data)(
                x=data.x[sampled["sub_node_local_ids"]],
                edge_index=sampled["sub_edge_index"],
                edge_attr=sampled["sub_edge_attr"],
                y=data.y[edge_index_value : edge_index_value + 1],
            )
            # Regression contract (T17-S4-2): the stored label must be paired with
            # the stored edge's own feature row and logits position.
            assert int(entry.y.item()) == int(data.y[edge_index_value])
            assert torch_mod.equal(
                entry.edge_attr[sampled["target_local_index"]][0], data.edge_attr[edge_index_value]
            )
            entry.target_index = sampled["target_local_index"]
            entry.source_edge = torch_mod.tensor([edge_index_value], dtype=torch_mod.long)
            entry.target_class = int(data.y[edge_index_value])
            entry.node_global_ids = sampled["sub_node_global_ids"]
            entry.n_sub_nodes = sampled["n_sub_nodes"]
            entry.n_sub_edges = sampled["n_sub_edges"]
            entry.attributions = None  # S12: explanation-anchored replay fills this
            self.entries.append(entry)
            added += 1
        evicted = self._enforce_capacity(seed=seed)
        return {
            "capacity": self.capacity,
            "entries": len(self.entries),
            "added": added,
            "evicted": evicted,
            "eviction_rule": self.eviction,
            "classes_requested_per_class": per_class,
            "num_hops": self.num_hops,
            "fanout": list(self.fanout) if isinstance(self.fanout, (list, tuple)) else [int(self.fanout)],
            "representation": "subgraph",
        }

    def _entry_class(self, entry: Any) -> int:
        if isinstance(entry, EdgeReplayEntry):
            return int(entry.label)
        return int(entry.y.reshape(-1)[0])

    def _enforce_capacity(self, *, seed: int = 42) -> int:
        """Evict random entries from the currently largest class (Doc 2 S4.2).

        T34 (bug-report #5): the sequential O(N^2) rescan per eviction is replaced by
        one counts/member-index build plus tombstoned removals; the same deterministic
        victim sequence is preserved (largest class by count, tie to lower label, then
        the same seeded uniform choice within the class, which is kept in ascending
        entry order as before).
        """
        rule = {"random_from_largest_class": "largest_class_random"}.get(self.eviction, self.eviction)
        if rule not in ("largest_class_random", "oldest"):
            raise ValueError(f"unknown eviction rule {self.eviction!r}")
        rng = np.random.default_rng(seed + 1_000_003)
        evicted = 0
        if len(self.entries) <= self.capacity:
            return 0
        if rule == "oldest":
            excess = len(self.entries) - self.capacity
            del self.entries[:excess]
            return excess
        counts: dict[int, int] = {}
        members: dict[int, list[int]] = {}
        for index, entry in enumerate(self.entries):
            label = self._entry_class(entry)
            counts[label] = counts.get(label, 0) + 1
            members.setdefault(label, []).append(index)
        live = len(self.entries)
        while live > self.capacity:
            largest_class = max(counts, key=lambda label: (counts[label], -label))
            positions = members[largest_class]
            pick = int(rng.integers(len(positions)))
            victim = positions.pop(pick)
            self.entries[victim] = None
            live -= 1
            counts[largest_class] -= 1
            evicted += 1
            if counts[largest_class] == 0:
                del counts[largest_class]
                del members[largest_class]
        self.entries = [entry for entry in self.entries if entry is not None]
        return evicted

    def sample(self, n: int, seed: int = 42) -> list[Any]:
        if not self.entries:
            return []
        rng = np.random.default_rng(seed)
        take = min(int(n), len(self.entries))
        positions = rng.choice(len(self.entries), size=take, replace=False)
        return [self.entries[int(position)] for position in positions]

    def class_balance(self) -> dict[str, int]:
        counts: dict[int, int] = {}
        for entry in self.entries:
            label = self._entry_class(entry)
            counts[label] = counts.get(label, 0) + 1
        return {str(key): value for key, value in sorted(counts.items())}

    def attach_attributions(self, attributions: Mapping[Any, Any]) -> int:
        """Attach explainer attributions to buffer entries by source edge id (S12)."""
        attached = 0
        for entry in self.entries:
            if isinstance(entry, EdgeReplayEntry) or not hasattr(entry, "source_edge"):
                continue
            source = int(entry.source_edge[0])
            if source in attributions:
                entry.attributions = torch.as_tensor(attributions[source], dtype=torch.float32)
                attached += 1
        return attached


class ReplayStrategy:
    name = "replay"

    def __init__(
        self,
        buffer: ReplayBuffer | None = None,
        alpha: float = 1.0,
        replay_batch: int = 32,
        *,
        representation: str = "subgraph",
    ) -> None:
        if buffer is not None:
            self.buffer = buffer
        else:
            self.buffer = ReplayBuffer(representation=representation)
        self.alpha = float(alpha)
        self.replay_batch = int(replay_batch)

    def update(self, model, data, *, settings: TrainSettings, seed: int, device: str, **kwargs) -> dict[str, Any]:
        # Replay on the PREVIOUS memory, then add the current period's samples
        # (Doc 2 S4.2: "After finishing period t, add class-balanced samples").
        if self.buffer.representation == "edge":
            # D-072: a fresh buffer batch each optimizer step; replay logits are
            # recomputed from the current full-graph node embeddings inside the
            # trainer (batch-only head). No fixed batch is held for the period.
            provider = lambda step: self.buffer.sample(self.replay_batch, seed=int(seed) + int(step))  # noqa: E731
            result = train_edge_classifier(
                model,
                data,
                settings=settings,
                seed=seed,
                device=device,
                replay_provider=provider,
                replay_alpha=self.alpha,
                **kwargs,
            )
            replay_entries_used = int(self.replay_batch)
        else:
            entries = self.buffer.sample(self.replay_batch, seed=seed)
            result = train_edge_classifier(
                model,
                data,
                settings=settings,
                seed=seed,
                device=device,
                replay_entries=entries,
                replay_alpha=self.alpha,
                **kwargs,
            )
            replay_entries_used = len(entries)
        buffer_stats = self.buffer.add_from_graph(data, seed=seed)
        return {
            "strategy": self.name,
            "buffer": buffer_stats,
            "buffer_class_balance": self.buffer.class_balance(),
            "replay_entries_used": replay_entries_used,
            "replay_representation": self.buffer.representation,
            "alpha": self.alpha,
            **result,
        }


# ---------------------------------------------------------------------------
# EWC
# ---------------------------------------------------------------------------
class EWCStrategy:
    """EWC with Doc 2 S4.3 ordering/accumulation.

    Correct order (T17-S4-1): the penalty for the update to period ``t`` uses the
    Fisher accumulated from periods ``< t`` and the parameters theta* from the
    previous checkpoint; AFTER training period ``t`` the Fisher of period ``t``
    (true-label train edges, ~2000) is estimated and **accumulated (sum)** into the
    running Fisher, and theta* is re-anchored to the new parameters.
    """

    name = "ewc"

    def __init__(
        self,
        *,
        fisher_edges: int = 2000,
        lam: float = 1000.0,
        accumulate: str = "sum",
        lambda_candidates: Sequence[float] = (100.0, 1000.0, 10000.0),
    ) -> None:
        if accumulate not in ("sum", "mean"):
            raise ValueError("accumulate must be 'sum' or 'mean'")
        self.fisher_edges = int(fisher_edges)
        self.lam = float(lam)
        self.accumulate = accumulate
        self.fisher: dict[str, torch.Tensor] | None = None
        self.theta_star: dict[str, torch.Tensor] | None = None
        self.n_updates = 0
        self.n_fisher_updates = 0
        self.fisher_history: list[dict[str, torch.Tensor]] = []
        self.lambda_candidates = [float(value) for value in lambda_candidates]

    # -- Fisher estimation (does NOT mutate accumulated state) -------------
    def estimate_fisher(self, model, data, *, seed: int = 42, device: str = "cpu") -> dict[str, torch.Tensor]:
        """Mean squared per-edge gradient of log p(true label) on train edges."""
        seed_everything(seed)
        model.to(device)
        model.eval()
        index = torch.nonzero(data.train_mask, as_tuple=False).flatten()
        if len(index) > self.fisher_edges:
            rng = np.random.default_rng(seed)
            pick = rng.choice(len(index), size=self.fisher_edges, replace=False)
            index = index[torch.as_tensor(np.sort(pick), dtype=torch.long)]
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        y = data.y.to(device)
        fisher = {name: torch.zeros_like(parameter, device=device) for name, parameter in model.named_parameters()}
        use_head_only = hasattr(model, "node_embeddings") and hasattr(model, "edge_logits")
        if use_head_only:
            # T32: head-only logits for the sampled Fisher edges (no full-edge head /
            # 1.3 GiB concat); the per-edge squared-gradient math is unchanged.
            with torch.enable_grad():
                hidden = model.node_embeddings(x, edge_index, edge_attr)
                all_logits = model.edge_logits(
                    hidden, edge_index, edge_attr, index.to(device)
                )
                log_probs = torch.log_softmax(all_logits, dim=-1)
                for position, edge in enumerate(index.tolist()):
                    model.zero_grad(set_to_none=True)
                    log_probs[position, int(y[edge])].backward(retain_graph=True)
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            fisher[name] += parameter.grad.detach() ** 2
                model.zero_grad(set_to_none=True)
        else:
            for edge in index.tolist():
                model.zero_grad()
                logits = model(x, edge_index, edge_attr)[edge].unsqueeze(0)
                log_prob = torch.log_softmax(logits, dim=-1)[0, int(y[edge])]
                log_prob.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        fisher[name] += parameter.grad.detach() ** 2
        count = max(len(index), 1)
        return {name: (value / count).detach().cpu() for name, value in fisher.items()}

    def accumulate_fisher(self, fisher_new: Mapping[str, torch.Tensor], *, n_edges: int) -> None:
        if self.fisher is None:
            self.fisher = {name: value.detach().cpu().clone() for name, value in fisher_new.items()}
        elif self.accumulate == "sum":
            self.fisher = {
                name: self.fisher[name] + fisher_new[name].detach().cpu() for name in self.fisher
            }
        else:  # running mean over updates
            # T34 (bug-report #4): the running mean must use the count *after* this
            # accumulation; the old 1/n_updates made the second update replace the
            # first (weight 1.0) instead of averaging. Track the Fisher count locally so
            # direct accumulate_fisher calls (and the update() order) are both correct.
            weight = 1.0 / float(self.n_fisher_updates + 1)
            self.fisher = {
                name: (1 - weight) * self.fisher[name] + weight * fisher_new[name].detach().cpu()
                for name in self.fisher
            }
        self.fisher_history.append({name: value.clone() for name, value in self.fisher.items()})
        self.n_fisher_edges = int(n_edges)
        self.n_fisher_updates += 1

    def anchor(self, model) -> None:
        self.theta_star = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}

    def penalty(self, model) -> torch.Tensor:
        if self.fisher is None or self.theta_star is None:
            return torch.zeros((), dtype=torch.float32)
        total = torch.zeros((), dtype=torch.float32)
        device = next(model.parameters()).device
        for name, parameter in model.named_parameters():
            fisher = self.fisher[name].to(device)
            star = self.theta_star[name].to(device)
            total = total + (fisher * (parameter - star) ** 2).sum()
        return 0.5 * self.lam * total

    def select_lambda(self, scores: Mapping[float, float]) -> dict[str, Any]:
        """Pick lambda by validation macro-F1 of OLD periods only (no future/current data)."""
        candidates = {float(value): float(score) for value, score in scores.items()}
        if not candidates:
            raise ValueError("lambda selection needs at least one candidate score")
        unknown = [value for value in candidates if value not in self.lambda_candidates]
        if unknown:
            raise ValueError(f"lambda candidates {unknown} not in the configured grid {self.lambda_candidates}")
        best_lambda = max(candidates, key=lambda value: (candidates[value], -value))
        self.lam = best_lambda
        return {
            "lambda_grid": sorted(self.lambda_candidates),
            "scores": {str(value): score for value, score in sorted(candidates.items())},
            "selected_lambda": best_lambda,
            "selected_score": candidates[best_lambda],
            "selection_metric": "validation macro-F1 of old periods only",
        }

    @staticmethod
    def lambda_selection_scores(
        evaluate_old_val_macro_f1: Any,
        *,
        lambda_candidates: Sequence[float] = (100.0, 1000.0, 10000.0),
    ) -> dict[float, float]:
        """Run a candidate evaluator per lambda and return {lambda: old-period val F1}."""
        return {float(value): float(evaluate_old_val_macro_f1(float(value))) for value in lambda_candidates}

    def update(self, model, data, *, settings: TrainSettings, seed: int, device: str, **kwargs) -> dict[str, Any]:
        penalty_fn = self.penalty if (self.fisher is not None and self.theta_star is not None) else None
        result = train_edge_classifier(
            model,
            data,
            settings=settings,
            seed=seed,
            device=device,
            penalty_fn=penalty_fn,
            **kwargs,
        )
        # T17-S4-1: Fisher AFTER training this period, estimated on its true-label
        # train edges, then accumulated across updates and theta* re-anchored.
        fisher_new = self.estimate_fisher(model, data, seed=seed, device=device)
        train_mask = getattr(data, "train_mask", None)
        n_train_edges = int(train_mask.sum()) if train_mask is not None else self.fisher_edges
        self.accumulate_fisher(fisher_new, n_edges=min(n_train_edges, self.fisher_edges))
        self.anchor(model)
        self.n_updates += 1
        result["fisher"] = {
            "estimated_after_update": True,
            "accumulate": self.accumulate,
            "n_accumulated_updates": self.n_updates,
            "fisher_edges": self.fisher_edges,
            "lambda": self.lam,
        }
        return {"strategy": self.name, "lambda": self.lam, "fisher_edges": self.fisher_edges, **result}


def build_strategy(name: str, *, config: Mapping[str, Any] | None = None, seed: int = 42) -> Any:
    """Factory for CLI/run-book strategy construction."""
    config = dict(config or {})
    if name == "ft":
        return FineTuneStrategy()
    if name == "replay":
        replay_cfg = dict(config.get("replay", config))
        representation = str(replay_cfg.get("representation", "edge"))
        return ReplayStrategy(
            buffer=ReplayBuffer(
                capacity=int(replay_cfg.get("buffer_edges", 2000)),
                fanout=tuple(replay_cfg.get("fanout", (10, 10))),
                eviction=str(replay_cfg.get("eviction", "largest_class_random")),
                representation=representation,
                max_subgraph_edges=int(replay_cfg.get("max_subgraph_edges", 2000)),
            ),
            alpha=float(replay_cfg.get("alpha", 1.0)),
        )
    if name == "ewc":
        ewc_cfg = dict(config.get("ewc", config))
        return EWCStrategy(
            fisher_edges=int(ewc_cfg.get("fisher_edges", 2000)),
            lam=float(ewc_cfg.get("lambda", 1000.0)),
            accumulate=str(ewc_cfg.get("accumulate", "sum")),
            lambda_candidates=tuple(
                ewc_cfg.get("lambda_candidates", (100.0, 1000.0, 10000.0))
            ),
        )
    if name in ("ear_a", "ear_b"):
        from src.cl.ear import build_ear_strategy

        return build_ear_strategy(name, config=config, seed=seed)
    raise KeyError(f"unknown strategy {name!r}; available: ft, replay, ewc, ear_a, ear_b")


__all__ = [
    "EWCStrategy",
    "EdgeReplayEntry",
    "FineTuneStrategy",
    "ReplayBuffer",
    "ReplayStrategy",
    "build_strategy",
]
