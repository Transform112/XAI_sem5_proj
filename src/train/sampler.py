"""Edge-level neighbor sampler for the canonical mini-batch trainer (T24/D-052).

Each step targets ``batch_size`` (default 1024) train edges. For a target batch we
build a bounded 2-hop neighbourhood:

* seed nodes = endpoints of the target edges;
* hop 1/2: from each frontier node sample up to ``fanout[hop]`` (<= 10) neighbour
  nodes (uniform without replacement);
* induced context edges = frozen graph edges whose both endpoints are in the
  sampled node set (``pairs`` cache makes this cheap even for the dense 42-host
  C1 graph);
* "bounded edge sampling": if the induced edge set exceeds ``max_edges_per_step``
  (default 60k), the non-target context edges are randomly subsampled to the cap.
  Target edges are always kept and are placed first, so their per-edge logits are
  ``logits[:n_targets]``.

The C1 graphs have only 40-42 hosts, so a naive node-neighbourhood subgraph would
cover almost all edges (no batching benefit); the explicit per-step edge cap keeps
the cost O(batch) while preserving 2-hop context around every target edge.

T26 ``fast_path`` (DEFAULT OFF, config ``train.sampler_fast_path``)
--------------------------------------------------------------------
An optional A+B shortcut for dense-host graphs, for use only if kit-v7 shows the
slow epoch bound:

* A (dense-host): if the 2-hop node walk covers *all* hosts, the induced context
  set is the whole edge set, so the expensive pair-mask broadcast over 1.1M edges
  is equivalent to sampling ``max_edges_per_step - n_targets`` context edges
  uniformly at random (targets always kept, duplicates excluded). Falls back to
  the standard construction when host coverage is incomplete.
* B (GPU-resident edges): when enabled on a CUDA device, ``edge_attr`` is uploaded
  once (~230 MB for the full graph) and per-step attributes are gathered with
  ``index_select`` on the device instead of a 12 MB host gather + copy per step.
  CPU tensors remain authoritative for edge ids so the seed -> subgraph mapping
  stays CPU-RNG driven.

WARNING: the fast path consumes the RNG stream differently from the standard
sampler, so a given seed maps to a different (but statistically equivalent under
full host coverage) sequence of subgraphs. Canonical training keeps this OFF until
explicitly enabled after a protocol/decision-log entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class SampledSubgraph:
    """One computation subgraph with target-edge logit positions."""

    x: Any
    edge_index: Any
    edge_attr: Any
    target_local_index: Any
    target_edge_ids: Any
    n_sub_nodes: int
    n_sub_edges: int
    n_context_edges: int
    n_targets: int
    sampler_mode: str = "standard"

    def to(self, device: Any) -> "SampledSubgraph":
        return SampledSubgraph(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            target_local_index=self.target_local_index.to(device),
            target_edge_ids=self.target_edge_ids,
            n_sub_nodes=self.n_sub_nodes,
            n_sub_edges=self.n_sub_edges,
            n_context_edges=self.n_context_edges,
            n_targets=self.n_targets,
            sampler_mode=self.sampler_mode,
        )


class EdgeNeighborSampler:
    """Deterministic-per-mode edge batch sampler for a frozen graph."""

    def __init__(
        self,
        data: Any,
        *,
        fanout: Sequence[int] = (10, 10),
        max_edges_per_step: int = 60_000,
        num_hops: int | None = None,
        fast_path: bool = False,
        device: str | None = None,
    ) -> None:
        import torch

        self.fanout = [int(value) for value in fanout]
        if not self.fanout or any(value < 1 for value in self.fanout):
            raise ValueError("fanout must contain positive per-hop neighbour counts")
        self.num_hops = int(num_hops if num_hops is not None else len(self.fanout))
        if int(max_edges_per_step) < 1:
            raise ValueError("max_edges_per_step must be positive")
        self.max_edges_per_step = int(max_edges_per_step)
        self.n_nodes = int(data.num_nodes)
        self.n_edges = int(data.num_edges)
        self.fast_path = bool(fast_path)
        self.device = None
        self._edge_attr_device = None
        self._all_edge_ids = np.arange(self.n_edges, dtype=np.int64)
        if self.fast_path and device is not None and str(device).startswith("cuda"):
            if torch.cuda.is_available():
                self.device = torch.device(device)
                # B: upload edge attributes once; ids stay on CPU for determinism.
                self._edge_attr_device = data.edge_attr.to(self.device)
        self.src = np.asarray(data.edge_index[0].tolist(), dtype=np.int64)
        self.dst = np.asarray(data.edge_index[1].tolist(), dtype=np.int64)
        self.edge_attr = data.edge_attr
        self.x = data.x
        self.y = data.y
        # unordered host-pair -> edge ids (C1 graphs are dense on 40-42 hosts).
        pairs: dict[tuple[int, int], list[int]] = {}
        for edge_id, (source, target) in enumerate(zip(self.src.tolist(), self.dst.tolist())):
            key = (source, target) if source <= target else (target, source)
            pairs.setdefault(key, []).append(edge_id)
        self.pair_edges = {key: np.asarray(values, dtype=np.int64) for key, values in pairs.items()}
        self.pairs = list(self.pair_edges.keys())
        self._pair_index = {key: position for position, key in enumerate(self.pairs)}
        self.edge_pair_index = np.asarray(
            [
                self._pair_index[(min(int(source), int(target)), max(int(source), int(target)))]
                for source, target in zip(self.src.tolist(), self.dst.tolist())
            ],
            dtype=np.int64,
        )
        adjacency: dict[int, set[int]] = {}
        for source, target in zip(self.src.tolist(), self.dst.tolist()):
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)
        self.adjacency = {node: np.asarray(sorted(values), dtype=np.int64) for node, values in adjacency.items()}

    # -- context construction ---------------------------------------------
    def _standard_context(self, targets: np.ndarray, rng: np.random.Generator, sampled_nodes: set[int]) -> np.ndarray:
        pair_mask = np.zeros(len(self.pairs), dtype=bool)
        node_list = sorted(sampled_nodes)
        for position, source in enumerate(node_list):
            for target in node_list[position:]:
                key = (source, target) if source <= target else (target, source)
                pair_position = self._pair_index.get(key)
                if pair_position is not None:
                    pair_mask[pair_position] = True
        induced = pair_mask[self.edge_pair_index]
        target_mask = np.zeros(self.n_edges, dtype=bool)
        target_mask[targets] = True
        context = np.nonzero(induced & ~target_mask)[0]
        return self._apply_context_cap(context, targets, rng)

    def _fast_context(self, targets: np.ndarray, rng: np.random.Generator, sampled_nodes: set[int]) -> np.ndarray | None:
        """A: dense-host shortcut (only when the 2-hop walk covers every host)."""
        if len(sampled_nodes) < self.n_nodes:
            return None
        budget = max(self.max_edges_per_step - len(targets), 0)
        if budget <= 0:
            return np.empty(0, dtype=np.int64)
        draw = rng.choice(self.n_edges, size=min(self.n_edges, budget + len(targets)), replace=False)
        context = np.sort(draw[~np.isin(draw, targets)][:budget])
        if len(context) < budget:
            remaining = np.setdiff1d(self._all_edge_ids, np.concatenate([targets, context]), assume_unique=False)
            extra_size = min(budget - len(context), len(remaining))
            if extra_size > 0:
                extra = rng.choice(len(remaining), size=extra_size, replace=False)
                context = np.sort(np.concatenate([context, remaining[extra]]))
        return context

    def _apply_context_cap(self, context: np.ndarray, targets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        budget_context = max(self.max_edges_per_step - len(targets), 0)
        if len(context) <= budget_context:
            return context
        if budget_context <= 0:
            return np.empty(0, dtype=np.int64)
        picked = rng.choice(len(context), size=budget_context, replace=False)
        return np.sort(context[picked])

    # -- sampling ----------------------------------------------------------
    def select_edge_ids(self, target_edges: Sequence[int] | np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        """Return ``(selected_global_edge_ids, sampler_mode)`` for one target batch.

        Targets come first (logit positions 0..n_targets-1); context ids follow.
        This is the reference id-selection used both by the relabelled
        ``sample`` path and by the T27 zero-relabel device batch builder.
        """
        targets = np.asarray([int(value) for value in target_edges], dtype=np.int64)
        if targets.size == 0:
            raise ValueError("target edge batch is empty")
        node_set = set(self.src[targets].tolist()) | set(self.dst[targets].tolist())
        frontier = sorted(node_set)
        sampled_nodes = set(node_set)
        for hop in range(self.num_hops):
            budget = self.fanout[min(hop, len(self.fanout) - 1)]
            next_frontier: list[int] = []
            for node in frontier:
                neighbours = self.adjacency.get(node, np.empty(0, dtype=np.int64))
                candidates = [int(value) for value in neighbours.tolist() if int(value) not in sampled_nodes]
                if not candidates:
                    continue
                if len(candidates) > budget:
                    picked = rng.choice(len(candidates), size=budget, replace=False)
                    candidates = [candidates[int(position)] for position in picked]
                for neighbour in candidates:
                    if neighbour not in sampled_nodes:
                        sampled_nodes.add(neighbour)
                        next_frontier.append(neighbour)
            frontier = next_frontier
            if not frontier:
                break
        sampler_mode = "standard"
        context = self._fast_context(targets, rng, sampled_nodes) if self.fast_path else None
        if context is not None:
            sampler_mode = "fast_dense_host" if self._edge_attr_device is None else "fast_dense_host_gpu_edges"
        else:
            context = self._standard_context(targets, rng, sampled_nodes)
        return np.concatenate([targets, context]), sampler_mode

    def sample(self, target_edges: Sequence[int] | np.ndarray, rng: np.random.Generator) -> SampledSubgraph:
        """Sample one bounded 2-hop computation subgraph for a target edge batch."""
        import torch

        targets = np.asarray([int(value) for value in target_edges], dtype=np.int64)
        selected, sampler_mode = self.select_edge_ids(targets, rng)
        selected_nodes = sorted(set(self.src[selected].tolist()) | set(self.dst[selected].tolist()))
        remap = {node: position for position, node in enumerate(selected_nodes)}
        local_src = [remap[int(value)] for value in self.src[selected].tolist()]
        local_dst = [remap[int(value)] for value in self.dst[selected].tolist()]
        edge_index = torch.as_tensor([local_src, local_dst], dtype=torch.long)
        selected_tensor = torch.as_tensor(selected, dtype=torch.long)
        if self._edge_attr_device is not None:
            edge_attr = self._edge_attr_device[selected_tensor.to(self._edge_attr_device.device)]
        else:
            edge_attr = self.edge_attr[selected_tensor]
        return SampledSubgraph(
            x=self.x[torch.as_tensor(selected_nodes, dtype=torch.long)],
            edge_index=edge_index,
            edge_attr=edge_attr,
            target_local_index=torch.arange(len(targets), dtype=torch.long),
            target_edge_ids=torch.as_tensor(targets, dtype=torch.long),
            n_sub_nodes=len(selected_nodes),
            n_sub_edges=int(len(selected)),
            n_context_edges=int(len(selected) - len(targets)),
            n_targets=int(len(targets)),
            sampler_mode=sampler_mode,
        )


__all__ = ["EdgeNeighborSampler", "SampledSubgraph"]
