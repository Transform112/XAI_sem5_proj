"""T27 optimized sampled-batch pipeline (static device cache + zero-relabel batches).

The frozen C1 graphs have only 40-42 hosts, so sampled edge batches do not need a
relabelled subgraph at all: cached device tensors are indexed with **global node
ids** and the model is called as ``model(x_all_nodes, edge_index_selected, edge_attr_selected)``
with the explicit ``size=(N, N)`` in the E-GraphSAGE layer. This removes per-step
Python relabelling, Data objects, host gathers and most CUDA sync points.

The reference/unoptimized path stays available: ``EdgeNeighborSampler.sample``
still builds the relabelled ``SampledSubgraph`` used before T27 and the trainer
keeps it whenever ``optimized_dispatch``/``sampler_fast_path`` are disabled. The
optimized and reference paths are tested to produce identical captured-batch
logits/loss (<=1e-6).

Sampling mapping change (documented): the optimized dense-host path draws context
edge ids with a vectorised GPU RNG (``torch.randperm``) instead of the CPU numpy
pair-mask walk, so a given seed maps to a different (statistically equivalent on
dense hosts) sequence of subgraphs. The standard path remains the reproducible
reference for audit.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class DeviceGraph:
    """All tensors needed for training/validation, moved to the device exactly once."""

    def __init__(self, data: Any, device: Any) -> None:
        import torch

        self.device = torch.device(device)
        self.x = data.x.to(self.device)
        self.edge_index = data.edge_index.to(self.device)
        self.edge_attr = data.edge_attr.to(self.device)
        self.y = data.y.to(self.device)
        self.n_nodes = int(data.num_nodes)
        self.n_edges = int(data.num_edges)
        self.train_idx_cpu = torch.nonzero(data.train_mask, as_tuple=False).flatten().cpu().numpy()
        self.val_idx = torch.nonzero(data.val_mask, as_tuple=False).flatten().to(self.device)
        self.test_idx = torch.nonzero(data.test_mask, as_tuple=False).flatten().to(self.device)
        self.train_label_counts = np.bincount(
            data.y[torch.nonzero(data.train_mask, as_tuple=False).flatten()].cpu().numpy(),
            minlength=int(data.y.max()) + 1,
        )

    def batch_from_ids(self, selected_ids: Any, n_targets: int):
        """Return ``(edge_index_sub, edge_attr_sub, target_y)`` using global ids."""
        import torch

        if not isinstance(selected_ids, torch.Tensor):
            selected_ids = torch.as_tensor(selected_ids, dtype=torch.long)
        selected_ids = selected_ids.to(self.device)
        edge_index_sub = self.edge_index.index_select(1, selected_ids)
        edge_attr_sub = self.edge_attr.index_select(0, selected_ids)
        target_y = self.y[selected_ids[: int(n_targets)]]
        return selected_ids, edge_index_sub, edge_attr_sub, target_y


def build_dense_host_plan(sampler: Any, *, n_hops: int | None = None) -> bool:
    """Precompute per-period host-coverage decision for the dense-host fast path.

    The fast path is used when two-hop reachability (without fanout limiting) from
    every host covers all hosts - true for the dense C1 graphs, false for sparse
    toy/ablation graphs, which keep the standard sampler.
    """
    n_nodes = int(sampler.n_nodes)
    if n_nodes > 128:
        return False
    adjacency = sampler.adjacency
    for start in range(n_nodes):
        reached = {start}
        frontier = [start]
        for _hop in range(int(n_hops if n_hops is not None else sampler.num_hops)):
            next_frontier: list[int] = []
            for node in frontier:
                for neighbour in adjacency.get(node, np.empty(0, dtype=np.int64)).tolist():
                    if int(neighbour) not in reached:
                        reached.add(int(neighbour))
                        next_frontier.append(int(neighbour))
            frontier = next_frontier
            if not frontier:
                break
        if len(reached) < n_nodes:
            return False
    return True


class FastSampledBatcher:
    """Zero-relabel sampled batch construction on cached device tensors."""

    def __init__(
        self,
        sampler: Any,
        graph: DeviceGraph,
        *,
        max_edges_per_step: int = 60_000,
        dense_host: bool = True,
        seed: int = 42,
    ) -> None:
        import torch

        self.sampler = sampler
        self.graph = graph
        self.max_edges_per_step = int(max_edges_per_step)
        self.dense_host = bool(dense_host)
        self.seed = int(seed)
        self._generator = torch.Generator(device=str(graph.device))
        self._generator.manual_seed(int(seed))
        # Precompute per-epoch batch orders once (CPU numpy, reused every epoch).
        self._epoch_permutations: dict[int, np.ndarray] = {}
        self._cpu_rng = np.random.default_rng(int(seed))

    def epoch_permutation(self, epoch: int, n_train: int) -> np.ndarray:
        permutation = self._epoch_permutations.get(int(epoch))
        if permutation is None:
            rng = np.random.default_rng(int(self.seed) * 1_000_003 + int(epoch))
            permutation = rng.permutation(n_train).astype(np.int64)
            self._epoch_permutations[int(epoch)] = permutation
        return permutation

    def _context_ids(self, n_targets: int, target_ids: np.ndarray) -> Any:
        """Vectorised context edge ids on the device (dense-host path)."""
        import torch

        budget = max(self.max_edges_per_step - int(n_targets), 0)
        if budget <= 0:
            return torch.empty(0, dtype=torch.long, device=self.graph.device)
        k = min(self.graph.n_edges, budget + int(n_targets))
        perm = torch.randperm(self.graph.n_edges, generator=self._generator, device=self.graph.device)[:k]
        targets_device = torch.as_tensor(target_ids, dtype=torch.long, device=self.graph.device)
        context = perm[~torch.isin(perm, targets_device)][:budget]
        if context.numel() < budget:
            # Robust fallback for tiny/exhausted edge pools (T26 regression class).
            remaining = torch.tensor(
                np.setdiff1d(
                    np.arange(self.graph.n_edges, dtype=np.int64),
                    np.concatenate([np.asarray(target_ids, dtype=np.int64), context.cpu().numpy()]),
                ),
                device=self.graph.device,
            )
            extra = min(int(budget - context.numel()), int(remaining.numel()))
            if extra > 0:
                picks = torch.randperm(int(remaining.numel()), generator=self._generator, device=self.graph.device)[:extra]
                context = torch.cat([context, remaining.index_select(0, picks)])
        return context

    def batch(self, target_ids: np.ndarray):
        """Return global-id tensors for one 1024-target-edge batch (targets first)."""
        import torch

        target_tensor = torch.as_tensor(target_ids, dtype=torch.long, device=self.graph.device)
        if self.dense_host:
            context = self._context_ids(len(target_ids), target_ids)
            selected = torch.cat([target_tensor, context])
        else:
            # Standard CPU sampler ids (relabelling not needed for the device batch).
            selected_np, _ = self.sampler.select_edge_ids(target_ids, self._cpu_rng)
            selected = torch.as_tensor(selected_np, dtype=torch.long, device=self.graph.device)
        return self.graph.batch_from_ids(selected, len(target_ids))


def sampled_eval_predictions(
    model: Any,
    graph: DeviceGraph,
    batcher: FastSampledBatcher,
    indices: Any,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    """Lean sampled evaluation on cached device tensors (low-memory lane fallback)."""
    import torch

    ids = indices.detach().cpu().numpy()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(ids), int(batch_size)):
            batch_ids = ids[offset : offset + int(batch_size)]
            _, edge_index_sub, edge_attr_sub, _ = graph.batch_from_ids(
                torch.as_tensor(batch_ids, dtype=torch.long), len(batch_ids)
            )
            logits = model(graph.x, edge_index_sub, edge_attr_sub)
            predictions.append(logits[: len(batch_ids)].argmax(dim=-1).detach().cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64)


__all__ = [
    "DeviceGraph",
    "FastSampledBatcher",
    "build_dense_host_plan",
    "sampled_eval_predictions",
]
