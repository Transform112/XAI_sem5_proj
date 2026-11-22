"""S3 models: E-GraphSAGE (primary) and an edge-aware GCN baseline.

Both models accept ``(x, edge_index, edge_attr, edge_batch=None)`` and return
per-edge class logits for ``edge_batch`` (or all edges when omitted), which is the
calling convention required by the PyG explainers (Doc 2 S3/S7). The layers
subclass ``MessagePassing`` so PyG can inject explainer edge masks inside
``propagate``; explainer mode always bypasses the T27 caches (no stale masks).

T27 performance additions (ported from the user's RTX-5060 custom code, reviewed):
* **E-GraphSAGE static aggregation cache.** ``message()`` returns ``edge_attr``
  only, so the mean aggregation is independent of node features and parameters.
  It is cached keyed by ``(edge_index.data_ptr, edge_attr.data_ptr, n_nodes,
  n_edges, edge_attr._version)`` and reused until the graph tensors change/mutate.
  This makes the legacy full-graph-propagate mini-batch regime cheap (the
  aggregation is computed once per period, not 774x/epoch).
* **Batch-only edge head.** ``edge_batch`` restricts the final linear head to the
  active batch/split edges; logits equal the corresponding slice of the full-edge
  head (tested <=1e-6), and gradients are unchanged because the loss already only
  used those edges.
* **GCN static caches.** The incident-edge mean is cached on the fixed graph; the
  multigraph is coalesced once with ``gcn_norm`` into a weighted simple graph
  (scatter_add over parallel edges is linear, so propagation is mathematically
  identical; tested forward+grad <=1e-6).
Both caches are invalidated when the underlying tensors change (pointer/version),
when the model is moved to another device, when ``edge_attr.requires_grad`` is
True, and whenever PyG Explainer is active (``_explain`` / ``_edge_mask``).
"""

from __future__ import annotations

import weakref

import torch
from torch import nn


def _edge_attr_version(tensor: torch.Tensor) -> int:
    """Version counter; inference tensors do not track one (use -1 -> cache miss)."""
    try:
        return int(tensor._version)
    except RuntimeError:
        return -1
from torch_geometric.nn import GCNConv, MessagePassing


class _CacheMixin:
    """Invalidate static caches on device moves and on explicit request."""

    def clear_cache(self) -> None:  # pragma: no cover - overridden where needed
        return None

    def _apply(self, fn, recurse: bool = True):  # type: ignore[override]
        result = super()._apply(fn, recurse=recurse)
        self.clear_cache()
        return result


class EdgeSAGELayer(_CacheMixin, MessagePassing):
    """One edge-aware GraphSAGE layer: mean-aggregate incoming edge features."""

    def __init__(self, in_node: int, in_edge: int, out: int) -> None:
        super().__init__(aggr="mean")
        self.lin = nn.Linear(in_node + in_edge, out)
        self._cache_key: tuple[int, int, int, int, int] | None = None
        self._cached_agg: torch.Tensor | None = None
        # T34 (bug-report #9): keep weak identities so a recycled data_ptr/storage
        # cannot make a different tensor reuse a stale static aggregation.
        self._cached_edge_index_ref: weakref.ReferenceType | None = None
        self._cached_edge_attr_ref: weakref.ReferenceType | None = None
        # T32 perf: graph_batch training alternates the main graph and the stacked
        # replay graph, so a single-slot cache would thrash and recompute the 1.13M-edge
        # aggregation every step. Keep a small multi-graph cache (identity-guarded).
        self._agg_cache: dict[tuple, tuple[torch.Tensor, weakref.ReferenceType, weakref.ReferenceType]] = {}
        self._agg_cache_max = 4

    def clear_cache(self) -> None:
        self._cache_key = None
        self._cached_agg = None
        self._cached_edge_index_ref = None
        self._cached_edge_attr_ref = None
        self._agg_cache.clear()

    def _explaining(self) -> bool:
        return bool(getattr(self, "_explain", False)) or getattr(self, "_edge_mask", None) is not None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        num_nodes = int(x.size(0))
        if self._explaining() or edge_attr.requires_grad:
            aggregated = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(num_nodes, num_nodes))
        else:
            key = (
                int(edge_index.data_ptr()),
                int(edge_index.storage_offset()),
                int(edge_attr.data_ptr()),
                int(edge_attr.storage_offset()),
                num_nodes,
                int(edge_index.shape[1]),
                _edge_attr_version(edge_attr),
            )
            entry = self._agg_cache.get(key)
            identity_ok = (
                entry is not None
                and entry[1]() is edge_index
                and entry[2]() is edge_attr
            )
            if identity_ok and entry is not None:
                aggregated = entry[0]
            else:
                # Memory-safe exact mean aggregation over destination nodes.
                # PyG's generic mean scatter materialises an E x dim index tensor
                # (>400 MiB at full size -> 3 GB OOM in T27); index_add_ computes the
                # same mathematical mean with O(N) output memory.
                aggregated = edge_attr.new_zeros((num_nodes, edge_attr.shape[1]))
                aggregated.index_add_(0, edge_index[1], edge_attr)
                counts = edge_attr.new_zeros((num_nodes, 1))
                counts.index_add_(0, edge_index[1], edge_attr.new_ones((edge_attr.shape[0], 1)))
                aggregated = aggregated / counts.clamp_min(1.0)
                self._cache_key = key
                self._cached_agg = aggregated
                self._cached_edge_index_ref = weakref.ref(edge_index)
                self._cached_edge_attr_ref = weakref.ref(edge_attr)
                self._agg_cache.pop(key, None)
                self._agg_cache[key] = (
                    aggregated,
                    self._cached_edge_index_ref,
                    self._cached_edge_attr_ref,
                )
                while len(self._agg_cache) > self._agg_cache_max:
                    self._agg_cache.pop(next(iter(self._agg_cache)))
        return torch.relu(self.lin(torch.cat([x, aggregated], dim=-1)))

    def message(self, edge_attr: torch.Tensor) -> torch.Tensor:  # noqa: D102 - PyG hook
        return edge_attr


class EdgeSAGEModel(_CacheMixin, nn.Module):
    """2-layer edge-aware GraphSAGE with an edge-classification head."""

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 10,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers = []
        in_node = node_dim
        for _ in range(num_layers):
            layers.append(EdgeSAGELayer(in_node, edge_dim, hidden_dim))
            in_node = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_dim + edge_dim, num_classes)

    def clear_cache(self) -> None:
        for layer in self.layers:
            layer.clear_cache()

    def node_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        hidden = x
        for layer in self.layers:
            hidden = layer(hidden, edge_index, edge_attr)
            hidden = self.dropout(hidden)
        return hidden

    def edge_logits(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_indices is not None:
            src = edge_index[0, edge_indices]
            dst = edge_index[1, edge_indices]
            edge_repr = torch.cat([hidden[src], hidden[dst], edge_attr[edge_indices]], dim=-1)
            return self.head(edge_repr)
        src, dst = edge_index
        edge_repr = torch.cat([hidden[src], hidden[dst], edge_attr], dim=-1)
        return self.head(edge_repr)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.node_embeddings(x, edge_index, edge_attr)
        return self.edge_logits(hidden, edge_index, edge_attr, edge_batch)


class EdgeGCNModel(_CacheMixin, nn.Module):
    """GCN baseline with the same edge head/loss/settings as E-GraphSAGE.

    Edge features are aggregated (mean over incident edges) into node features
    first, then two GCN layers run on a coalesced weighted simple graph, then the
    identical edge head is applied (batch-restricted when ``edge_batch`` given).
    """

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 10,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.edge_to_node = nn.Linear(edge_dim, hidden_dim)
        self.node_input = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim, normalize=False, add_self_loops=False) for _ in range(max(1, num_layers))])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_dim + edge_dim, num_classes)
        self._cache_key: tuple[int, int, int, int, int] | None = None
        self._cached_agg: torch.Tensor | None = None
        self._adj_key: tuple[int, int, int, int] | None = None
        self._coalesced_edge_index: torch.Tensor | None = None
        self._coalesced_edge_weight: torch.Tensor | None = None
        self._norm_edge_index: torch.Tensor | None = None
        self._norm_edge_weight: torch.Tensor | None = None
        self._cached_edge_index_ref: weakref.ReferenceType | None = None
        self._cached_edge_attr_ref: weakref.ReferenceType | None = None
        self._adj_edge_index_ref: weakref.ReferenceType | None = None
        # T32 perf: main + stacked-replay graphs alternate in a step; keep small
        # multi-graph caches so the 1.13M-edge aggregation/normalisation is not
        # recomputed for the main graph on every replay step.
        self._agg_cache: dict[tuple, tuple[torch.Tensor, weakref.ReferenceType, weakref.ReferenceType]] = {}
        self._adj_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, weakref.ReferenceType]] = {}
        self._multi_cache_max = 4

    def clear_cache(self) -> None:
        self._cache_key = None
        self._cached_agg = None
        self._adj_key = None
        self._cached_edge_index_ref = None
        self._cached_edge_attr_ref = None
        self._adj_edge_index_ref = None
        self._coalesced_edge_index = None
        self._coalesced_edge_weight = None
        self._norm_edge_index = None
        self._norm_edge_weight = None
        self._agg_cache.clear()
        self._adj_cache.clear()
        for conv in self.convs:
            conv._cached_edge_index = None
            conv._cached_adj_t = None

    def _explaining(self) -> bool:
        return any(
            bool(getattr(conv, "_explain", False)) or getattr(conv, "_edge_mask", None) is not None
            for conv in self.convs
        )

    def node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        n_nodes = int(x.shape[0])
        key = (
            int(edge_index.data_ptr()),
            int(edge_index.storage_offset()),
            int(edge_attr.data_ptr()),
            int(edge_attr.storage_offset()),
            n_nodes,
            int(edge_index.shape[1]),
            _edge_attr_version(edge_attr),
        )
        entry = self._agg_cache.get(key)
        identity_ok = (
            entry is not None
            and entry[1]() is edge_index
            and entry[2]() is edge_attr
        )
        if (
            not edge_attr.requires_grad
            and identity_ok
            and entry is not None
        ):
            aggregated = entry[0]
        else:
            aggregated = torch.zeros((n_nodes, edge_attr.shape[1]), dtype=edge_attr.dtype, device=edge_attr.device)
            counts = torch.zeros((n_nodes, 1), dtype=edge_attr.dtype, device=edge_attr.device)
            ones = torch.ones((edge_attr.shape[0], 1), dtype=edge_attr.dtype, device=edge_attr.device)
            aggregated.index_add_(0, src, edge_attr)
            aggregated.index_add_(0, dst, edge_attr)
            counts.index_add_(0, src, ones)
            counts.index_add_(0, dst, ones)
            aggregated = aggregated / counts.clamp_min(1.0)
            if not edge_attr.requires_grad:
                self._cache_key = key
                self._cached_agg = aggregated
                self._cached_edge_index_ref = weakref.ref(edge_index)
                self._cached_edge_attr_ref = weakref.ref(edge_attr)
                self._agg_cache.pop(key, None)
                self._agg_cache[key] = (
                    aggregated,
                    self._cached_edge_index_ref,
                    self._cached_edge_attr_ref,
                )
                while len(self._agg_cache) > self._multi_cache_max:
                    self._agg_cache.pop(next(iter(self._agg_cache)))
        hidden = torch.relu(self.node_input(x) + self.edge_to_node(aggregated))
        # Precompute the exact GCN normalization once per frozen graph: gcn_norm on
        # the multigraph (self-loops included) then coalesce duplicate pairs by
        # summing the normalized weights. scatter-add over duplicates is linear, so
        # propagation on the weighted simple graph is mathematically identical to the
        # multigraph (forward and gradients tested <=1e-6).
        adj_key = (
            int(edge_index.data_ptr()),
            int(edge_index.storage_offset()),
            n_nodes,
            int(edge_index.shape[1]),
        )
        adj_entry = self._adj_cache.get(adj_key)
        adj_identity_ok = (
            adj_entry is not None and adj_entry[4]() is edge_index
        )
        if adj_entry is not None and adj_identity_ok:
            coalesced_ei, coalesced_ew, norm_ei, norm_ew = adj_entry[:4]
            self._adj_key = adj_key
            self._adj_edge_index_ref = adj_entry[4]
            self._norm_edge_index = norm_ei
            self._norm_edge_weight = norm_ew
            self._coalesced_edge_index = coalesced_ei
            self._coalesced_edge_weight = coalesced_ew
        else:
            from torch_geometric.nn.conv.gcn_conv import gcn_norm
            from torch_geometric.utils import coalesce

            norm_ei, norm_ew = gcn_norm(
                edge_index,
                None,
                num_nodes=n_nodes,
                improved=False,
                add_self_loops=True,
                flow="source_to_target",
                dtype=hidden.dtype,
            )
            coalesced_ei, coalesced_ew = coalesce(norm_ei, norm_ew, num_nodes=n_nodes, reduce="sum")
            self._adj_key = adj_key
            self._adj_edge_index_ref = weakref.ref(edge_index)
            self._norm_edge_index = norm_ei
            self._norm_edge_weight = norm_ew
            self._coalesced_edge_index = coalesced_ei
            self._coalesced_edge_weight = coalesced_ew
            self._adj_cache.pop(adj_key, None)
            self._adj_cache[adj_key] = (
                coalesced_ei,
                coalesced_ew,
                norm_ei,
                norm_ew,
                self._adj_edge_index_ref,
            )
            while len(self._adj_cache) > self._multi_cache_max:
                self._adj_cache.pop(next(iter(self._adj_cache)))
        if self._explaining():
            for conv in self.convs:
                conv._cached_edge_index = None
                hidden = torch.relu(conv(hidden, self._norm_edge_index, self._norm_edge_weight))
                hidden = self.dropout(hidden)
        else:
            for conv in self.convs:
                hidden = torch.relu(conv(hidden, self._coalesced_edge_index, self._coalesced_edge_weight))
                hidden = self.dropout(hidden)
        return hidden

    def edge_logits(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_indices is not None:
            src = edge_index[0, edge_indices]
            dst = edge_index[1, edge_indices]
            edge_repr = torch.cat([hidden[src], hidden[dst], edge_attr[edge_indices]], dim=-1)
            return self.head(edge_repr)
        src, dst = edge_index
        edge_repr = torch.cat([hidden[src], hidden[dst], edge_attr], dim=-1)
        return self.head(edge_repr)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.node_embeddings(x, edge_index, edge_attr)
        return self.edge_logits(hidden, edge_index, edge_attr, edge_batch)


MODEL_REGISTRY = {"egraphsage": EdgeSAGEModel, "gcn": EdgeGCNModel}


def build_model(name: str, *, node_dim: int, edge_dim: int, config: dict) -> nn.Module:
    """Instantiate a registered model from a training config mapping."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    model_cfg = dict(config.get("model", config))
    return MODEL_REGISTRY[name](
        node_dim=int(node_dim),
        edge_dim=int(edge_dim),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_classes=int(model_cfg.get("num_classes", 10)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


__all__ = ["MODEL_REGISTRY", "EdgeGCNModel", "EdgeSAGELayer", "EdgeSAGEModel", "build_model"]
