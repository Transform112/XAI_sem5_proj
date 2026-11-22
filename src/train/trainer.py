"""Deterministic edge-classification training loop (S3, D-052 canonical batching).

Canonical batching = **neighbor-sampled edge mini-batches** (D-052, frozen protocol
ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§4.1): each step draws ``batch_size`` (default 1024) train-mask target edges, builds
a bounded 2-hop neighbor-sampled subgraph (fanout <= 10 nodes per hop, bounded
per-step edges => O(batch) cost, no full-graph forward per batch), and applies the
class-weighted cross-entropy at those target edges only. Adam lr 1e-3, max 30
epochs, early stopping on validation macro-F1 (patience 5); the same schedule is
used by FT/replay/EWC/EAR.

Diagnostic paths kept for reference (never the default):

* ``batching="full_batch"`` - one transductive full-graph pass per epoch with a
  chunked edge-head loss (T18 diagnostic; ~1 optimizer step/epoch is why T21-1
  rejected it as canonical);
* ``batching="mini_batch"`` - legacy per-batch full-graph forward; the *loss
  functions* of the two reference paths are numerically equivalent (tested),
  which is a statement about the loss definitions, not about the historical
  per-batch training objective (T21-3 wording fix).

Measured memory is recorded in the timing block (``max_memory_allocated_bytes``,
T21-2), together with ``updates_per_epoch``/``n_updates`` for audit.
"""

from __future__ import annotations

import copy
import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from src.train.fast import DeviceGraph, FastSampledBatcher, build_dense_host_plan, sampled_eval_predictions
from src.train.sampler import EdgeNeighborSampler, SampledSubgraph
from src.utils.logging import get_logger
from src.utils.seed import seed_everything

LOGGER = get_logger("xdrift.trainer")

BATCHING_MODES = ("sampled", "graph_batch", "full_batch", "mini_batch")


@dataclass
class TrainSettings:
    lr: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 30
    patience: int = 5
    class_weights: str = "balanced"
    eval_metric: str = "macro_f1"
    #: "sampled" (canonical, D-052) | "full_batch" (diagnostic) | "mini_batch" (legacy reference)
    batching: str = "sampled"
    #: 2-hop neighbour fanout per hop (nodes), <= 10 per frozen protocol.
    neighbor_fanout: tuple[int, ...] = (10, 10)
    #: Bounded per-step edges for the sampled subgraph (targets are always kept).
    max_edges_per_step: int = 60_000
    #: Edge-head chunk size for the full_batch diagnostic only.
    chunk_edges: int = 262_144
    #: T27: static device cache + zero-relabel global-id batches (canonical).
    optimized_dispatch: bool = True
    #: T27: vectorised dense-host sampler (default ON after T27 validation; the
    #: standard relabelled path remains callable for the reference/equivalence audit).
    sampler_fast_path: bool = True
    #: Optional imbalance experiments (defaults preserve the frozen recipe).
    loss_type: str = "ce"  # "ce" | "focal"
    focal_gamma: float = 2.0
    class_weight_cap: float = 0.0  # 0 = no clipping
    balanced_batches: bool = False  # graph_batch only: sample edges ∝ 1/class_count
    node_noise_std: float = 0.0  # graph-valid augmentation: jitter node features

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TrainSettings":
        train_cfg = dict(config.get("train", config))
        batching = str(train_cfg.get("batching", "sampled"))
        if batching not in BATCHING_MODES:
            raise ValueError(f"unknown batching mode {batching!r}; expected one of {BATCHING_MODES}")
        fanout = train_cfg.get("neighbor_fanout", train_cfg.get("fanout", (10, 10)))
        if isinstance(fanout, (list, tuple)):
            neighbor_fanout = tuple(int(value) for value in fanout)
        else:
            neighbor_fanout = (int(fanout), int(fanout))
        return cls(
            lr=float(train_cfg.get("lr", 1e-3)),
            batch_size=int(train_cfg.get("batch_size", 1024)),
            max_epochs=int(train_cfg.get("max_epochs", 30)),
            patience=int(train_cfg.get("patience", 5)),
            class_weights=str(train_cfg.get("class_weights", "balanced")),
            eval_metric=str(train_cfg.get("eval_metric", "macro_f1")),
            batching=batching,
            neighbor_fanout=neighbor_fanout,
            max_edges_per_step=int(train_cfg.get("max_edges_per_step", 60_000)),
            chunk_edges=int(train_cfg.get("chunk_edges", 262_144)),
            optimized_dispatch=bool(train_cfg.get("optimized_dispatch", True)),
            sampler_fast_path=bool(train_cfg.get("sampler_fast_path", True)),
            loss_type=str(train_cfg.get("loss_type", "ce")),
            focal_gamma=float(train_cfg.get("focal_gamma", 2.0)),
            class_weight_cap=float(train_cfg.get("class_weight_cap", 0.0)),
            balanced_batches=bool(train_cfg.get("balanced_batches", False)),
            node_noise_std=float(train_cfg.get("node_noise_std", 0.0)),
        )


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int] | None = None) -> float:
    from sklearn.metrics import f1_score

    if labels is None:
        labels = np.union1d(np.unique(y_true), np.unique(y_pred))
    if len(labels) == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> dict[str, float]:
    from sklearn.metrics import f1_score

    scores = f1_score(y_true, y_pred, labels=list(labels), average=None, zero_division=0)
    return {str(int(label)): float(score) for label, score in zip(labels, scores)}


def balanced_class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=n_classes).astype("float64")
    weights = np.zeros(n_classes, dtype="float64")
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return torch.as_tensor(weights, dtype=torch.float32)


class FocalCrossEntropy(nn.Module):
    """Class-weighted focal loss for optional imbalance experiments.

    ``F.cross_entropy`` gives the per-edge CE; focal down-weights easy examples
    with ``(1 - p_t) ** gamma``. The default ``gamma=0`` is ordinary CE.
    """

    def __init__(self, weight: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def _mask_indices(data: Any, split: str) -> torch.Tensor:
    mask = getattr(data, f"{split}_mask", None)
    if mask is None:
        raise KeyError(f"graph has no {split}_mask")
    return torch.nonzero(mask, as_tuple=False).flatten()


def _sampled_eval_predictions(
    model: nn.Module,
    data: Any,
    indices: torch.Tensor,
    *,
    sampler: Any,
    device: torch.device,
    batch_size: int = 4096,
    seed: int = 42,
) -> np.ndarray:
    """Fallback evaluation on bounded neighbour-sampled subgraphs (low-memory lanes)."""
    predictions: list[np.ndarray] = []
    rng = np.random.default_rng(int(seed))
    ids = indices.cpu().numpy()
    with torch.no_grad():
        for offset in range(0, len(ids), int(batch_size)):
            batch = ids[offset : offset + int(batch_size)]
            subgraph = sampler.sample(batch, rng).to(device)
            logits = model(subgraph.x, subgraph.edge_index, subgraph.edge_attr)
            predictions.append(logits[subgraph.target_local_index].argmax(dim=-1).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64)


def evaluate(
    model: nn.Module,
    data: Any,
    split: str,
    *,
    device: torch.device,
    labels: Sequence[int] | None = None,
    sampler: Any | None = None,
    eval_batch_size: int = 4096,
) -> dict[str, Any]:
    """Score one split; falls back to bounded sampled evaluation on low-memory GPUs."""
    model.eval()
    indices = _mask_indices(data, split)
    y_true = data.y[indices].cpu().numpy()
    mode = "full_graph"
    try:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        with torch.no_grad():
            logits = model(x, edge_index, edge_attr)
        y_pred = logits[indices].argmax(dim=-1).cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        if sampler is None:
            raise
        torch.cuda.empty_cache()
        mode = "sampled_fallback"
        y_pred = _sampled_eval_predictions(
            model,
            data,
            indices,
            sampler=sampler,
            device=device,
            batch_size=int(eval_batch_size),
        )
    if labels is None:
        labels = np.union1d(np.unique(data.y.cpu().numpy()), np.unique(y_pred))
    return {
        "split": split,
        "n_edges": int(len(y_true)),
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "per_class_f1": per_class_f1(y_true, y_pred, labels),
        "labels": [int(label) for label in labels],
        "evaluation_mode": mode,
    }


def _evaluate_cached(
    model: nn.Module,
    data: Any,
    device_graph: DeviceGraph | None,
    batcher: FastSampledBatcher | None,
    split: str,
    *,
    device: torch.device,
    labels: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Lean eval on the cached device tensors (T27); sampled fallback on 3 GB cards.

    The full-graph forward is attempted only when the free device memory exceeds a
    conservative estimate of its activations, so the legacy lane does not pay a
    failed OOM attempt every epoch. Split target tensors are precomputed once.
    """
    if device_graph is None:
        return evaluate(model, data, split, device=device, labels=labels)
    indices = getattr(device_graph, f"{split}_idx")
    model.eval()
    mode = "full_graph"
    use_sampled = False
    if device.type == "cuda" and batcher is not None:
        free_bytes = int(torch.cuda.mem_get_info()[0])
        hidden = 128
        estimated = int(device_graph.n_edges * 4 * 3 * hidden) + int(
            device_graph.n_edges * int(device_graph.edge_attr.shape[1]) * 4
        )
        use_sampled = free_bytes < estimated
    if use_sampled:
        mode = "sampled_lean"
        y_pred = sampled_eval_predictions(model, device_graph, batcher, indices)
    else:
        try:
            with torch.inference_mode():
                logits = model(device_graph.x, device_graph.edge_index, device_graph.edge_attr)
            y_pred = logits[indices].argmax(dim=-1).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            if batcher is None:
                raise
            torch.cuda.empty_cache()
            mode = "sampled_lean"
            y_pred = sampled_eval_predictions(model, device_graph, batcher, indices)
    y_true = data.y[indices.cpu()].cpu().numpy()
    if labels is None:
        labels = np.union1d(np.unique(data.y.cpu().numpy()), np.unique(y_pred))
    return {
        "split": split,
        "n_edges": int(len(y_true)),
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "per_class_f1": per_class_f1(y_true, y_pred, labels),
        "labels": [int(label) for label in labels],
        "evaluation_mode": mode,
    }


def evaluate_graph_batch(
    model: nn.Module,
    data: Any,
    split: str,
    *,
    device: torch.device,
    cached_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    labels: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Lean T27 full-graph propagation + batch-only-head evaluation.

    Mathematically identical to computing all-edge logits and slicing the split
    (tested); avoids materialising 1.13M head outputs and re-copying tensors.
    """
    model.eval()
    if cached_tensors is not None:
        x, edge_index, edge_attr = cached_tensors
    else:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
    indices = _mask_indices(data, split)
    supports_edge_batch = "edge_batch" in inspect.signature(model.forward).parameters
    with torch.inference_mode():
        if supports_edge_batch:
            indices_device = indices.to(device)
            y_pred = model(x, edge_index, edge_attr, edge_batch=indices_device).argmax(dim=-1).cpu().numpy()
        else:
            logits = model(x, edge_index, edge_attr)
            y_pred = logits[indices].argmax(dim=-1).cpu().numpy()
    y_true = data.y[indices].cpu().numpy()
    if labels is None:
        labels = np.union1d(np.unique(data.y.cpu().numpy()), np.unique(y_pred))
    return {
        "split": split,
        "n_edges": int(len(y_true)),
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "per_class_f1": per_class_f1(y_true, y_pred, labels),
        "labels": [int(label) for label in labels],
        "evaluation_mode": "full_graph_batch_head" if supports_edge_batch else "full_graph",
    }


# ---------------------------------------------------------------------------
# Reference loss helpers (full_batch diagnostic vs mini_batch legacy; T21-3)
# ---------------------------------------------------------------------------

def _stack_replay_entries(replay: Sequence[Any]) -> tuple[Any, Any, Any, Any, Any, int] | None:
    """T32: stack replay subgraphs into one disjoint graph (one model call/step).

    Each buffer entry has its own local node/edge indexing; offsets make the union
    graph disjoint. The replay CE is the mean over all entry targets, so the caller
    scales it by the entry count to preserve the old sum-of-per-entry-mean loss.
    """
    if not replay:
        return None
    import torch

    xs: list[Any] = []
    edge_indices: list[Any] = []
    edge_attrs: list[Any] = []
    targets: list[Any] = []
    labels: list[Any] = []
    node_offset = 0
    edge_offset = 0
    for entry_x, entry_edge_index, entry_edge_attr, target_index, entry_y in replay:
        xs.append(entry_x)
        edge_indices.append(entry_edge_index + int(node_offset))
        edge_attrs.append(entry_edge_attr)
        targets.append(target_index.reshape(-1).long() + int(edge_offset))
        labels.append(entry_y.reshape(-1)[:1].long())
        node_offset += int(entry_x.shape[0])
        edge_offset += int(entry_edge_index.shape[1])
    return (
        torch.cat(xs, dim=0),
        torch.cat(edge_indices, dim=1),
        torch.cat(edge_attrs, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(labels, dim=0),
        len(replay),
    )


def _replay_loss(model: Any, replay_batch: Any, criterion: Any, alpha: float, *, supports_edge_batch: bool) -> Any:
    if replay_batch is None:
        return None
    replay_x, replay_edge_index, replay_edge_attr, replay_targets, replay_y, count = replay_batch
    if supports_edge_batch:
        replay_logits = model(replay_x, replay_edge_index, replay_edge_attr, edge_batch=replay_targets)
    else:
        replay_logits = model(replay_x, replay_edge_index, replay_edge_attr)[replay_targets]
    return float(alpha) * float(count) * criterion(replay_logits, replay_y)


def _edge_replay_loss(
    model: Any,
    hidden: torch.Tensor,
    entries: Sequence[Any],
    data: Any,
    criterion: Any,
    alpha: float,
    device: torch.device,
) -> torch.Tensor | None:
    """D-072 replay loss: old edge features + global host ids, current node embeddings.

    Node embeddings come from the CURRENT period's full-graph forward (passed as
    ``hidden``); only the stored old edge rows and their global host endpoints are
    projected through the batch-only head. CE is a mean over the fresh replay batch,
    matching ``L = CE(new) + alpha * CE(replay)``. No full-edge head is materialised.
    """
    if not entries:
        return None
    global_ids = getattr(data, "node_global_ids", None)
    if global_ids is None:
        raise ValueError("edge replay requires data.node_global_ids for current graph mapping")
    id_to_index = {
        int(global_id): index
        for index, global_id in enumerate(global_ids.detach().cpu().tolist())
    }
    src: list[int] = []
    dst: list[int] = []
    attrs: list[torch.Tensor] = []
    labels: list[int] = []
    for entry in entries:
        source = id_to_index.get(int(entry.src_global))
        target = id_to_index.get(int(entry.dst_global))
        if source is None or target is None:
            # host absent from the current period graph: drop this replay edge
            continue
        src.append(int(source))
        dst.append(int(target))
        attrs.append(torch.as_tensor(entry.edge_attr, dtype=torch.float32))
        labels.append(int(entry.label))
    if not src:
        return None
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_attr = torch.stack(attrs, dim=0).to(device)
    replay_y = torch.tensor(labels, dtype=torch.long, device=device)
    targets = torch.arange(len(src), dtype=torch.long, device=device)
    logits = model.edge_logits(hidden, edge_index, edge_attr, targets)
    return float(alpha) * criterion(logits, replay_y)


def _weighted_ce_sum(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, y, weight=weights, reduction="sum")


def _target_weight_sum(y: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return torch.as_tensor(float(len(y)), dtype=torch.float32, device=y.device)
    return weights.to(y.device)[y].sum()


def _chunked_edge_mean_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor | None,
    chunk_edges: int,
) -> torch.Tensor:
    total = logits.new_zeros(())
    chunk = max(int(chunk_edges), 1)
    for offset in range(0, len(indices), chunk):
        batch = indices[offset : offset + chunk]
        total = total + _weighted_ce_sum(logits[batch], y[batch], weights)
    return total / _target_weight_sum(y[indices], weights)


def full_batch_mean_loss(
    model: nn.Module,
    data: Any,
    train_idx: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    chunk_edges: int = 262_144,
) -> torch.Tensor:
    """Diagnostic: one transductive pass + exact chunked class-weighted edge-mean loss."""
    logits = model(data.x, data.edge_index, data.edge_attr)
    return _chunked_edge_mean_loss(logits, data.y, train_idx, weights, chunk_edges)


def mini_batch_mean_loss(
    model: nn.Module,
    data: Any,
    train_idx: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Legacy reference loss: full-graph forward per edge batch (O(E^2); diagnostic only)."""
    total = 0.0
    weight_total = 0.0
    for offset in range(0, len(train_idx), batch_size):
        batch = train_idx[offset : offset + batch_size]
        logits = model(data.x, data.edge_index, data.edge_attr)
        batch_loss = _weighted_ce_sum(logits[batch], data.y[batch], weights)
        total = total + float(batch_loss)
        weight_total += float(_target_weight_sum(data.y[batch], weights))
    return torch.as_tensor(total / max(weight_total, 1e-12), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        return {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "memory_device": "cuda",
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        }
    try:
        import psutil

        rss = int(psutil.Process().memory_info().rss)
        return {
            "max_memory_allocated_bytes": rss,
            "memory_device": "cpu",
            "peak_rss_note": "current process RSS in bytes (CPU training has no CUDA allocator peak)",
        }
    except Exception:  # noqa: BLE001
        return {"max_memory_allocated_bytes": None, "memory_device": "cpu"}


def train_edge_classifier(
    model: nn.Module,
    data: Any,
    *,
    settings: TrainSettings,
    seed: int = 42,
    device: str | torch.device = "cpu",
    penalty_fn: Callable[[nn.Module], torch.Tensor] | None = None,
    replay_entries: Sequence[Any] = (),
    replay_alpha: float = 1.0,
    replay_provider: Callable[[int], Sequence[Any]] | None = None,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    """Train one model on one period graph and return metrics/timings/memory."""
    seed_everything(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    model.to(device)
    n_classes = int(data.y.max()) + 1
    # A period can be missing a rare class (observed in the CIC sample, where
    # P1 has no Web Attacks).  The model head is created for the full taxonomy,
    # so use its width for class weights; absent classes get zero weight.
    if hasattr(model, "head") and hasattr(model.head, "out_features"):
        n_classes = max(n_classes, int(model.head.out_features))
    batching = str(settings.batching)
    if batching not in BATCHING_MODES:
        raise ValueError(f"unknown batching mode {batching!r}")
    if replay_provider is not None:
        if batching != "graph_batch":
            raise ValueError(
                "D-072 edge-replay provider requires batching='graph_batch' "
                "(full-graph node embeddings); sampled/full/mini modes are unsupported"
            )
        if not (hasattr(model, "node_embeddings") and hasattr(model, "edge_logits")):
            raise ValueError(
                "D-072 edge replay requires model.node_embeddings/edge_logits "
                "(E-GraphSAGE/GCN); use the subgraph replay diagnostic otherwise"
            )
    # T27 static caching: the frozen graph is moved to the device exactly once and
    # never rebuilt per epoch/batch. The same cached tensors serve the reference
    # and optimized paths, so no duplicated device copies are allocated.
    use_optimized = bool(settings.optimized_dispatch) and batching == "sampled"
    device_graph = DeviceGraph(data, device) if use_optimized else None
    train_idx = _mask_indices(data, "train")
    y_train = data.y[train_idx].cpu().numpy()
    weights = balanced_class_weights(y_train, n_classes).to(device)
    if settings.class_weight_cap and settings.class_weight_cap > 0:
        weights = weights.clamp(max=float(settings.class_weight_cap))
    if settings.loss_type == "focal":
        criterion = FocalCrossEntropy(weights, gamma=float(settings.focal_gamma))
    elif settings.loss_type == "ce":
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        raise ValueError(f"unknown loss_type {settings.loss_type!r}; expected 'ce' or 'focal'")
    if settings.balanced_batches and batching != "graph_batch":
        raise ValueError("balanced_batches currently requires batching='graph_batch'")
    optimiser = torch.optim.Adam(model.parameters(), lr=settings.lr)
    if device_graph is not None:
        x, edge_index, edge_attr, y = (
            device_graph.x,
            device_graph.edge_index,
            device_graph.edge_attr,
            device_graph.y,
        )
    else:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        y = data.y.to(device)
    replay = []
    for entry in replay_entries:
        replay.append(
            (
                entry.x.to(device),
                entry.edge_index.to(device),
                entry.edge_attr.to(device),
                entry.target_index.to(device),
                entry.y.to(device),
            )
        )
    replay_batch = _stack_replay_entries(replay)
    sampler = None
    batcher = None
    dense_host = False
    if batching == "sampled":
        sampler = EdgeNeighborSampler(
            data,
            fanout=settings.neighbor_fanout,
            max_edges_per_step=int(settings.max_edges_per_step),
            fast_path=bool(settings.sampler_fast_path),
            device=str(device),
        )
        if use_optimized:
            # Per-period host-coverage decision for the vectorised dense-host fast path.
            dense_host = build_dense_host_plan(sampler) and bool(settings.sampler_fast_path)
            batcher = FastSampledBatcher(
                sampler,
                device_graph,
                max_edges_per_step=int(settings.max_edges_per_step),
                dense_host=dense_host,
                seed=int(seed),
            )
    # T27 staged/static training state (both optimized modes).
    train_idx_dev = train_idx.to(device)
    cached_graph_tensors = (x, edge_index, edge_attr)
    supports_edge_batch = "edge_batch" in inspect.signature(model.forward).parameters
    if hasattr(model, "clear_cache"):
        model.clear_cache()
    epochs = int(epochs_override or settings.max_epochs)
    best_val = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epoch_seconds: list[float] = []
    updates_per_epoch: list[int] = []
    phase_seconds_per_epoch: list[dict[str, float]] = []
    subgraph_edges_per_epoch: list[dict[str, float]] = []
    sample_seconds = 0.0
    forward_backward_seconds = 0.0
    eval_seconds = 0.0
    _reset_peak_memory(device)
    start = time.monotonic()
    replay_step = 0
    for epoch in range(epochs):
        epoch_start = time.monotonic()
        model.train()
        total_loss = 0.0
        steps = 0
        epoch_sample_seconds = 0.0
        epoch_forward_seconds = 0.0
        subgraph_sizes: list[int] = []
        if batching == "graph_batch":
            # T27 B-mode: legacy full-graph propagation semantics (774 updates/epoch,
            # exact CPU permutation RNG) with cached static aggregations and a
            # batch-only head; mathematically identical to the pre-T27 mini-batch
            # loop, gradients still flow through the node linears/head only.
            if settings.balanced_batches:
                # Graph-valid rare-class exposure: sample real training edges with
                # replacement, probability ∝ 1/class_count. No synthetic edges.
                labels_np = y[train_idx.to(device)].detach().cpu().numpy()
                counts = np.bincount(labels_np, minlength=n_classes).astype("float64")
                probs = 1.0 / np.maximum(counts[labels_np], 1.0)
                probs /= probs.sum()
                rng = np.random.default_rng(int(seed) + int(epoch))
                draws = rng.choice(len(train_idx), size=len(train_idx), replace=True, p=probs)
                shuf_batches = train_idx_dev[torch.as_tensor(draws, dtype=torch.long, device=device)]
                n_batches = len(draws)
            else:
                permutation = torch.randperm(len(train_idx))  # exact legacy RNG sequence
                shuf_batches = train_idx_dev[permutation.to(device)]
                n_batches = len(permutation)
            # Graph-valid augmentation: jitter node features only for training
            # forwards; evaluation always uses the untouched graph.
            x_train = x
            if settings.node_noise_std > 0:
                x_train = x + torch.randn_like(x) * float(settings.node_noise_std)
            batch_losses = []
            for offset in range(0, n_batches, int(settings.batch_size)):
                batch = shuf_batches[offset : offset + int(settings.batch_size)]
                forward_start = time.monotonic()
                optimiser.zero_grad(set_to_none=True)
                if replay_provider is not None and supports_edge_batch:
                    # One full-graph node-embedding pass per step; reused for the
                    # new-edge loss and the current batch of old replay edges.
                    hidden = model.node_embeddings(x_train, edge_index, edge_attr)
                    batch_logits = model.edge_logits(hidden, edge_index, edge_attr, batch)
                    loss = criterion(batch_logits, y[batch])
                elif supports_edge_batch:
                    batch_logits = model(x_train, edge_index, edge_attr, edge_batch=batch)
                    loss = criterion(batch_logits, y[batch])
                else:
                    logits = model(x_train, edge_index, edge_attr)
                    loss = criterion(logits[batch], y[batch])
                if replay_provider is not None:
                    entries = replay_provider(replay_step)
                    replay_step += 1
                    replay_loss = _edge_replay_loss(
                        model, hidden, entries, data, criterion, replay_alpha, device
                    )
                else:
                    replay_loss = _replay_loss(
                        model, replay_batch, criterion, replay_alpha, supports_edge_batch=supports_edge_batch
                    )
                if replay_loss is not None:
                    loss = loss + replay_loss
                if penalty_fn is not None:
                    loss = loss + penalty_fn(model)
                loss.backward()
                optimiser.step()
                epoch_forward_seconds += time.monotonic() - forward_start
                batch_losses.append(loss.detach())
                subgraph_sizes.append(int(len(batch)))
                steps += 1
            total_loss = float(torch.stack(batch_losses).mean()) if batch_losses else 0.0
        elif batching == "sampled" and batcher is not None:
            # T27 optimized path: cached device tensors + global-node-id batches,
            # precomputed per-epoch order, GPU index_select; no relabelling/Data objects.
            permutation = batcher.epoch_permutation(int(epoch), len(train_idx))
            batch_losses = []
            for offset in range(0, len(permutation), int(settings.batch_size)):
                batch_positions = permutation[offset : offset + int(settings.batch_size)]
                target_ids = device_graph.train_idx_cpu[batch_positions]
                sample_start = time.monotonic()
                _, edge_index_sub, edge_attr_sub, target_y = batcher.batch(target_ids)
                epoch_sample_seconds += time.monotonic() - sample_start
                subgraph_sizes.append(int(edge_index_sub.shape[1]))
                forward_start = time.monotonic()
                optimiser.zero_grad(set_to_none=True)
                logits = model(device_graph.x, edge_index_sub, edge_attr_sub)
                target_logits = logits[: len(target_ids)]
                loss = criterion(target_logits, target_y)
                replay_loss = _replay_loss(
                    model, replay_batch, criterion, replay_alpha, supports_edge_batch=supports_edge_batch
                )
                if replay_loss is not None:
                    loss = loss + replay_loss
                if penalty_fn is not None:
                    loss = loss + penalty_fn(model)
                loss.backward()
                optimiser.step()
                epoch_forward_seconds += time.monotonic() - forward_start
                batch_losses.append(loss.detach())
                steps += 1
            # Single sync per epoch instead of one per step.
            total_loss = float(torch.stack(batch_losses).mean()) if batch_losses else 0.0
        elif batching == "sampled":
            rng = np.random.default_rng(int(seed) * 1_000_003 + int(epoch))
            permutation = rng.permutation(len(train_idx))
            for offset in range(0, len(permutation), int(settings.batch_size)):
                batch_positions = permutation[offset : offset + int(settings.batch_size)]
                target_ids = train_idx[batch_positions].numpy()
                sample_start = time.monotonic()
                subgraph: SampledSubgraph = sampler.sample(target_ids, rng)
                subgraph = subgraph.to(device)
                epoch_sample_seconds += time.monotonic() - sample_start
                subgraph_sizes.append(int(subgraph.n_sub_edges))
                forward_start = time.monotonic()
                optimiser.zero_grad()
                logits = model(subgraph.x, subgraph.edge_index, subgraph.edge_attr)
                target_logits = logits[subgraph.target_local_index]
                target_y = y[subgraph.target_edge_ids.to(device)]
                loss = criterion(target_logits, target_y)
                replay_loss = _replay_loss(
                    model, replay_batch, criterion, replay_alpha, supports_edge_batch=supports_edge_batch
                )
                if replay_loss is not None:
                    loss = loss + replay_loss
                if penalty_fn is not None:
                    loss = loss + penalty_fn(model)
                loss.backward()
                optimiser.step()
                epoch_forward_seconds += time.monotonic() - forward_start
                total_loss += float(loss.detach())
                steps += 1
        elif batching == "full_batch":
            forward_start = time.monotonic()
            optimiser.zero_grad()
            logits = model(x, edge_index, edge_attr)
            loss = _chunked_edge_mean_loss(logits, y, train_idx.to(device), weights, settings.chunk_edges)
            replay_loss = _replay_loss(
                model, replay_batch, criterion, replay_alpha, supports_edge_batch=supports_edge_batch
            )
            if replay_loss is not None:
                loss = loss + replay_loss
            if penalty_fn is not None:
                loss = loss + penalty_fn(model)
            loss.backward()
            optimiser.step()
            epoch_forward_seconds += time.monotonic() - forward_start
            total_loss = float(loss.detach())
            steps = 1
        else:  # mini_batch legacy reference (full-graph forward per batch)
            permutation = torch.randperm(len(train_idx))
            for offset in range(0, len(permutation), int(settings.batch_size)):
                batch = train_idx[permutation[offset : offset + int(settings.batch_size)]].to(device)
                forward_start = time.monotonic()
                optimiser.zero_grad()
                logits = model(x, edge_index, edge_attr)
                loss = criterion(logits[batch], y[batch])
                replay_loss = _replay_loss(
                    model, replay_batch, criterion, replay_alpha, supports_edge_batch=supports_edge_batch
                )
                if replay_loss is not None:
                    loss = loss + replay_loss
                if penalty_fn is not None:
                    loss = loss + penalty_fn(model)
                loss.backward()
                optimiser.step()
                epoch_forward_seconds += time.monotonic() - forward_start
                total_loss += float(loss.detach())
                steps += 1
        updates_per_epoch.append(int(steps))
        eval_start = time.monotonic()
        if batching == "graph_batch":
            val = evaluate_graph_batch(model, data, "val", device=device, cached_tensors=cached_graph_tensors)
        else:
            val = _evaluate_cached(model, data, device_graph, batcher, "val", device=device)
        epoch_eval_seconds = time.monotonic() - eval_start
        sample_seconds += epoch_sample_seconds
        forward_backward_seconds += epoch_forward_seconds
        eval_seconds += epoch_eval_seconds
        phase_seconds_per_epoch.append(
            {
                "epoch": int(epoch + 1),
                "sample_seconds": round(epoch_sample_seconds, 4),
                "forward_backward_seconds": round(epoch_forward_seconds, 4),
                "eval_seconds": round(epoch_eval_seconds, 4),
            }
        )
        if subgraph_sizes:
            subgraph_edges_per_epoch.append(
                {
                    "epoch": int(epoch + 1),
                    "mean": round(float(np.mean(subgraph_sizes)), 2),
                    "min": int(min(subgraph_sizes)),
                    "max": int(max(subgraph_sizes)),
                }
            )
        epoch_seconds.append(time.monotonic() - epoch_start)
        LOGGER.info(
            "epoch %s/%s loss=%.4f val_macro_f1=%.4f (batching=%s, updates=%s, sample=%.1fs, fwd/bwd=%.1fs, eval=%.1fs)",
            epoch + 1,
            epochs,
            total_loss / max(steps, 1),
            val["macro_f1"],
            batching,
            steps,
            epoch_sample_seconds,
            epoch_forward_seconds,
            epoch_eval_seconds,
        )
        if val["macro_f1"] > best_val:
            best_val = val["macro_f1"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        elif epoch + 1 - best_epoch >= settings.patience:
            LOGGER.info("early stopping at epoch %s (best %s)", epoch + 1, best_epoch)
            break
    train_seconds = time.monotonic() - start
    final_eval_start = time.monotonic()
    memory = _peak_memory(device)
    model.load_state_dict(best_state)
    if batching == "graph_batch":
        val = evaluate_graph_batch(model, data, "val", device=device, cached_tensors=cached_graph_tensors)
        test = evaluate_graph_batch(model, data, "test", device=device, cached_tensors=cached_graph_tensors)
    else:
        val = _evaluate_cached(model, data, device_graph, batcher, "val", device=device)
        test = _evaluate_cached(model, data, device_graph, batcher, "test", device=device)
    final_eval_seconds = time.monotonic() - final_eval_start
    n_params = sum(parameter.numel() for parameter in model.parameters())
    n_updates = int(sum(updates_per_epoch))
    sampled_edge_summary = (
        {
            "subgraph_edges_mean": round(float(np.mean([item["mean"] for item in subgraph_edges_per_epoch])), 2),
            "subgraph_edges_min": int(min(item["min"] for item in subgraph_edges_per_epoch)),
            "subgraph_edges_max": int(max(item["max"] for item in subgraph_edges_per_epoch)),
            "subgraph_edges_per_epoch": subgraph_edges_per_epoch,
        }
        if subgraph_edges_per_epoch
        else {}
    )
    result = {
        "settings": settings.__dict__,
        "seed": seed,
        "n_classes": n_classes,
        "n_train_edges": int(len(train_idx)),
        "epochs_run": len(epoch_seconds),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val,
        "val": val,
        "test": test,
        "timing": {},
        "best_state": {key: value.detach().cpu() for key, value in best_state.items()},
    }
    timing = result["timing"]
    timing.update(
        {
            "train_seconds": round(train_seconds, 3),
            "epoch_seconds_mean": round(float(np.mean(epoch_seconds)), 4),
            "epochs": len(epoch_seconds),
            "n_params": int(n_params),
            "device": str(device),
            # T28-2: report the training mode explicitly; sampler-only flags are
            # attached only for the sampled mode so graph_batch metrics stay clean.
            "training_mode": batching,
            "batching": batching,
            "batch_size": int(settings.batch_size),
            "updates_per_epoch": int(updates_per_epoch[-1]) if updates_per_epoch else 0,
            "updates_per_epoch_all": [int(value) for value in updates_per_epoch],
            "n_updates": n_updates,
            "sample_seconds": round(sample_seconds, 3),
            "forward_backward_seconds": round(forward_backward_seconds, 3),
            "eval_seconds": round(eval_seconds, 3),
            "final_eval_seconds": round(final_eval_seconds, 3),
            "phase_seconds_per_epoch": phase_seconds_per_epoch,
            "eval_mode_final": val.get("evaluation_mode"),
        }
    )
    if batching == "graph_batch":
        timing.update(
            {
                "graph_batch_optimized": True,
                "edge_batch_head": bool(supports_edge_batch),
                "static_aggregation_cache": hasattr(model, "clear_cache"),
                "training_mode_semantics": "legacy_transductive_full_graph_propagation",
            }
        )
    elif batching == "sampled":
        timing.update(
            {
                "neighbor_fanout": list(settings.neighbor_fanout),
                "max_edges_per_step": int(settings.max_edges_per_step),
                "sampler_fast_path": bool(settings.sampler_fast_path),
                "optimized_dispatch": bool(use_optimized),
                "sampler_impl": "optimized_global_ids" if batcher is not None else "reference_relabelled",
                "dense_host_fast_sampler": bool(dense_host),
                "device_graph_cached": device_graph is not None,
            }
        )
    else:  # full_batch / mini_batch reference/diagnostic modes
        timing.update(
            {
                "chunk_edges": int(settings.chunk_edges) if batching == "full_batch" else None,
                "reference_path": True,
            }
        )
    timing.update(sampled_edge_summary)
    timing.update(memory)
    return result


__all__ = [
    "BATCHING_MODES",
    "TrainSettings",
    "balanced_class_weights",
    "evaluate",
    "full_batch_mean_loss",
    "macro_f1",
    "mini_batch_mean_loss",
    "per_class_f1",
    "train_edge_classifier",
]
