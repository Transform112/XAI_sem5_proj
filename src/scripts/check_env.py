#!/usr/bin/env python
"""Environment check for the explanation-drift project.

Prints a JSON report and exits non-zero when a required check fails.
Required checks: torch import, core libraries, toy MessagePassing forward,
edge-feature masking through propagate, PyG explainer API, CPU determinism.
CUDA is required only when --require-cuda is passed (canonical/legacy GPU lanes).

Usage:
    python src/scripts/check_env.py [--require-cuda] [--json-out PATH]
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import traceback

REPORT: dict = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "checks": {},
    "ok": True,
}


def _record(name: str, fn, required: bool = True) -> bool:
    try:
        detail = fn()
        REPORT["checks"][name] = {"ok": True, "detail": detail}
        return True
    except Exception as exc:  # noqa: BLE001 - report, do not crash
        REPORT["checks"][name] = {
            "ok": False,
            "detail": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        }
        if required:
            REPORT["ok"] = False
        return False


def check_torch(require_cuda: bool) -> str:
    import torch

    info = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        info["device_count"] = torch.cuda.device_count()
        info["cudnn"] = torch.backends.cudnn.version()
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        _ = (a @ b).sum().item()
        info["gpu_matmul"] = "ok"
    elif require_cuda:
        raise RuntimeError("CUDA unavailable but --require-cuda was passed")
    return json.dumps(info)


def check_core_libs() -> str:
    import importlib

    versions = {}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "yaml", "networkx"):
        versions[mod] = importlib.import_module(mod).__version__
    return json.dumps(versions)


def check_torch_geometric() -> str:
    import torch_geometric

    return f"torch_geometric {torch_geometric.__version__}"


def check_toy_message_passing() -> str:
    """Toy E-GraphSAGE-style layer + edge head that must run and mask through propagate."""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.nn import MessagePassing

    class EdgeSAGELayer(MessagePassing):
        def __init__(self, in_node: int, in_edge: int, out: int):
            super().__init__(aggr="mean")
            self.lin = torch.nn.Linear(in_node + in_edge, out)

        def forward(self, x, edge_index, edge_attr):
            agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            return torch.relu(self.lin(torch.cat([x, agg], dim=-1)))

        def message(self, edge_attr):
            return edge_attr

    class EdgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = EdgeSAGELayer(8, 5, 16)
            self.l2 = EdgeSAGELayer(16, 5, 16)
            self.head = torch.nn.Linear(32 + 5, 3)

        def forward(self, x, edge_index, edge_attr):
            h = self.l1(x, edge_index, edge_attr)
            h = self.l2(h, edge_index, edge_attr)
            src, dst = edge_index
            z = torch.cat([h[src], h[dst], edge_attr], dim=-1)
            return self.head(z)

    torch.manual_seed(0)
    num_nodes, num_edges = 20, 60
    x = torch.ones(num_nodes, 8)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 5)
    model = EdgeModel()
    out = model(x, edge_index, edge_attr)
    assert out.shape == (num_edges, 3), out.shape

    # Mask-through-propagate check: zeroing edge features must change the output.
    masked_out = model(x, edge_index, torch.zeros_like(edge_attr))
    if torch.allclose(out, masked_out):
        raise RuntimeError("edge features did not affect output: mask is bypassing propagate")
    return json.dumps({"edge_logits_shape": list(out.shape), "mask_effect": "ok"})


def check_pyg_explainer() -> str:
    """Verify a PyG explainer can explain a propagate-based GNN.

    Tries edge-level explanation first (the project's task level); falls back to
    node-level when the installed PyG version rejects task_level='edge'.
    """
    import torch
    from torch_geometric.data import Data
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.nn import GCNConv

    class ToyGCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = GCNConv(8, 16)
            self.c2 = GCNConv(16, 3)

        def forward(self, x, edge_index):
            x = torch.relu(self.c1(x, edge_index))
            x = self.c2(x, edge_index)
            return torch.log_softmax(x, dim=-1)

    torch.manual_seed(0)
    x = torch.randn(30, 8)
    edge_index = torch.randint(0, 30, (2, 90))
    y = torch.randint(0, 3, (30,))
    data = Data(x=x, edge_index=edge_index, y=y)

    errors = []
    for task_level in ("edge", "node"):
        try:
            explainer = Explainer(
                model=ToyGCN(),
                algorithm=GNNExplainer(epochs=20),
                explanation_type="model",
                node_mask_type="attributes",
                edge_mask_type="object",
                model_config=dict(
                    mode="multiclass_classification",
                    task_level=task_level,
                    return_type="log_probs",
                ),
            )
            explanation = explainer(data.x, data.edge_index)
            edge_mask = getattr(explanation, "edge_mask", None)
            if edge_mask is None:
                raise RuntimeError("explanation returned no edge_mask")
            return json.dumps(
                {
                    "task_level_used": task_level,
                    "edge_mask_shape": list(edge_mask.shape),
                    "pyg_explainer": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{task_level}: {type(exc).__name__}: {exc}")
    raise RuntimeError("no explainer configuration worked; " + " | ".join(errors))


def check_determinism() -> str:
    """Two identical CPU training runs with the same seed must match exactly."""
    import torch

    def run() -> float:
        random.seed(123)
        torch.manual_seed(123)
        model = torch.nn.Sequential(torch.nn.Linear(16, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        x = torch.randn(64, 16)
        y = torch.randn(64, 1)
        for _ in range(3):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            opt.step()
        return float(loss.detach())

    first, second = run(), run()
    if abs(first - second) > 0.0:
        raise RuntimeError(f"non-deterministic CPU training: {first} != {second}")
    return f"identical losses: {first:.8f}"


def check_pyyaml_config() -> str:
    import tempfile

    import yaml

    payload = {"data_root": "env:XDRIFT_DATA", "seed": 1}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(payload, handle)
        path = handle.name
    loaded = yaml.safe_load(open(path, encoding="utf-8"))
    if loaded["seed"] != 1:
        raise RuntimeError("yaml round-trip failed")
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    _record("torch", lambda: check_torch(args.require_cuda))
    _record("core_libs", check_core_libs)
    _record("torch_geometric", check_torch_geometric)
    _record("toy_message_passing", check_toy_message_passing)
    _record("pyg_explainer", check_pyg_explainer)
    _record("determinism", check_determinism)
    _record("config_yaml", check_pyyaml_config)

    text = json.dumps(REPORT, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"\nreport written to {args.json_out}", file=sys.stderr)
    return 0 if REPORT["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
