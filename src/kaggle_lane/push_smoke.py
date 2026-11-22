"""Smoke test for the Kaggle GPU lane (official CLI path).

Pushes a PRIVATE script kernel that reports Python/torch/GPU info, waits for completion, and
downloads its output. Proves token + push + GPU scheduling + output retrieval before any real run.

Usage (repo root, project venv):
    .venv\\Scripts\\python.exe kaggle_lane\\push_smoke.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from run_kernel import OUTPUTS_DIR, push_and_wait

KERNEL_CODE = '''\
import json
import platform
import torch

report = {
    "kind": "xdrift-kaggle-smoke",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}
if torch.cuda.is_available():
    report["device"] = torch.cuda.get_device_name(0)
    report["capability"] = list(torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda")
    report["gpu_matmul_ok"] = bool((x @ x).isfinite().all().item())

print("XDRIFT_SMOKE_JSON=" + json.dumps(report, sort_keys=True))
'''


def main() -> int:
    started = time.strftime("%Y%m%d-%H%M%S")
    slug = f"xdrift-gpu-smoke-{started}"
    script = OUTPUTS_DIR / f"_{slug}_kernel.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(KERNEL_CODE, encoding="utf-8")
    summary = push_and_wait(
        slug=slug,
        code_path=script,
        gpu=True,
        internet=False,
        title=f"xdrift GPU smoke {started}",
        timeout_s=1200,
    )
    return 0 if summary.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
