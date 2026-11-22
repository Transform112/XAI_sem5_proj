#!/usr/bin/env bash
# Bootstrap the KAGGLE lane inside a notebook or a Kaggle script.
#
# Verified Kaggle image (D-029, 2026-11-01):
#   Tesla T4 (capability 7.5), Python 3.13.15, torch 2.11.0+cu128 (image-provided)
#   -> platforms/requirements/kaggle.txt pins only PyG (2.8.1), never torch.
#
# Assumptions:
#   - the repository is available at $REPO_DIR (see README.md and ../.. ); on
#     Kaggle it normally lives under /kaggle/working/<repo> and is cloned from a
#     GitHub remote or extracted from an uploaded Kaggle dataset
#   - Kaggle's image already provides the CUDA-enabled torch build
#
# Usage (in a Kaggle cell):
#   !git clone https://github.com/<user>/<repo>.git /kaggle/working/xai_proj
#   !REPO_DIR=/kaggle/working/xai_proj bash /kaggle/working/xai_proj/platforms/setup/setup_kaggle.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
echo "[kaggle-setup] repo: $REPO_DIR"
echo "[kaggle-setup] python: $(python --version 2>&1)"

python -m pip install --quiet --upgrade pip
echo "[kaggle-setup] installing PyG pin (image torch is kept) + common deps"
python -m pip install --quiet -r platforms/requirements/kaggle.txt
python -m pip install --quiet -r platforms/requirements/common.txt

echo "[kaggle-setup] installed versions:"
python -c "import sys, torch, torch_geometric; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda:', torch.cuda.is_available()); print('pyg', torch_geometric.__version__)"

echo "[kaggle-setup] environment check"
python src/scripts/check_env.py --json-out "env_check_kaggle.json" || {
    echo "[kaggle-setup] WARNING: environment check reported failures" >&2
    exit 1
}
echo "[kaggle-setup] done: record env_check_kaggle.json in the run results and use lane=kaggle in manifests"
