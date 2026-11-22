#!/usr/bin/env bash
# Canonical POSIX entry point for every stage (Linux, macOS, Kaggle, WSL).
#
# Bootstraps only: finds the repository root, a Python interpreter and git, then
# delegates to the OS-neutral launcher src/scripts/run_lane.py, which validates the
# lane/config, exports XDRIFT_* variables and writes the run manifest.
#
# Usage:
#   bash src/scripts/run_lane.sh --lane cpu --config configs/audit.yaml \
#       --entry src/scripts/run_data_audit.py -- --dataset NF-UNSW-NB15-v3
#
#   bash src/scripts/run_lane.sh --lane kaggle --entry src/scripts/run_data_audit.py --dry-run
#   bash src/scripts/run_lane.sh --list-lanes
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

LANE=""
CONFIG="configs/base.yaml"
ENTRY=""
RUN_NAME=""
PY="${XDRIFT_PYTHON:-}"
DRY_RUN=""
LIST_LANES=""
SKIP_CUDA=""
EXTRA=()
SET_OVERRIDES=()
ENV_OVERRIDES=()

usage() {
    sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lane)        LANE="$2"; shift 2 ;;
        --lane=*)      LANE="${1#*=}"; shift ;;
        --config)      CONFIG="$2"; shift 2 ;;
        --config=*)    CONFIG="${1#*=}"; shift ;;
        --entry)       ENTRY="$2"; shift 2 ;;
        --entry=*)     ENTRY="${1#*=}"; shift ;;
        --run-name)    RUN_NAME="$2"; shift 2 ;;
        --run-name=*)  RUN_NAME="${1#*=}"; shift ;;
        --python)      PY="$2"; shift 2 ;;
        --python=*)    PY="${1#*=}"; shift ;;
        --set)         SET_OVERRIDES+=("$2"); shift 2 ;;
        --env)         ENV_OVERRIDES+=("$2"); shift 2 ;;
        --dry-run)     DRY_RUN="--dry-run"; shift ;;
        --json)        EXTRA+=("--json"); shift ;;
        --skip-cuda-check) SKIP_CUDA="--skip-cuda-check"; shift ;;
        --list-lanes)  LIST_LANES="--list-lanes"; shift ;;
        --)            shift; EXTRA=("$@"); break ;;
        -h|--help)     usage 0 ;;
        *)             echo "[run_lane.sh] unknown argument: $1" >&2; usage 2 ;;
    esac
done

# --- git for the manifest commit SHA (no absolute paths in src/ code). --------
if [[ -z "${XDRIFT_GIT:-}" ]] && command -v git >/dev/null 2>&1; then
    export XDRIFT_GIT="$(command -v git)"
fi

# --- Python: XDRIFT_PYTHON > project venv > python3 > python. -----------------
if [[ -z "$PY" ]]; then
    if [[ -x "$ROOT/.venv/bin/python" ]]; then
        PY="$ROOT/.venv/bin/python"
    elif [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then
        PY="$ROOT/.venv/Scripts/python.exe"
    elif command -v python3 >/dev/null 2>&1; then
        PY="python3"
    else
        PY="python"
    fi
fi

LAUNCHER="$ROOT/src/scripts/run_lane.py"
[[ -f "$LAUNCHER" ]] || { echo "[run_lane.sh] launcher not found: $LAUNCHER" >&2; exit 2; }

ARGS=("$LAUNCHER" "--root" "$ROOT")
[[ -n "$LANE" ]] && ARGS+=("--lane" "$LANE")
[[ -n "$CONFIG" ]] && ARGS+=("--config" "$CONFIG")
[[ -n "$ENTRY" ]] && ARGS+=("--entry" "$ENTRY")
[[ -n "$RUN_NAME" ]] && ARGS+=("--run-name" "$RUN_NAME")
[[ -n "$PY" && "$PY" != "python" && "$PY" != "python3" ]] && ARGS+=("--python" "$PY")
[[ -n "$LIST_LANES" ]] && ARGS+=("--list-lanes")
[[ -n "$DRY_RUN" ]] && ARGS+=("--dry-run")
[[ -n "$SKIP_CUDA" ]] && ARGS+=("--skip-cuda-check")
for item in "${SET_OVERRIDES[@]:-}"; do [[ -n "$item" ]] && ARGS+=("--set" "$item"); done
for item in "${ENV_OVERRIDES[@]:-}"; do [[ -n "$item" ]] && ARGS+=("--env" "$item"); done
if [[ ${#EXTRA[@]} -gt 0 ]]; then
    ARGS+=("--")
    ARGS+=("${EXTRA[@]}")
fi

exec "$PY" "${ARGS[@]}"
