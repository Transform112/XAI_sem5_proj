# %% [markdown]
# # 00 - Environment setup (v2, reviewed)
#
# Checks the runtime before any data work:
#
# - where the project and datasets live;
# - CSV headers (BOM-safe) validated against the feature contract;
# - SHA-256 of each raw file, verified against a data manifest when present;
# - Python packages (required vs optional for the v2 plan), PyTorch / PyG
#   sampler backend and GPU;
# - provenance (UTC time, git commit) written to the setup report.
#
# It only inspects files; the data audit notebook reads the full CSVs.

# %%
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

print("Notebook: 00 - environment setup (v2)")
print("Started from:", Path.cwd())

# %%
def find_project_root(start: Path | None = None) -> Path | None:
    env = os.environ.get("XDRIFT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = start or Path.cwd()
    for candidate in [here, *here.parents]:
        if (candidate / "datasets").is_dir() and (candidate / "new_pipeline").is_dir():
            return candidate
    return None


ON_KAGGLE = Path("/kaggle/input").exists()
_root = find_project_root()
if _root is None and not ON_KAGGLE:
    raise RuntimeError(
        "Could not locate the repository (a directory containing both datasets/ and "
        "new_pipeline/). Set XDRIFT_ROOT instead of letting outputs land in the cwd."
    )
PROJECT_ROOT = _root or Path.cwd().resolve()
OUTPUT_ROOT = (
    Path("/kaggle/working/xdrift_pipeline")
    if ON_KAGGLE
    else PROJECT_ROOT / "new_pipeline" / "results"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print("Project root :", PROJECT_ROOT)
print("Kaggle input :", ON_KAGGLE)
print("Output root  :", OUTPUT_ROOT)
print("Python       :", sys.version.replace("\n", " "))

# %%
DATASETS = {
    "NF-UNSW-NB15-v3": "NF-UNSW-NB15-v3.csv",
    "NF-CSE-CIC-IDS2018-v3": "NF-CICIDS2018-v3.csv",
}
EXPECTED_COLUMNS = [
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR",
    "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS",
    "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE", "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV", "Label", "Attack",
]
LABEL_COLUMNS = ["Label", "Attack"]
DROP_COLUMNS = [
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR", "L4_SRC_PORT", *LABEL_COLUMNS,
]
SOURCE_FEATURES = [c for c in EXPECTED_COLUMNS if c not in DROP_COLUMNS]
MISSING_INDICATORS = [
    "SRC_TO_DST_SECOND_BYTES_missing",
    "DST_TO_SRC_SECOND_BYTES_missing",
]
CANONICAL_MODEL_INPUTS = [*SOURCE_FEATURES, *MISSING_INDICATORS]

# Columns that are CODES or IDENTIFIERS, not magnitudes. Standardising them as
# continuous numbers is meaningless; stage 02/04 must bucket / one-hot / drop
# them (plan D7.4). DNS_QUERY_ID is a per-transaction random ID: treat it as a
# shortcut/noise candidate and audit its per-class distribution.
CODE_LIKE_COLUMNS = [
    "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE", "TCP_FLAGS", "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
]
IDENTIFIER_LIKE_COLUMNS = ["DNS_QUERY_ID"]

# %%
# Static self-consistency of the constants (cheap). The real check against the
# files' headers happens per dataset below and is enforced there.
assert len(EXPECTED_COLUMNS) == 55 and len(set(EXPECTED_COLUMNS)) == 55, "duplicate/missing expected columns"
assert len(SOURCE_FEATURES) == 48, f"expected 48 source features, got {len(SOURCE_FEATURES)}"
assert len(CANONICAL_MODEL_INPUTS) == 50
print("FEATURE CONTRACT (constants)")
print(f"  physical {len(EXPECTED_COLUMNS)} | labels {len(LABEL_COLUMNS)} | "
      f"dropped metadata {len(DROP_COLUMNS) - len(LABEL_COLUMNS)} | "
      f"source features {len(SOURCE_FEATURES)} | + indicators {len(MISSING_INDICATORS)} "
      f"= {len(CANONICAL_MODEL_INPUTS)} canonical inputs")
print("  [NOTE] canonical inputs still contain code-like columns:", CODE_LIKE_COLUMNS)
print("  [NOTE] identifier-like column:", IDENTIFIER_LIKE_COLUMNS)
print("  [NOTE] indicators are justified only if the audit shows >0.1% missing/sentinel "
      "(see missing_indicator_evidence in the audit JSON).")

# %%
def find_dataset_path(name: str) -> Path | None:
    file_name = DATASETS[name]
    roots = [Path("/kaggle/input/nf-v3-combined-unsw-cicids")]
    if Path("/kaggle/input").is_dir():
        roots.extend(Path("/kaggle/input").glob("*"))
    for root in roots:
        candidate = root / file_name
        if candidate.is_file():
            return candidate
    local = PROJECT_ROOT / "datasets" / "raw" / name / file_name
    return local if local.is_file() else None


DATASET_FILES = {name: find_dataset_path(name) for name in DATASETS}
for name, path in DATASET_FILES.items():
    print(f"dataset {name:26s} -> {path}")

# %%
HASH_FILES = os.environ.get("XDRIFT_SETUP_HASH", "1") != "0"
MANIFEST_PATH = (OUTPUT_ROOT if ON_KAGGLE else PROJECT_ROOT / "new_pipeline" / "configs") / "data_manifest.json"
WRITE_MANIFEST = os.environ.get("XDRIFT_WRITE_MANIFEST", "0") == "1"
ALLOW_SCHEMA_DRIFT = os.environ.get("XDRIFT_ALLOW_SCHEMA_DRIFT", "0") == "1"


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


manifest: dict = {}
if MANIFEST_PATH.is_file():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    print("data manifest:", MANIFEST_PATH)
else:
    print("data manifest: none yet (run once with XDRIFT_WRITE_MANIFEST=1 to pin hashes)")

# %%
dataset_checks: list[dict[str, object]] = []
failures: list[str] = []
for dataset, path in DATASET_FILES.items():
    print("\n" + "=" * 78)
    print("DATASET:", dataset)
    print("=" * 78)
    if path is None:
        print("  [FAIL] file not found")
        dataset_checks.append({"dataset": dataset, "status": "missing", "path": None})
        failures.append(f"{dataset}: missing")
        continue
    size = path.stat().st_size
    print(f"  path   : {path}")
    print(f"  size   : {size:,} bytes ({size / 1024**2:.1f} MiB)")
    # utf-8-sig strips a BOM; without it header[0] would not match.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        first_row = handle.readline().rstrip("\r\n").split(",")
    matches = header == EXPECTED_COLUMNS
    print(f"  columns: {len(header)} (expected {len(EXPECTED_COLUMNS)}); header order matches: {matches}")
    if not matches:
        print("  missing:", [c for c in EXPECTED_COLUMNS if c not in header])
        print("  extra  :", [c for c in header if c not in EXPECTED_COLUMNS])
    actual_features = [c for c in header if c not in DROP_COLUMNS]
    features_ok = actual_features == SOURCE_FEATURES
    print(f"  source features from the real header == contract: {features_ok}")
    row_ok = len(first_row) == len(header)
    sha256, sha_state = None, "skipped"
    if HASH_FILES:
        print("  hashing file (streaming)...")
        sha256 = sha256_file(path)
        print("  sha256 :", sha256)
        pinned = manifest.get(dataset, {}).get("sha256")
        if pinned:
            sha_state = "match" if pinned == sha256 else "MISMATCH"
        else:
            sha_state = "unpinned"
        print("  manifest check:", sha_state)
    status = "ok" if (matches and features_ok and row_ok and sha_state != "MISMATCH") else "fail"
    if status != "ok":
        failures.append(f"{dataset}: header/feature/hash contract violated")
    dataset_checks.append({
        "dataset": dataset, "status": status, "path": str(path), "size_bytes": size,
        "sha256": sha256, "manifest_check": sha_state, "columns": len(header),
        "header_matches": matches, "source_features_match": features_ok,
    })

if WRITE_MANIFEST:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_manifest = {c["dataset"]: {"sha256": c["sha256"], "size_bytes": c["size_bytes"]}
                    for c in dataset_checks if c.get("sha256")}
    MANIFEST_PATH.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
    print("wrote manifest:", MANIFEST_PATH)

# %%
# Import name -> distribution name. importlib.metadata.version() needs the
# DISTRIBUTION name ("scikit-learn", "PyYAML"), not the import name.
REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "yaml", "torch", "torch_geometric", "pyarrow"]
OPTIONAL = {
    "matplotlib": "figures", "networkx": "graph diagnostics",
    "xgboost": "05b tabular ceiling", "lightgbm": "05b tabular ceiling",
    "optuna": "06a tuning", "captum": "07 Integrated Gradients (optional)",
    "pytest": "gate contract tests",
}
SAMPLER_BACKENDS = ["pyg_lib", "torch_sparse"]   # at least one for neighbour sampling
DIST_FALLBACK = {"sklearn": "scikit-learn", "yaml": "PyYAML", "torch_geometric": "torch-geometric",
                 "pyg_lib": "pyg-lib", "torch_sparse": "torch-sparse", "torch_scatter": "torch-scatter"}
try:
    PKG_TO_DIST = importlib.metadata.packages_distributions()
except Exception:
    PKG_TO_DIST = {}


def package_version(import_name: str) -> str:
    candidates = [*PKG_TO_DIST.get(import_name, []), DIST_FALLBACK.get(import_name, import_name)]
    for dist in candidates:
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


package_checks: list[dict[str, object]] = []
print("PACKAGE CHECKS")
for package in [*REQUIRED, *OPTIONAL, *SAMPLER_BACKENDS]:
    kind = "required" if package in REQUIRED else ("sampler" if package in SAMPLER_BACKENDS else "optional")
    installed = importlib.util.find_spec(package) is not None
    version = package_version(package) if installed else None
    flag = " OK " if installed else ("MISS" if kind == "required" else "skip")
    print(f"  [{flag}] {package:16s} {kind:9s} {version or ''}")
    package_checks.append({"package": package, "kind": kind, "installed": installed, "version": version})
if not any(p["installed"] for p in package_checks if p["package"] in SAMPLER_BACKENDS):
    print("  [WARN] neither pyg_lib nor torch_sparse found: PyG neighbour sampling "
          "(needed for CIC and the matched-fanout plan) will be unavailable.")
print("Python executable:", sys.executable)

# %%
gpu: dict[str, object] = {"available": False}
nvidia_smi = shutil.which("nvidia-smi")
if nvidia_smi:
    result = subprocess.run(
        [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    print(result.stdout.strip() or result.stderr.strip() or "no GPU reported")
    if result.stdout.strip():
        gpu = {"available": True, "source": "nvidia-smi", "detail": result.stdout.strip()}
if importlib.util.find_spec("torch") is not None:
    try:
        import torch
        print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available(),
              "| devices:", torch.cuda.device_count())
        if torch.cuda.is_available():
            gpu = {"available": True, "source": "torch", "device": torch.cuda.get_device_name(0),
                   "device_count": torch.cuda.device_count(), "cuda": torch.version.cuda}
    except Exception as exc:                      # broken CUDA libs must not abort setup
        print("[WARN] torch import failed:", repr(exc))
        gpu["torch_error"] = repr(exc)
print("GPU summary:", gpu)

# %%
OUTPUT_DIRS = {
    "data_audit": OUTPUT_ROOT / "data_audit",
    "env_checks": OUTPUT_ROOT / "env_checks",
    "figures": OUTPUT_ROOT / "figures",
}
for name, path in OUTPUT_DIRS.items():
    path.mkdir(parents=True, exist_ok=True)
    print(f"output {name:12s}: {path} (writable={os.access(path, os.W_OK)})")


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=False)
        return out.stdout.strip() or None
    except Exception:
        return None

# %%
report = {
    "notebook": "00_setup", "version": "2.0",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": git_commit(),
    "project_root": str(PROJECT_ROOT), "kaggle": ON_KAGGLE,
    "python": sys.version, "platform": platform.platform(),
    "feature_contract": {
        "physical_columns": len(EXPECTED_COLUMNS), "label_columns": LABEL_COLUMNS,
        "source_features": SOURCE_FEATURES, "source_feature_count": len(SOURCE_FEATURES),
        "missing_indicators": MISSING_INDICATORS,
        "canonical_model_inputs": CANONICAL_MODEL_INPUTS,
        "canonical_input_count": len(CANONICAL_MODEL_INPUTS),
        "code_like_columns": CODE_LIKE_COLUMNS,
        "identifier_like_columns": IDENTIFIER_LIKE_COLUMNS,
    },
    "datasets": dataset_checks, "packages": package_checks, "gpu": gpu,
    "failures": failures,
    "output_root": str(OUTPUT_ROOT),
    "output_dirs": {k: str(v) for k, v in OUTPUT_DIRS.items()},
}
report_path = OUTPUT_DIRS["env_checks"] / "setup_report.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("report:", report_path)

# %%
print("=" * 78)
print("SETUP SUMMARY")
print("=" * 78)
for item in dataset_checks:
    print(f"  [{str(item['status']).upper():5s}] {item['dataset']:26s} {item.get('path')}")
    if item.get("sha256"):
        print(f"          sha256 {item['sha256']}  manifest={item['manifest_check']}")
missing = [p["package"] for p in package_checks if p["kind"] == "required" and not p["installed"]]
print("  missing REQUIRED packages:", missing or "none")
print("  gpu available            :", bool(gpu.get("available")))
print("  report                   :", report_path)
if failures and not ALLOW_SCHEMA_DRIFT:
    raise RuntimeError("Setup contract failed: " + "; ".join(failures))
print("\nNext: 01_data_audit. It reads the full CSVs and reports rows vs physical lines, "
      "exact / feature duplicates, label conflicts, hosts, timestamps and per-column stats.")