"""Build (or verify) the private Kaggle dataset staging folder for the T4 lane (D-065 / T35).

The Kaggle dataset is the immutable input carrier for the canonical Kaggle lane:
raw NF-UNSW-NB15-v3 CSV + frozen C1 protocol artifacts + runtime code snapshot +
machine-readable hash manifest. Kaggle auto-extracts uploaded .zip files, so the
kernel handles both zip and extracted-tree deliveries.

Usage (repo root, project venv):
    # 1) regenerate the code snapshot from the CURRENT committed HEAD
    powershell -NoProfile -ExecutionPolicy Bypass -File src\\kaggle_lane\\remote_rtx5060\\make_code_snapshot.ps1
    # 2) build the dataset staging folder
    .venv\\Scripts\\python.exe src\\kaggle_lane\\build_dataset.py
    # 3) create the private dataset (first time) or upload a new version
    $env:KAGGLE_API_TOKEN = (Get-Content "$env:USERPROFILE\\.kaggle\\kgat.txt" -Raw).Trim()
    .venv\\Scripts\\python.exe -m kaggle datasets create -p src\\kaggle_lane\\outputs\\dataset_stage -t -r skip
    #    or, for a refresh:
    .venv\\Scripts\\python.exe -m kaggle datasets version -p src\\kaggle_lane\\outputs\\dataset_stage -t -r skip -m "refresh: <reason>"

``--check`` verifies an existing staging folder against its ``env_manifest.json``
without rewriting it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
KAGGLE_LANE = Path(__file__).resolve().parent
DEFAULT_OUT = KAGGLE_LANE / "outputs" / "dataset_stage"
DATASET_ID = "mysteriousavailable/xdrift-nf-unsw-c1-inputs"
DATASET_TITLE = "xdrift-nf-unsw-c1-inputs"
CSV_REL = Path("datasets/raw/NF-UNSW-NB15-v3/NF-UNSW-NB15-v3.csv")
FROZEN_ROOT = KAGGLE_LANE / "remote_rtx5060" / "frozen"
CODE_ROOT = KAGGLE_LANE / "remote_rtx5060" / "code"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def git_executable() -> str:
    candidates = ("git", r"C:\Program Files\Git\bin\git.exe")
    for candidate in candidates:
        found = shutil.which(candidate) if "/" not in candidate and "\\" not in candidate else None
        if found:
            return found
        if Path(candidate).is_file():
            return candidate
    raise SystemExit("git not found (needed to pin the snapshot commit)")


def git_short_head() -> str:
    result = subprocess.run([git_executable(), "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE)
    if result.returncode != 0:
        raise SystemExit("git rev-parse failed")
    return result.stdout.strip()


def read_snapshot_info(code_dir: Path) -> dict[str, str]:
    info_path = code_dir / "snapshot_info.txt"
    if not info_path.is_file():
        raise SystemExit(f"missing {info_path}; run src/kaggle_lane/remote_rtx5060/make_code_snapshot.ps1 first")
    values: dict[str, str] = {}
    for line in info_path.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def zip_frozen(stage: Path) -> tuple[Path, list[dict[str, Any]]]:
    frozen_zip = stage / "frozen.zip"
    entries: list[dict[str, Any]] = []
    files = sorted(path for path in FROZEN_ROOT.rglob("*") if path.is_file())
    with zipfile.ZipFile(frozen_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            rel = path.relative_to(FROZEN_ROOT).as_posix()
            archive.write(path, rel)
            entries.append({"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return frozen_zip, entries


def build(out_dir: Path) -> dict[str, Any]:
    head = git_short_head()
    snapshot = read_snapshot_info(CODE_ROOT)
    snapshot_commit = snapshot.get("snapshot_sha", "")
    if snapshot_commit != head:
        raise SystemExit(
            f"code snapshot is for {snapshot_commit!r} but HEAD is {head!r}; "
            "re-run src/kaggle_lane/remote_rtx5060/make_code_snapshot.ps1 so the dataset matches the committed HEAD"
        )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_source = REPO_ROOT / CSV_REL
    if not csv_source.is_file():
        raise SystemExit(f"raw CSV missing: {csv_source}")
    csv_target = out_dir / csv_source.name
    shutil.copy2(csv_source, csv_target)
    csv_sha = sha256_file(csv_target)

    frozen_zip, frozen_entries = zip_frozen(out_dir)
    frozen_sha = sha256_file(frozen_zip)

    code_zip_name = f"xdrift_repo_{head}.zip"
    code_source = CODE_ROOT / code_zip_name
    if not code_source.is_file():
        raise SystemExit(f"code snapshot zip missing: {code_source}")
    code_target = out_dir / code_zip_name
    shutil.copy2(code_source, code_target)
    shutil.copy2(CODE_ROOT / "snapshot_info.txt", out_dir / "snapshot_info.txt")

    env_manifest = {
        "schema_version": 1,
        "dataset_slug": DATASET_ID,
        "git_commit": head,
        "code_zip": {
            "name": code_zip_name,
            "sha256": sha256_file(code_target),
            "bytes": code_target.stat().st_size,
            "snapshot_sha": snapshot_commit,
            "dirty_worktree_files_not_in_snapshot": int(snapshot.get("dirty_worktree_files_not_in_snapshot", "0") or 0),
        },
        "csv": {
            "name": csv_target.name,
            "sha256": csv_sha,
            "bytes": csv_target.stat().st_size,
            "target_path": CSV_REL.as_posix(),
        },
        "frozen_zip": {"name": frozen_zip.name, "sha256": frozen_sha, "bytes": frozen_zip.stat().st_size, "files": frozen_entries},
        "protocol": "docs/05 v1 frozen D-042; grouped mask 010a80d9",
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "env_manifest.json").write_text(json.dumps(env_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "frozen_artifacts.sha256").write_text(
        "\n".join(f"{entry['sha256']}  {entry['path']}" for entry in frozen_entries) + "\n", encoding="utf-8"
    )
    description = (
        "XDRIFT frozen C1 inputs (private) for the Kaggle T4 canonical lane (D-065).\n\n"
        f"Contains:\n- {csv_target.name} (raw, {csv_target.stat().st_size} bytes, sha256 {csv_sha})\n"
        f"- frozen.zip: Protocol v1 (D-042) frozen C1 artifacts sha256 {frozen_sha}\n"
        f"- {code_zip_name}: runtime code snapshot from committed HEAD {head} "
        f"({env_manifest['code_zip']['dirty_worktree_files_not_in_snapshot']} dirty WIP files intentionally excluded)\n"
        "- env_manifest.json + frozen_artifacts.sha256: machine-readable hashes the bring-up kernel verifies.\n\n"
        "Do not mutate; frozen state is the protocol (D-042).\n"
    )
    (out_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": DATASET_TITLE,
                "id": DATASET_ID,
                "licenses": [{"name": "CC0-1.0"}],
                "subtitle": "XDRIFT frozen C1 NetFlow-NB15-v3 inputs + code snapshot (private)",
                "description": description,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return env_manifest


def check(out_dir: Path) -> int:
    manifest_path = out_dir / "env_manifest.json"
    if not manifest_path.is_file():
        print(f"CHECK FAILED: {manifest_path} missing", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    csv = out_dir / manifest["csv"]["name"]
    if not csv.is_file() or sha256_file(csv) != manifest["csv"]["sha256"]:
        problems.append("csv hash/size mismatch")
    code = out_dir / manifest["code_zip"]["name"]
    if not code.is_file() or sha256_file(code) != manifest["code_zip"]["sha256"]:
        problems.append("code zip hash mismatch")
    frozen = out_dir / manifest["frozen_zip"]["name"]
    if not frozen.is_file() or sha256_file(frozen) != manifest["frozen_zip"]["sha256"]:
        problems.append("frozen zip hash mismatch")
    frozen_extracted = out_dir / "frozen"
    zip_members: dict[str, str] | None = None
    if frozen.is_file():
        with zipfile.ZipFile(frozen) as archive:
            zip_members = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist() if not name.endswith("/")}
    for entry in manifest["frozen_zip"]["files"]:
        if zip_members is not None:
            if zip_members.get(entry["path"]) != entry["sha256"]:
                problems.append(f"frozen zip entry mismatch: {entry['path']}")
        else:
            target = frozen_extracted / entry["path"]
            if not target.is_file() or sha256_file(target) != entry["sha256"]:
                problems.append(f"frozen entry mismatch: {entry['path']}")
    if problems:
        print("CHECK FAILED: " + "; ".join(problems[:10]), file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", "dir": str(out_dir), "commit": manifest["git_commit"], "csv_bytes": manifest["csv"]["bytes"]}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    if args.check:
        return check(out_dir)
    env_manifest = build(out_dir)
    print(json.dumps({"out_dir": str(out_dir), "commit": env_manifest["git_commit"], "csv_bytes": env_manifest["csv"]["bytes"], "frozen_files": len(env_manifest["frozen_zip"]["files"]), "code_zip": env_manifest["code_zip"]["name"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
