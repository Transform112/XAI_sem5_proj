#!/usr/bin/env python
"""Verify raw dataset identity against published provenance (stage T3).

Reads each raw dataset directory read-only, streams the CSV(es) in chunks and
produces evidence that the files are the datasets we claim they are:

* exact row count vs the published UQ flow count,
* 55-column schema vs the registry contract (43 NetFlow-v2 + 10 temporal
  features + 2 label columns),
* multiclass label distribution vs published class counts,
* binary/multiclass label consistency and missing labels,
* NaN / infinity cell counts,
* exact duplicate flow count,
* timestamp parseability, range and monotonicity,
* host cardinality,
* archive size + SHA-256 agreement with ``datasets/raw/provenance.json``.

Outputs (small, commit-friendly):
    results/text/data_audit/identity_<dataset>.json
    results/text/data_audit/identity_<dataset>.md
    results/text/data_audit/identity_summary.md
    results/text/data_audit/identity_summary.json
    results/text/data_audit/identity_manifest.json   (reproducibility record)

The raw files are never modified. Published values that disagree with the
measurement are reported with the exact delta and are never forced.

Usage:
    python src/scripts/verify_datasets.py --dataset all --config configs/audit.yaml
    python src/scripts/verify_datasets.py --dataset NF-UNSW-NB15-v3
    python src/scripts/verify_datasets.py --dataset NF-BoT-IoT-v3 --strict

Run it through the lane launcher to get lane validation + a launch manifest:
    powershell -NoProfile -ExecutionPolicy Bypass -File src/scripts/run_lane.ps1 `
        -Lane legacy -Config configs/audit.yaml `
        -Entry src/scripts/verify_datasets.py -- --dataset all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_LAUNCHER_ROOT = Path(__file__).resolve().parents[2]
if str(_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAUNCHER_ROOT))

from src.data.identity import (  # noqa: E402
    summarize_identity_reports,
    verify_dataset,
    write_identity_report,
)
from src.data.registry import (  # noqa: E402
    DatasetSpecError,
    load_dataset_spec,
    resolve_dataset_names,
)
from src.utils.config import ConfigError, load_config  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.manifest import RunManifest, utc_now, write_json  # noqa: E402
from src.utils.paths import PathConfigError, find_project_root, load_paths  # noqa: E402

_STATUS_ORDER = {"fail": 0, "warn": 1, "pass": 2, "missing": 3}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="verify_datasets",
        description="Verify raw NF-v3 dataset identity and provenance (read-only).",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="dataset name, comma-separated names, or 'all' (default: config list / all)",
    )
    parser.add_argument("--config", default="configs/audit.yaml", help="audit config YAML")
    parser.add_argument("--root", default=None, help="project root override")
    parser.add_argument("--lane", default=None, help="path lane (default: XDRIFT_LANE or config default)")
    parser.add_argument("--data-root", default=None, help="override data root (raw lives under <root>/raw)")
    parser.add_argument("--out-dir", default=None, help="output directory (default: results/text/data_audit)")
    parser.add_argument("--strict", action="store_true", help="exit non-zero if any dataset fails a check")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="do not rescan raw data; rebuild identity_summary.{md,json} from existing identity_<dataset>.json files",
    )
    parser.add_argument(
        "--skip-archive-hash",
        action="store_true",
        help="do not recompute archive SHA-256 (provenance archive hashes are trusted)",
    )
    return parser.parse_args(argv)


def _console_table(reports: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "dataset                      rows measured   published  match  classes  nan  inf  dup     hosts  status",
        "---------------------------  -------------  ----------  -----  -------  ---  ---  ------  -----  ------",
    ]
    for report in reports:
        rows = report["rows"]
        lines.append(
            f"{report['dataset']:<27}  {rows['measured']:>13,}  {str(rows['published']):>10}  "
            f"{str(rows['match']):<5}  {len(report['labels']['measured']):>7}  "
            f"{report['missing']['total']:>3}  {report['inf']['total']:>3}  "
            f"{report['duplicates']['exact']:>6,}  {report['hosts']['union_unique']:>5}  "
            f"{report['status']:<6}"
        )
    return "\n".join(lines)


def _summary_payload(reports: Sequence[Mapping[str, Any]], missing: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "datasets": [
            {
                "dataset": report["dataset"],
                "reference_version": report.get("reference_version"),
                "status": report["status"],
                "rows": report["rows"],
                "labels": {
                    "measured": report["labels"]["measured"],
                    "measured_aggregated": report["labels"].get("measured_aggregated"),
                    "taxonomy": report["labels"].get("taxonomy"),
                    "published_classes": report["labels"]["published_classes"],
                    "class_table_delta": report["labels"]["class_table_delta"],
                    "mismatches": report["labels"]["mismatches"],
                },
                "duplicates": report["duplicates"],
                "missing": report["missing"]["total"],
                "inf": report["inf"]["total"],
                "hosts": report["hosts"],
                "timestamps": report["timestamps"],
                "schema": {
                    "n_columns": report["schema"]["n_columns"],
                    "matches_spec": report["schema"]["matches_spec"],
                },
                "checks": report["checks"],
                "files": [
                    {
                        "relative_path": record["relative_path"],
                        "bytes": record["bytes"],
                        "sha256": record["sha256"],
                    }
                    for record in report["files"]
                ],
            }
            for report in reports
        ],
        "missing": list(missing),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve() if args.root else find_project_root()
    paths = load_paths(lane=args.lane, project_root=root)
    if args.data_root:
        data_root = Path(args.data_root).expanduser().resolve()
        raw_root = data_root / "raw"
    else:
        raw_root = paths.raw_root
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else paths.audit_root

    config = load_config(args.config, project_root=root, lane=args.lane)
    log_level = str(config.get("logging", {}).get("level", "INFO")) if isinstance(config.get("logging"), Mapping) else "INFO"
    logger = setup_logging("xdrift.verify", level=log_level)

    requested = args.dataset
    if requested == "all" and isinstance(config.get("datasets"), list):
        requested = list(config["datasets"])
    try:
        names = resolve_dataset_names(requested, project_root=root)
    except DatasetSpecError as exc:
        print(f"[verify] ERROR: {exc}", file=sys.stderr)
        return 2

    provenance_path = raw_root / "provenance.json"
    provenance: dict[str, Any] = {}
    if provenance_path.is_file():
        try:
            # PowerShell 5.1's Out-File utf8 writes a BOM; accept it.
            provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - corrupt file
            logger.warning("could not read provenance.json: %s", exc)

    identity_options = config.get("identity", {}) if isinstance(config.get("identity"), Mapping) else {}
    identity_options = dict(identity_options)
    if args.skip_archive_hash:
        identity_options["hash_archives"] = False

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        # Rebuild the summary from existing per-dataset reports without rescanning
        # the (large) raw CSVs; useful after a reference/taxonomy change.
        reports = []
        for name in names:
            report_path = out_dir / f"identity_{name}.json"
            if not report_path.is_file():
                print(f"[verify] ERROR: missing existing report {report_path}", file=sys.stderr)
                return 2
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        summary_md = out_dir / "identity_summary.md"
        summary_md.write_text(summarize_identity_reports(reports), encoding="utf-8")
        write_json(out_dir / "identity_summary.json", _summary_payload(reports, []))
        print(_console_table(reports))
        print(f"\n[verify] summary rebuilt from existing reports: {summary_md}")
        print(f"[verify] reference_version: {reports[0].get('reference_version')}")
        return 0

    manifest = RunManifest(
        "dataset_identity",
        paths=paths,
        config=config,
        lane=paths.lane,
        seed=config.get("seed") if isinstance(config.get("seed"), int) else None,
        out_path=out_dir / "identity_manifest.json",
        extra={"raw_root": str(raw_root), "datasets_requested": names},
    )
    manifest.start()

    reports: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    try:
        for name in names:
            logger.info("verifying %s", name)
            try:
                spec = load_dataset_spec(name, project_root=root)
            except DatasetSpecError as exc:
                logger.error("%s: %s", name, exc)
                missing.append({"dataset": name, "status": "missing", "error": str(exc)})
                manifest.note(f"{name}: {exc}")
                continue
            raw_dir = raw_root / name
            if not raw_dir.is_dir():
                message = f"raw directory not found: {raw_dir}"
                logger.error("%s: %s", name, message)
                missing.append({"dataset": name, "status": "missing", "error": message})
                manifest.note(f"{name}: {message}")
                continue
            try:
                report = verify_dataset(
                    spec,
                    raw_dir,
                    options=identity_options,
                    provenance=provenance.get(name),
                )
            except FileNotFoundError as exc:
                logger.error("%s: %s", name, exc)
                missing.append({"dataset": name, "status": "missing", "error": str(exc)})
                manifest.note(f"{name}: {exc}")
                continue
            paths_written = write_identity_report(report, out_dir)
            reports.append(report)
            for artifact in paths_written.values():
                manifest.add_artifact(artifact, role="identity_report")
            manifest.note(
                f"{name}: status={report['status']} rows={report['rows']['measured']} "
                f"published={report['rows']['published']}"
            )
            logger.info(
                "%s: status=%s rows=%s published=%s",
                name,
                report["status"],
                f"{report['rows']['measured']:,}",
                report["rows"]["published"],
            )

        summary_md = out_dir / "identity_summary.md"
        summary_md.write_text(summarize_identity_reports(reports), encoding="utf-8")
        summary_json = out_dir / "identity_summary.json"
        write_json(
            summary_json,
            {
                "schema_version": 1,
                "generated_at_utc": utc_now(),
                "datasets": [
                    {
                        "dataset": report["dataset"],
                        "reference_version": report.get("reference_version"),
                        "status": report["status"],
                        "rows": report["rows"],
                        "labels": {
                            "measured": report["labels"]["measured"],
                            "published_classes": report["labels"]["published_classes"],
                            "class_table_delta": report["labels"]["class_table_delta"],
                            "mismatches": report["labels"]["mismatches"],
                        },
                        "duplicates": report["duplicates"],
                        "missing": report["missing"]["total"],
                        "inf": report["inf"]["total"],
                        "hosts": report["hosts"],
                        "timestamps": report["timestamps"],
                        "schema": {
                            "n_columns": report["schema"]["n_columns"],
                            "matches_spec": report["schema"]["matches_spec"],
                        },
                        "checks": report["checks"],
                        "files": [
                            {
                                "relative_path": record["relative_path"],
                                "bytes": record["bytes"],
                                "sha256": record["sha256"],
                            }
                            for record in report["files"]
                        ],
                    }
                    for report in reports
                ],
                "missing": missing,
            },
        )
        for artifact in (summary_md, summary_json):
            manifest.add_artifact(artifact, role="identity_summary")
        manifest.set_dataset(
            "NF-v3-family",
            raw_root,
            patterns=("*.csv",),
            hash_files=False,
        )
        manifest.payload["extra"]["identity"] = {
            report["dataset"]: {
                "status": report["status"],
                "rows_measured": report["rows"]["measured"],
                "rows_published": report["rows"]["published"],
                "files": [
                    {
                        "relative_path": record["relative_path"],
                        "bytes": record["bytes"],
                        "sha256": record["sha256"],
                    }
                    for record in report["files"]
                ],
            }
            for report in reports
        }
        manifest.finish("success")
    except Exception as exc:  # noqa: BLE001 - record and re-raise after manifest
        manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
        manifest.write()
        raise
    manifest.write()

    print(_console_table(reports) if reports else "[verify] no datasets verified")
    if missing:
        print("\n[verify] missing datasets:")
        for item in missing:
            print(f"  - {item['dataset']}: {item['error']}")
    print(f"\n[verify] reports written to {out_dir}")
    print(f"[verify] manifest: {manifest.paths.relative(manifest.resolve_out_path())}")

    statuses = {report["status"] for report in reports}
    if args.strict and ("fail" in statuses or missing):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
