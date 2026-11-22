"""Aggregate the full Kaggle matrix CL metrics (headline table, T37/D-068)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

SRC = Path("results/text/verification/data_audit/kaggle_matrix_full_metrics.csv")
OUT = Path("results/text/verification/data_audit/kaggle_matrix_full")
STRATS = ["ft", "replay", "ewc"]
MODELS = ["egraphsage", "gcn"]


def main() -> int:
    rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        key = (row["strategy"], row["model"])
        metric = row["metric"]
        block = grouped.setdefault(key, {})
        for field in ("final_average_macro_f1", "forgetting", "bwt"):
            value = row.get(field)
            if value is None or value == "" or value.lower() == "nan":
                continue
            block.setdefault(f"{field}:{metric}", []).append(float(value))
    table: list[dict] = []
    for strategy in STRATS:
        for model in MODELS:
            block = grouped.get((strategy, model), {})
            entry = {"strategy": strategy, "model": model}
            for field in ("final_average_macro_f1", "forgetting", "bwt"):
                for metric in ("all_classes", "supported_set"):
                    values = block.get(f"{field}:{metric}", [])
                    if not values:
                        continue
                    mean = sum(values) / len(values)
                    sd = (sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)) ** 0.5
                    entry[f"{field}_{metric}_mean"] = round(mean, 6)
                    entry[f"{field}_{metric}_sd"] = round(sd, 6)
                    entry[f"{field}_{metric}_n"] = len(values)
            table.append(entry)
    columns = sorted({key for row in table for key in row})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "headline_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    (OUT / "headline_table.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(table, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
