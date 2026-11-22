#!/usr/bin/env python
"""Annotate a frozen scorecard with D-039 gate separation (decision_inputs).

The scorecard is a frozen artifact: this script does NOT change any measured
value or check status. It only adds the machine-readable ``decision_inputs``
block that separates G1b data-integrity evidence from the G1c graph-lift
decision input, and regenerates the markdown from the annotated JSON.

Usage:
    python src/scripts/annotate_scorecard_decision_inputs.py
    python src/scripts/annotate_scorecard_decision_inputs.py --scorecard results/text/data_audit/scorecard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.audit import _decision_inputs, _scorecard_markdown  # noqa: E402
from src.utils.manifest import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scorecard", default=None, help="default: results/text/data_audit/scorecard.json")
    args = parser.parse_args(argv)
    path = Path(args.scorecard) if args.scorecard else _ROOT / "results" / "data_audit" / "scorecard.json"
    scorecard = json.loads(path.read_text(encoding="utf-8"))
    scorecard["decision_inputs"] = _decision_inputs(scorecard["checks"])
    write_json(path, scorecard)
    markdown_path = path.with_suffix(".md")
    markdown_path.write_text(_scorecard_markdown(scorecard), encoding="utf-8")
    print(f"[annotate] {path}: decision_inputs added")
    for gate, block in scorecard["decision_inputs"].items():
        if isinstance(block, dict) and "status" in block:
            print(f"  {gate}: {block['status']}")
    print(f"[annotate] markdown regenerated: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
