"""Assemble the single-file paper draft from the signed-section Markdown files."""

from __future__ import annotations

from pathlib import Path

OUT = Path("paper_writing/manuscript/XDRIFT_paper_draft.md")
SECTIONS = [
    ("00_abstract.md", ""),
    ("01_introduction.md", ""),
    ("02_related_work.md", ""),
    ("03_method.md", ""),
    ("04_results_s5.md", ""),
    ("05_results_s8.md", ""),
    ("06_results_rq2_rq3.md", ""),
    ("07_discussion.md", ""),
    ("08_conclusion.md", ""),
    ("09_figures_tables_plan.md", ""),
]
HEADER = """# XDRIFT — Calibrated Explanation Drift in Continual GNN Models

**Draft assembled 2026-11-03 from the signed/frozen section files.**

**Status.** All numeric results in this draft are verifier-signed or verifier-audited
(see `docs/VERIFICATION_REPORT_*.md`). RQ1 is a clean negative result (H1 not
supported), RQ2/RQ3 are signed negative/exploratory results (H2/H3 not supported
after Holm), and RQ4/EAR is dropped under Doc 1 §7.4 after a bounded attempt — **no EAR numbers are
claimed in this draft**. The protocol was frozen before results; negative results
are reported as measured.

**Paper artifacts.** Sections: `paper_writing/manuscript/*.md`. Figures:
`paper_writing/figures/`. References: `paper_writing/literature/references.bib`;
verified citation register: `docs/04_LITERATURE_AND_NOVELTY.md`.

---

"""


def main() -> None:
    parts = [HEADER]
    for name, _ in SECTIONS:
        path = Path("paper_writing/manuscript") / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        parts.append(f"<!-- source: paper_writing/manuscript/{name} -->\n\n{text}\n\n---\n\n")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
