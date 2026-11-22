"""XDRIFT analysis toolkit — stages S5 and S8–S13 (analysis only; no training).

Purpose and authority
---------------------
This package is the *standalone, reproducible* implementation of every
quantitative analysis the paper needs. It never trains a model, never mutates
data state, splits, masks, anchor sets or thresholds (Protocol v1 is FROZEN at
G1c, ``docs/DECISIONS.md`` D-042). Every definition is taken from:

* ``docs/05_EXPERIMENT_PROTOCOL.md`` (FROZEN v1):
  §8 drift metrics, §9 noise floor / excess drift / H1, §10 RQ2 structural
  change, §11 RQ3 drift→forgetting, §12 RQ4 / H4, §13 statistics.
* ``docs/source_documents/Doc 2b - Implementation Guide Part 2 (S7-S13).md``:
  S8 drift metrics + noise floor, S9 excess drift / RQ1, S10 RQ2, S11 RQ3,
  S12 EAR comparison, S13 statistics / tables / figures / numbers file.
* ``docs/source_documents/Doc 2 - Implementation Guide ... .md`` §S5: CL evaluation matrix
  ``R[i, j]``, final average F1, forgetting ``F_j``, ``BWT`` and the per-class
  F1 tensor (RQ3 input).
* ``research/lit/claims_evidence_skeleton.md``: the **binding artifact paths**
  for claims C1–C7 and the ``paper_writing/manuscript/numbers.json`` schema.
* ``docs/DECISIONS.md`` D-033 (rare-class policy / dual macro-F1), D-030
  (hierarchical pilot bootstrap), D-019 (RQ3 competing baselines).

Module map
----------
``artifacts``     exact artifact path registry copied from the claim→evidence
                  skeleton (``results/text/tables/*.csv``, ``results/images/s8_*``,
                  ``paper_writing/manuscript/numbers.json``) — every other module writes through it.
``metrics_drift`` S8 similarity/drift family: attribution cosine, top-k
                  Jaccard (k = 3/5/10), Spearman, edge-mask Jaccard@m=5 and
                  cosine; primary metric = ``1 - Jaccard(top-5)``.
``metrics_cl``    S5 ``R[i,j]`` matrix, final mean macro-F1, forgetting, BWT,
                  per-class F1 tensor (npz + csv) and the pre-registered later
                  forgetting labels used by S11.
``stats``         docs/05 §13 statistics primitives: percentile bootstrap,
                  stratified + hierarchical bootstrap, paired bootstrap,
                  Spearman / partial Spearman, permutation test, Holm, effect
                  sizes, L2 logistic-regression AUC under leave-one-period-out
                  CV, paired AUC-delta bootstrap.
``noise_floor``   S8/S9: null-update and explainer-repeat drift distributions,
                  per-class q95, excess = drift − null mean, above-floor flags,
                  H1 tests (fraction above floor vs 5%, binomial + normal
                  approximation, Wilcoxon on excess, Holm across the family).
``rq3``           S11: competing-baseline helper (drift + confidence/margin/
                  retention/F1-level vs baselines only), ΔAUC with CI.
``synthetic``     deterministic synthetic fixtures with the real schemas, used
                  by the test suite and the demo driver (never real results).
``tables``        S13 table generators writing the skeleton's exact CSV paths.
``figures``       S13 figure generators writing ``results/images/s8_*`` etc.
``numbers``       ``paper_writing/manuscript/numbers.json``-compatible exporter + validator.
``run_synthetic_demo``  one-command driver that proves the artefact contract on
                  synthetic data under ``src/analysis/examples/`` (never writes
                  synthetic data into the real ``results/`` tree).
``s5_crosscheck``  T19: cross-checks the engineer S5 writer artifacts against
                  this reference (R matrix, dual macro, per-class, forgetting/BWT)
                  and writes ``src/analysis/reports/s5_crosscheck.md``.
``validators``    T19: schema/column/invariant validators for real artifacts
                  (``python -m src.analysis.validators --stage s8``).
``config_defaults/`` T19: proposed ``configs/{drift,rq3,ear}.yaml`` defaults for
                  the engineer to materialise (analyst does not write configs/).
``pipeline_fixtures`` T20: schema-faithful synthetic S6 anchor sets / S7
                  explanation cache fixtures (tests + demo only).
``pipeline_readiness`` T20: artifact → producer CLI → validator-stage readiness
                  report (``src/analysis/reports/pipeline_readiness.md``).
``s8_validation`` T20: independent S8/S9 validation + H1 arithmetic recomputation
                  (``src/analysis/reports/s8_validation.md``).

Tests are under ``src/src/analysis/tests`` (``pytest analysis``); no synthetic artefact
is ever written to the real results tree.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
