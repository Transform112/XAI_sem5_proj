# `src/analysis/tests/` — analysis unit tests

Tests for the analysis toolkit: artifact layout/path contracts, drift/noise
math, RQ2/RQ3 readiness, and example fixtures.

## How to execute

```bash
python -m pytest src/analysis/tests -q
```

(Requires the project environment with `pyyaml`, `numpy`, `pandas`,
`scipy`, `torch`; the system Python lacks these, so CI/venv is required.)
Tests are included from `src/analysis/` and use fixtures in
`src/analysis/examples/`.
