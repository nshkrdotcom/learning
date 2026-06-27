# Reference Mechanism Benchmarks

Reference mechanism benchmarks test the workbench against tasks with known ground truth. They are framework benchmarks, not model leaderboards.

## Command

Run the built-in toy reference suite:

```bash
uv run mwb benchmark framework
```

Select a suite explicitly:

```bash
uv run mwb benchmark framework --suite toy
```

The command writes:

- `.mechanism/benchmarks/latest_framework_benchmark.json`
- `.mechanism/benchmarks/<benchmark_ref>.json`

The report is also indexed in SQLite as `benchmark_reports`, with per-task rows in `reference_tasks`.

## Built-In Toy Suite

The `toy` suite includes three deterministic reference tasks:

- `toy_residual_sign`: a planted residual-direction task where the direct writer is the known mechanism.
- `negative_control_surface_confound`: a tempting surface-token confound with high proxy score and no causal effect.
- `synthetic_sae_split_absorption`: a synthetic dictionary containing a known split latent and a known absorbed feature.

These tasks are intentionally small, deterministic fixtures. They are not evidence about a real model. They check whether the workbench can recover known mechanisms, block false positives, and detect dictionary artifacts before users trust it on real backends.

## Scoring

Candidate units are scored with:

- proxy score,
- exact intervention effect,
- empirical null effects,
- empirical p-values,
- Benjamini-Hochberg FDR-adjusted q-values.

Known-mechanism tasks pass when the expected unit is significant under the empirical null and is the top exact-effect unit.

Negative-control tasks pass when a tempting proxy feature is rejected as a causal unit and recorded with a blocker such as `tempting_confound`.

Synthetic SAE tasks pass when the detector finds expected feature splitting and absorption patterns from the fixture weights.

## Calibration Fields

Each report includes:

- `proxy_vs_exact_correlation`
- `fdr_adjusted_p_value`
- `significant_candidate_count`
- `null_seed_count`
- `empirical_null_abs_p95`
- `fdr_alpha`

These fields are the first calibration loop: proxy methods must keep earning trust against exact effects and empirical nulls.

## Contribution Guide

New benchmark suites should be deterministic, small enough for CI, and explicit about ground truth.

Required task fields:

- stable `task_id`,
- `task_kind`,
- `ground_truth`,
- deterministic `fixture`,
- pass/fail scoring rule,
- at least one negative control or empirical-null path when candidates are scanned.

Do not add a benchmark that only checks whether files are written. Every benchmark must test a mechanistic classification, blocker, or calibration behavior that can fail.

Optional heavy integrations, such as Tracr-compiled models, ACDC/EAP, or SAEBench/RAVEL fixtures, should stay behind optional dependencies or integration markers until they are reproducible in CI.
