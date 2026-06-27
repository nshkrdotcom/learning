# Causal Verification

Causal verification runs intervention operations, writes receipts, and records telemetry. It is the first layer that can produce intervention evidence, but claim-bearing use still requires a `PredictionLock`, a passing static compiler gate, and clean blockers.

## Command

Diagnostic dry run:

```bash
uv run mwb verify docs/fixtures/hypothesis_phase5.json --diagnostic-only --dry-run
```

Executed diagnostic run from fixture data:

```bash
uv run mwb verify docs/fixtures/hypothesis_phase5.json --diagnostic-only
```

Outputs are written under:

```text
.mechanism/runs/<run_ref>/
```

The emitted files include:

- `verification_run.json`
- `run_manifest.json`
- `intervention_receipts.jsonl`
- `verification_results.jsonl`
- `telemetry.jsonl`
- `control_metrics.json`
- `blocker_report.json` when blockers exist

SQLite indexes are rebuildable for verification runs, intervention receipts, verification results, and telemetry reports.

## Operations

Supported receipt operations include:

- `resample_ablate`
- `noising`
- `denoising`
- `feature_amplify`
- `zero_ablate`

`noising` and `denoising` are recorded with opposite `causal_direction` values:

- `noising`: `clean_to_corrupt`
- `denoising`: `corrupt_to_clean`

`feature_amplify` records the amplification `coefficient`.

`zero_ablate` is allowed as a diagnostic operation, but its default claim ceiling is `diagnostic_only`.

## Metrics

For each operation, the workbench computes:

- `baseline_margin`
- `intervened_margin`
- `target_delta`
- `matched_control_delta`
- `specificity_gap`
- `effect_size`

Telemetry computes:

- `kl_drift`
- `activation_norm_drift`

If KL or norm drift exceeds configured thresholds, the run records `off_manifold_intervention`.

## Real Adapter Path

The real integration path uses TransformerLens and SAELens to:

1. Run clean and corrupt prompts with activation caches.
2. Encode the hook activation with the SAE.
3. Resample a selected feature activation from corrupt into clean.
4. Decode the SAE delta back into the residual stream.
5. Patch the TransformerLens hook and re-run logits.
6. Write the same receipts, metrics, and telemetry as fixture-backed verification.

Run it explicitly:

```bash
MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_causal_verification_integration.py -m integration
```

## MechanismCard Evidence Examples

MechanismCards may cite:

- `causal_necessity` only when a claim-bearing run has a prediction lock, passing static gate, exact intervention receipts, clean controls, and telemetry below thresholds.
- `diagnostic_only` when the run used `--diagnostic-only`, dry-run planning, zero ablation under the default policy, or off-manifold telemetry.
- `insufficient_evidence` when intervention receipts exist but blockers such as `off_manifold_intervention` or `control_leaky` remain unresolved.

Projection-only and static-compiler-only evidence must not be upgraded to causal language.

## Claim Boundary

Receipts prove that an intervention was specified and executed or planned under recorded conditions. They do not by themselves prove mechanism-level explanation, generalization, or mediation.
