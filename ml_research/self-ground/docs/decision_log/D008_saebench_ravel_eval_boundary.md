# D008: SAEBench/RAVEL Evaluation Boundary

Date: 2026-06-21

## Decision

Current SELF-GROUND Phase 3 results are RAVEL-shaped custom token-contrast
evaluation, not upstream SAEBench or upstream RAVEL execution.

## Probe Result

Probe command:

```bash
uv run python scripts/probe_saebench_ravel_bridge.py \
  --out runs/tooling_spikes/saebench_ravel_bridge
```

Artifact:

```text
runs/tooling_spikes/saebench_ravel_bridge/probe_result.json
```

Status in this environment: `not_installed`.

The probe attempted these imports:

- `sae_bench`
- `saebench`
- `sae_bench.evals.ravel`
- `sae_bench.evals.ravel.eval`
- `ravel`

Each attempt failed with `ModuleNotFoundError`. No real upstream eval ran.

## Custom Temporary Evaluator

The current temporary evaluator is:

```text
scripts/run_negation_ravel_eval.py
src/self_ground/real_behavioral_intervention.py
src/self_ground/ravel_adapter/scoring.py
```

It runs real TransformerLens + SAELens decoded intervention and scores
SELF-GROUND token contrasts. It is RAVEL-shaped because it separates
target-prompt movement from matched-control movement, but it is not an upstream
RAVEL benchmark implementation.

## Replacement Requirement

Replacing the custom evaluator requires an installed upstream SAEBench/RAVEL API
that can accept all of:

- SELF-GROUND negation task rows or an adapter to them,
- the current TransformerLens model and SAELens SAE, or precomputed compatible
  activations,
- cause/isolation scoring over matched target/control prompts,
- artifact export sufficient for SELF-GROUND claim auditing.

If the upstream API assumes factual entity attributes too deeply, SELF-GROUND
must keep a minimal custom negation evaluator and document the exact delta.

## Disallowed Claims Until Replacement

Until an upstream eval actually runs, reports must not claim:

- real SAEBench integration,
- real upstream RAVEL benchmark results,
- broad RAVEL generality,
- benchmark-level causal localization.

Supported wording is:

```text
RAVEL-shaped custom token-contrast evaluation over real decoded SAE
interventions.
```
