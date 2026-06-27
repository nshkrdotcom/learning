# Head-Specific Induction Causality v1

## Status

Pre-registered practice experiment. This document is the contract for the next head-specific induction causality pass and was written before implementing the new sweep.

## Motivation

The lab has already shown that raw previous-occurrence attention can be misleading on simple repeated-token prompts. The next practice step is to test whether stricter head-specific interventions and stricter metrics change that conclusion.

## Prior result being addressed

The previous controlled patching pass used full layer-level attn_out patching, not head-specific patching. It found that raw high-attention heads were false-positive-prone and that random comparison heads produced the only replicated positive-specific signal. This experiment tests whether a stricter head-specific intervention and stricter metric changes that conclusion.

## Research question

Can GPT-2 small induction-like repeated-token behavior be causally linked to specific attention heads, using head-specific interventions and a metric that separates true induction from target/logit/control artifacts?

## Hypotheses

H0: No selected head shows replicated positive-minus-control causal effect under head-specific intervention.

H1: At least one selected head shows a replicated positive-minus-control causal effect under head-specific intervention.

Expected learning result: Even if H0 holds, the experiment is useful because it tests whether the earlier result was an artifact of layer-level patching and permissive metrics.

## Model

Use only `gpt2-small` through TransformerLens. No Gemma, Pythia, SAEs, nnsight, Neuronpedia, dashboards, databases, or broad circuit-discovery machinery are part of this experiment.

## Prompt families

Use the existing six-family controls for baseline, attention, and logit-lens summaries:

- `positive_repeat_sequence`
- `no_repeat_control`
- `shuffled_repeat_control`
- `distractor_repeat_control`
- `same_token_frequency_control`
- `random_expected_token_control`

For the main head-specific sweep, use at least:

- `positive_repeat_sequence`
- `distractor_repeat_control`
- `random_expected_token_control`
- `same_token_frequency_control`

## Metrics

Primary metric: `true_vs_control_logit_diff`.

Registered metrics:

- `target_logit`: existing simple target logit metric. This is secondary and weaker because target-logit movement can be target-specific without being induction-specific.
- `true_vs_control_logit_diff`: true expected next-token logit minus wrong/control token logit. This is the primary metric.
- `normalized_induction_score`: positive prompt true-vs-control logit diff minus matched control prompt true-vs-control logit diff.

Denominator-zero effect sizes must be recorded as undefined with `effect_size_status=denominator_zero`, not silently set to zero.

## Candidate sets

- `raw_attention_candidates`: top heads from raw previous-occurrence attention.
- `control_firing_candidates`: heads that fire on controls.
- `random_comparison_candidates`: deterministic random heads.
- `known_induction_search_candidates`: all heads or all selected-layer heads scored by a head-specific sweep.

Candidate heads must be discovered from local artifacts or from the bounded sweep. Do not assume known induction heads from memory.

## Interventions

- `head_specific_ablation`: zero or mean-ablate one attention head output at a selected position.
- `head_specific_clean_to_corrupt_patch`: copy one clean head activation into the corrupt/control run.
- `layer_level_attn_out_patch`: allowed only as a comparison to prior work, never labeled head-specific.

The main sweep uses `head_specific_clean_to_corrupt_patch` at the final position unless explicitly recorded otherwise.

## Replication plan

Run seeds 0, 1, and 2. Each seed uses the same model, prompt-family design, primary metric, intervention, selected families, and default examples per family. A seed may use its own generated controls run as source data.

One seed is never enough.

## Decision rules

A candidate is not considered interesting if controls move as much as positives.

A candidate is not considered head-specific unless the artifact records `head_specific_patch=true` and `actual_patch_scope` identifies a single head.

A candidate is not considered replicated unless the same layer/head has positive-minus-control causal gap > 0 on at least two of three seeds.

A random comparison head cannot be used as induction evidence unless the report explicitly explains why it is no longer just a random comparison artifact.

One seed is never enough.

## What counts as a positive result

A narrow positive result requires the same layer/head to be `head_specific_positive_candidate` in at least two of three seeds under `true_vs_control_logit_diff`, with controls moving less than positives and all relevant artifacts recording `head_specific_patch=true`.

Even then, it is only a narrow practice candidate under this prompt set, component, position, intervention, and metric.

## What counts as a negative result

The experiment is negative if no same layer/head replicates under the decision rules, if raw-attention heads become nonspecific, if random comparison heads are the only apparent positive candidates, or if head-specific patching is unsupported and only layer-level fallback is available.

A negative result is useful because it makes the lab harder to fool.

## What will not be claimed

This experiment will not claim a discovered induction head, a full circuit, broad GPT-2 behavior, or a publishable benchmark result from one seed, raw attention, target-logit movement, or layer-level `attn_out` patching.

## Run commands

```bash
uv run python scripts/inspect_head_hooks.py \
  --config configs/gpt2_small_induction_controls.yaml \
  --prompt "A B C D A B C"

uv run python scripts/run_head_specific_induction_sweep.py \
  --config configs/gpt2_small_head_specific_induction.yaml \
  --source-run runs/20260626_144001_gpt2_small_induction_controls

uv run python scripts/compare_head_specific_induction_runs.py \
  --runs \
    runs/<head_specific_seed0_run> \
    runs/<head_specific_seed1_run> \
    runs/<head_specific_seed2_run>
```

## Artifact list

Expected hook inspection artifacts:

- `head_hook_inspection.json`
- `head_hook_inspection.md`

Expected per-seed sweep artifacts:

- `config.yaml`
- `source_run.txt`
- `prompt_sample.csv`
- `head_hook_resolution.json`
- `head_specific_patching_results.csv`
- `head_specific_patching_by_family.csv`
- `head_specific_patching_by_head.csv`
- `head_specific_induction_summary.json`
- `summary.md`
- `figures/head_specific_head_gaps.png`
- `figures/head_specific_status_counts.png`

Expected consolidated artifacts:

- `reports/head_specific_induction_causality_v1/run_manifest.json`
- `reports/head_specific_induction_causality_v1/head_specific_multiseed_by_head.csv`
- `reports/head_specific_induction_causality_v1/head_specific_multiseed_summary.json`
- `reports/head_specific_induction_causality_v1/head_specific_induction_causality_v1.md`
- `reports/head_specific_induction_causality_v1/figures/multiseed_head_gaps.png`
- `reports/head_specific_induction_causality_v1/figures/status_by_seed.png`
