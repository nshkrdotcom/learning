# Head-Specific Candidate Failure Taxonomy v1

## Status

Pre-registered diagnostic pass.

This pass is written after `Head-Specific Candidate Characterization v1` and before running any new candidate search. It does not promote any previously falsified head back to candidate status.

## Source Result

The source result is:

- `docs/results/head_specific_candidate_characterization_v1.md`
- `docs/learning_notes/2026-06-26_candidate_characterization.md`
- `reports/head_specific_candidate_characterization_v1/counterexamples/`

The completed characterization result classified all 16 fixed heads as `falsified_candidate`. The primary heads `L7H7`, `L9H11`, `L7H11`, `L7H0`, and `L0H8` all failed the characterization rule. Prior raw-attention comparison heads and deterministic negative controls also ended falsified, while some comparison or negative-control rows still produced seed-level support. That means the slice-level rules remain false-positive-prone.

## Research Question

Why did the fixed candidate set fail, and what failure modes must be controlled before any future head search is justified?

## Non-Goals

This pass will not:

- search for new heads;
- add new candidate heads;
- reinterpret falsified heads as promising;
- claim an induction head, circuit, or broad GPT-2 property;
- use the taxonomy as a winner-selection mechanism.

## Inputs

Primary counterexample inputs:

- `reports/head_specific_candidate_characterization_v1/counterexamples/counterexamples_L7H7.csv`
- `reports/head_specific_candidate_characterization_v1/counterexamples/counterexamples_L9H11.csv`
- `reports/head_specific_candidate_characterization_v1/counterexamples/counterexamples_L7H11.csv`
- `reports/head_specific_candidate_characterization_v1/counterexamples/counterexamples_L7H0.csv`
- `reports/head_specific_candidate_characterization_v1/counterexamples/counterexamples_L0H8.csv`

Summary input:

- `docs/results/head_specific_candidate_characterization_v1.md`

The counterexample CSV files are the row-level source of truth. The summary document is used only for global facts such as final status, negative-control support, and whether local OV/QK diagnostic signatures appeared without causal specificity.

## Failure Modes

The taxonomy uses deterministic labels:

- `control_moved`: a control family moved enough to defeat specificity.
- `target_swap_leak`: target-swap controls produced large effects.
- `reversed_control_leak`: reversed-control prompts produced effects.
- `domain_flip`: symbolic, word, or number domains disagreed.
- `length_flip`: short, medium, or long variants disagreed.
- `intervention_disagreement`: clean-to-corrupt patching, zero ablation, and mean ablation disagreed.
- `position_mismatch`: final and previous/source positions disagreed.
- `attention_effect_decoupled`: attention/effect correlation was weak, absent, or opposite-signed.
- `ov_qk_local_only`: OV/QK local diagnostic support appeared without causal specificity.
- `negative_control_support`: negative-control or comparison heads produced comparable seed-level support, weakening slice-level interpretation.

## Deterministic Classification Rules

Row-level rules:

- Rows already labeled `control_moved` map to `control_moved`.
- Rows already labeled `wrong_target_control_moved` or rows from `target_swap` families map to `target_swap_leak`.
- Rows from `reversed_control` families with positive effect map to `reversed_control_leak`.
- Positive rows whose effect sign conflicts across `token_domain` for the same head map to `domain_flip`.
- Positive rows whose effect sign conflicts across `sequence_length_bucket` for the same head map to `length_flip`.
- Positive rows whose effect sign conflicts across interventions for the same head map to `intervention_disagreement`.
- Positive rows whose effect sign conflicts across positions for the same head map to `position_mismatch`.
- Rows already labeled `token_domain_or_length_failure` map to `domain_flip` or `length_flip` using available metadata.
- Rows already labeled `position_intervention_mismatch` map to `position_mismatch` or `intervention_disagreement` using available metadata.

Head-level rules:

- A primary head with weak or negative mean attention/effect correlation in the consolidated result receives `attention_effect_decoupled`.
- A head with OV/QK support in the consolidated result but final status `falsified_candidate` receives `ov_qk_local_only`.
- If the consolidated result records support rows for comparison or negative-control groups, primary-head summaries include `negative_control_support` as a global caution.

## Outputs

The script will write:

- `reports/head_specific_candidate_characterization_v1/failure_taxonomy/failure_taxonomy_by_row.csv`
- `reports/head_specific_candidate_characterization_v1/failure_taxonomy/failure_taxonomy_by_head.csv`
- `reports/head_specific_candidate_characterization_v1/failure_taxonomy/failure_taxonomy_summary.json`
- `reports/head_specific_candidate_characterization_v1/failure_taxonomy/failure_taxonomy.md`
- `docs/results/head_specific_candidate_characterization_failure_taxonomy_v1.md`

## Decision Rule

The taxonomy can justify only one kind of next step: calibration. It cannot justify a new candidate search.

A future candidate search remains blocked if the taxonomy shows that:

- controls moved often enough to defeat specificity;
- target-swap or reversed controls produced large effects;
- domain or length flips were common;
- intervention or position variants disagreed;
- negative controls produced support under the same seed-level logic.

## Refused Claims

This taxonomy does not show an induction head, a circuit, or a broad GPT-2 property. It diagnoses failure modes in a local practice pipeline.

## Reproduction Command

```bash
uv run python scripts/build_characterization_failure_taxonomy.py \
  --counterexamples reports/head_specific_candidate_characterization_v1/counterexamples \
  --summary docs/results/head_specific_candidate_characterization_v1.md \
  --output reports/head_specific_candidate_characterization_v1/failure_taxonomy
```
