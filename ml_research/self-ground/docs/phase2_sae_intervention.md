# Phase 2: Decoded SAE Feature Intervention

## What Decoded SAE Intervention Means

Decoded SAE intervention means the pipeline captures real residual activations, encodes them with a real SAELens SAE, modifies selected SAE feature activations, decodes back to residual space, patches those decoded residual activations into the real TransformerLens model, reruns logits, and measures logit-contrast changes.

## Why Compatibility Verification Is Required

SAE artifacts are tied to model architecture, hook point, activation width, and SAELens API behavior. Before intervention, the repo verifies:

- model activation shape,
- encoded SAE activation shape,
- decoded residual shape,
- `d_model`,
- `d_sae`,
- whether decoded output can patch the hook activation.

If compatibility fails, no intervention rows are written.

## Patch Modes

`replace` uses the decoded modified residual as the hook activation where shape-compatible.

`delta` computes:

```text
decoded_modified - decoded_original
```

and adds that delta to the original hook activation. `delta` is the default because it preserves the original activation outside the SAE-induced change.

## Operations

`ablate` sets selected SAE feature activations to zero.

`amplify` multiplies selected SAE feature activations by `factor`. A factor of `1.0` is rejected because it is a no-op.

## Metrics

Each intervention row records baseline and patched logit contrasts for:

- `x_pos`
- `x_neg`
- `x_para`
- `x_decoy`

It then records signed and absolute deltas:

```text
signed_negation_delta_mean = mean(delta[x_pos], delta[x_para])
signed_control_delta_mean = mean(delta[x_neg], delta[x_decoy])
absolute_negation_delta_mean = mean(abs(delta[x_pos]), abs(delta[x_para]))
absolute_control_delta_mean = mean(abs(delta[x_neg]), abs(delta[x_decoy]))
signed_specificity_score = signed_negation_delta_mean - signed_control_delta_mean
absolute_specificity_score = absolute_negation_delta_mean - absolute_control_delta_mean
```

## Artifacts

Successful run:

- `config.json`
- `compatibility.json`
- `selected_features.json`
- `intervention_results.jsonl`
- `summary.csv`
- `README.md`

Blocked run:

- `config.json`
- `compatibility.json`
- `README.md`

## Limitations

This does not prove complete mechanism discovery or genuine introspection. It tests whether configured SAE features have measurable decoded-intervention effects on a negation-related logit contrast under matched controls.

## Next Phase

The next phase should compare SAE decoded intervention effects against richer behavioral tasks, alternative token contrasts, and cross-layer candidate mechanisms.
