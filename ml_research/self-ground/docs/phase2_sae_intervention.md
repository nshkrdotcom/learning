# Phase 2: Decoded SAE Feature Intervention

## What Decoded SAE Intervention Means

Decoded SAE intervention means the pipeline captures real residual activations, encodes them with a real SAELens SAE, modifies selected SAE feature activations, decodes back to residual space, patches those decoded residual activations into the real TransformerLens model, reruns logits, and measures logit-contrast changes.

## Semantic SAE Compatibility

Shape compatibility is necessary but not sufficient. A same-width SAE can still
be invalid for decoded intervention if it was trained on a different checkpoint,
layer, hook type, hook point, or activation convention.

A decoded SAE intervention is production-compatible only when all of the
following pass:

- the requested TransformerLens model matches the model declared by SAE metadata,
- the requested hook point matches the SAE hook metadata,
- the requested hook layer matches when layer metadata is available,
- the requested hook type matches when hook-type metadata is available,
- activation width matches,
- captured activation shape is supported,
- SAE encode/decode shapes are patch-compatible,
- reconstruction metrics are finite.

`EleutherAI/pythia-70m` and `EleutherAI/pythia-70m-deduped` are different
checkpoints. A SAE declaring `pythia-70m-deduped` must not be used with
`EleutherAI/pythia-70m`, even when tensor widths match.

The compatibility artifact separates:

- `shape_compatible`
- `metadata_compatible`
- `reconstruction_compatible`
- `compatible`

`compatible=true` requires all production checks to pass. Shape-only diagnostic
output is not sufficient for decoded intervention.

## Why Compatibility Verification Is Required

SAE artifacts are tied to model architecture, hook point, activation width, and SAELens API behavior. Before intervention, the repo verifies:

- declared SAE model and requested model,
- declared SAE hook and requested hook,
- model activation shape,
- encoded SAE activation shape,
- decoded residual shape,
- `d_model`,
- `d_sae`,
- whether decoded output can patch the hook activation,
- finite reconstruction MSE, relative L2, and max absolute error.

If compatibility fails, no intervention rows are written.

## Patch Modes

`replace` uses the decoded modified residual as the hook activation where semantically and shape-compatible.

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

Blocked runs are successful safety behavior when metadata, shape, or
reconstruction checks fail. They should not be interpreted as implementation
failure unless the blocker reports a repo bug.

## Next Phase

Phase 3 is implemented as multi-task token-contrast evaluation and candidate
evidence reporting. See `docs/phase3_token_contrast_evaluation.md`.
