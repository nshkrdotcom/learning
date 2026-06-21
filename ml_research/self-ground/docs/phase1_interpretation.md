# Phase 1 Interpretation

## What Residual Ranking Means

Residual ranking computes a contrast over real activations:

```text
score = mean(x_pos) + mean(x_para) - mean(x_neg) - mean(x_decoy)
```

For Phase 1, features are raw residual stream dimensions named `resid_N`. A high absolute score means that dimension differs between negation/paraphrase examples and matched affirmation/decoy examples.

## Why Residual Dimensions Are Limited

Raw residual dimensions are basis-dependent. A single dimension is not a sparse feature and should not be treated as a semantic mechanism. Residual ranking is useful as a real, low-friction first signal, not as final interpretability evidence.

## What Residual Intervention Proves

Residual intervention patches selected residual dimensions in a real TransformerLens forward pass. If logits change, the run proves that those raw dimensions causally affect the measured logit contrast under that patch.

The Phase 1 intervention metric compares absolute logit-contrast movement on negation conditions with movement on matched controls:

```text
negation_specific_delta = mean(abs(delta[x_pos]), abs(delta[x_para]))
control_delta = mean(abs(delta[x_neg]), abs(delta[x_decoy]))
specificity_score = negation_specific_delta - control_delta
```

## What It Does Not Prove

It does not prove sparse-feature mechanism discovery. It does not identify an SAE feature. It does not show broad model understanding. It does not establish genuine introspection.

## Why SAE Decoded Intervention Comes Next

The next step is to replace raw residual dimensions with real SAE features: encode residual activations, ablate or amplify selected SAE features, decode back to residual space, patch the model, and measure logit/behavior changes. That is Phase 2.
