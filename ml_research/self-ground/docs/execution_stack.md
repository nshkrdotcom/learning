# Execution Stack

SELF-GROUND is a task/evidence harness built on existing mechanistic
interpretability libraries. It does not own a generic execution backend or
intervention framework.

## TransformerLens

Used for:

- local model loading,
- tokenization through the model tokenizer,
- activation capture at named hook points,
- `run_with_hooks` activation patching,
- logits after patched and unpatched forward passes.

SELF-GROUND code that calls TransformerLens should remain a thin call site for
the specific negation experiment artifacts.

## SAELens

Used for:

- pretrained SAE loading,
- SAE metadata where available,
- SAE encode/decode,
- SAE feature activations.

SELF-GROUND adds semantic compatibility checks around SAELens objects so a
same-width SAE for the wrong model, hook, layer, or hook type fails closed.

## SELF-GROUND

Owned by this repo:

- negation task generation,
- token validation,
- feature/control selection from artifact-backed rankings,
- activation-density-matched control selection,
- result aggregation,
- conservative claim status,
- local JSONL/CSV/Markdown audit artifacts.

## Not Owned By SELF-GROUND

SELF-GROUND does not own:

- a generic model execution engine,
- a generic activation patching framework,
- an SAE training framework,
- a benchmark suite,
- an experiment dashboard,
- a generic plugin/backend platform.

## Current Artifact Signals

Execution configs must record:

```json
{
  "engine_backend": "transformer_lens",
  "sae_backend": "sae_lens"
}
```

Residual smoke diagnostics and feature-space proxy results remain
claim-ineligible. The local `mechanism_report.json` and `mechanism_report.md`
artifacts are the source of truth for SELF-GROUND claim status.
