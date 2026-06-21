# SAE Decoded Intervention Blocker

Decoded SAE intervention is not part of Phase 1. Phase 1 intentionally stops at raw residual-dimension intervention because no concrete, tested SAELens release/id has been verified for `EleutherAI/pythia-70m` at `blocks.2.hook_resid_post`.

## Why This Is Blocked

Decoded SAE intervention requires all of the following to be known and tested:

- SAELens release name.
- SAE id.
- Model and hook point compatibility.
- Encoder input dimension matching the captured residual activation width.
- Decoder output shape matching the hook activation shape.
- A confirmed reconstruction or delta-patching convention.

Without that information, adding decoded SAE intervention would be a placeholder integration, which this repo does not allow.

## Candidate SAE Verification

Set:

```bash
export SELF_GROUND_SAE_RELEASE=<tested-release>
export SELF_GROUND_SAE_ID=<tested-sae-id>
```

Then run:

```bash
uv run pytest --run-integration tests/integration/test_sae_adapter_optional.py
```

A successful verification should confirm:

- the SAE loads through `SAELensAdapter.from_pretrained`,
- `adapter.d_in` matches the residual activation width for the target hook point,
- `encode` returns `[batch, d_sae]` or `[batch, position, d_sae]`,
- `decode` can return residual-space activations compatible with the hook point.

## Next Concrete Implementation Step

After a candidate SAE passes shape verification:

1. Capture residual activations at the hook point.
2. Encode them with the verified SAE.
3. Modify selected SAE features.
4. Decode back to residual space.
5. Patch decoded residual activations through TransformerLens hooks.
6. Measure logit-contrast deltas on `x_pos`, `x_neg`, `x_para`, and `x_decoy`.

Only then should `sae_interventions.py` be added.
