# SAE Decoded Intervention Blocker

Decoded SAE reinjection is not implemented in this milestone because no concrete, tested SAELens release/id was provided for `EleutherAI/pythia-70m` at `blocks.2.hook_resid_post`.

## Command Attempted

```bash
uv run pytest --run-integration tests/integration/test_sae_adapter_optional.py
```

## Result

The test is correctly environment-gated and skips unless both variables are set:

```bash
SELF_GROUND_SAE_RELEASE=...
SELF_GROUND_SAE_ID=...
```

Without those identifiers, there is no verified real SAE artifact to load, no confirmed `d_in`/`d_sae` shape pair, and no safe basis for implementing decoded reinjection.

## Next Concrete Step

Run:

```bash
SELF_GROUND_SAE_RELEASE=<tested-release> \
SELF_GROUND_SAE_ID=<tested-sae-id> \
uv run pytest --run-integration tests/integration/test_sae_adapter_optional.py
```

Once encode/decode shapes are confirmed, implement `sae_interventions.py` against that real SAE and add an integration test that measures logit-contrast change after decoded residual patching.
