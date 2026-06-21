# SAE Decoded Intervention Compatibility Workflow

Phase 2 includes compatibility verification and decoded SAE intervention infrastructure. Real decoded intervention runs only when compatibility succeeds for a concrete SAELens release/id.

If no SAE release/id is available, this document is the blocker workflow. It is not a reason to fabricate decoded intervention outputs.

## Known Compatible Evidence Run

The repo has a verified small SAE path:

- model: `EleutherAI/pythia-70m-deduped`
- hook point: `blocks.2.hook_resid_post`
- SAE release: `pythia-70m-deduped-res-sm`
- SAE id: `blocks.2.hook_resid_post`

The successful compatibility, ranking, and decoded intervention evidence is
summarized in `docs/phase2_run_evidence.md`.

## What Is Needed

Decoded SAE intervention requires all of the following to be known and tested:

- SAELens release name.
- SAE id.
- Model and hook point metadata compatibility.
- Encoder input dimension matching the captured residual activation width.
- Decoder output shape matching the hook activation shape.
- Finite reconstruction metrics.
- A confirmed reconstruction or delta-patching convention.

Without that information, the repo can still run Phase 1 and can write a structured SAE compatibility failure artifact, but it must not write decoded intervention rows.

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

Or run the Phase 2 compatibility command:

```bash
uv run python scripts/check_sae_compatibility.py \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --sae-release "$SELF_GROUND_SAE_RELEASE" \
  --sae-id "$SELF_GROUND_SAE_ID" \
  --device cpu \
  --out runs/check_sae_compatibility.json
```

A successful verification confirms:

- the SAE loads through `SAELensAdapter.from_pretrained`,
- the SAE-declared model matches the requested model,
- the SAE-declared hook matches the requested hook point,
- `adapter.d_in` matches the residual activation width for the target hook point,
- `encode` returns `[batch, d_sae]` or `[batch, position, d_sae]`,
- `decode` can return residual-space activations compatible with the hook point,
- reconstruction metrics are finite.

`EleutherAI/pythia-70m` and `EleutherAI/pythia-70m-deduped` are distinct
checkpoints. A deduped SAE must fail closed if requested against the non-deduped
checkpoint.

## Decoded Intervention Command

After a candidate SAE passes semantic compatibility verification:

```bash
uv run python scripts/run_real_sae_intervention.py \
  --ranking-dir runs/real_sae_ranking_pythia70m \
  --out runs/real_sae_intervention_pythia70m \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --sae-release "$SELF_GROUND_SAE_RELEASE" \
  --sae-id "$SELF_GROUND_SAE_ID" \
  --top-k-features 5 \
  --operation ablate \
  --patch-mode delta \
  --device cpu
```

If compatibility fails, the run writes `config.json`, `compatibility.json`, and `README.md` explaining the blocker.
