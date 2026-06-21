# QC Checklist

This checklist records the Phase 2 implementation review state. Commit and push
are verified after this file is committed.

## Phase 0: Public Surface And Proxy Isolation

- [x] no fake adapters in `src`
- [x] no fake CLI modes
- [x] no `--adapter fake` option
- [x] stale non-proxy `FeatureEffect` schema removed
- [x] feature-space proxy outputs remain explicitly named proxy
- [x] public CLI exposes real Phase 1 and Phase 2 commands

## Phase 1: Real Residual Pipeline Preservation

- [x] `uv sync` passes
- [x] fast tests pass
- [x] integration tests pass with optional SAE tests skipped when env is absent
- [x] real model check command runs
- [x] real activation ranking command runs
- [x] real residual intervention command runs
- [x] real artifacts are written under `runs/check_real_model.json`
- [x] real artifacts are written under `runs/test_real_activation_ranking`
- [x] real artifacts are written under `runs/test_real_residual_intervention`

## Phase 2: SAE Decoded Intervention Infrastructure

- [x] SAE compatibility checker implemented
- [x] SAE compatibility checker writes structured JSON
- [x] missing SAE release/id produces explicit blocker artifact
- [x] concrete SAE release/id identified from SAELens metadata
- [x] concrete SAE compatibility run passes
- [x] concrete SAE ranking run passes
- [x] concrete SAE decoded intervention run passes
- [x] SAE encoded shape validation checks batch and sequence dimensions
- [x] decoded SAE patching primitives implemented
- [x] decoded SAE intervention experiment implemented
- [x] decoded SAE intervention writes no rows when compatibility fails
- [x] SAE ranking requires real `--sae-release` and `--sae-id`
- [x] optional SAE integration tests skip unless `SELF_GROUND_SAE_RELEASE` and `SELF_GROUND_SAE_ID` are configured

## Phase 2: Semantic SAE Compatibility Hardening

- [x] SAE metadata extraction implemented
- [x] model identity compatibility enforced
- [x] `pythia-70m` vs `pythia-70m-deduped` mismatch rejected
- [x] hook point compatibility enforced
- [x] hook layer compatibility enforced when metadata is available
- [x] hook type compatibility enforced when metadata is available
- [x] shape-only diagnostic cannot enable production intervention
- [x] reconstruction metrics computed and serialized
- [x] semantic mismatch writes blocker artifact and no intervention rows
- [x] SAE ranking rejects semantic mismatch
- [x] docs explain shape compatibility is not sufficient
- [x] known-compatible deduped command documented
- [x] mismatch command verified
- [x] metadata-correct compatibility command verified
- [x] decoded intervention command verified

## Documentation And Claims

- [x] README updated
- [x] Phase 2 docs updated
- [x] SAE blocker workflow documented
- [x] artifacts documented
- [x] no proxy output is described as causal evidence
- [x] no residual-dimension result is described as sparse SAE mechanism discovery
- [x] no complete SELF-GROUND, broad mechanism discovery, or genuine introspection claim is made

## Final Review Evidence

- [x] `uv run ruff check .`
- [x] `uv run pytest`
- [x] `uv run pytest --run-integration`
- [x] forbidden production-path scan reviewed
- [x] overclaiming scan reviewed
- [x] artifact contents inspected
- [x] code reviewed systematically against implementation prompt

## Git Finalization

- [x] `git status` reviewed before implementation closeout
- [x] commit `0cd9742` created for the Phase 2 implementation
- [x] commit `0cd9742` pushed to `origin/main`
- [x] additional real SAE evidence prepared for commit
- [x] evidence commit created
- [x] evidence commit pushed
