# MechanismLab Architecture

MechanismLab is the reusable claim/evidence/run framework being extracted from
SELF-GROUND. SELF-GROUND remains the first project and case study; it is not the
entire framework.

## Scope

MechanismLab owns:

- typed claim specifications,
- experiment specifications,
- run manifests,
- artifact contracts,
- artifact-gated claim reports,
- backend/plugin manifests,
- local tracking and artifact registration.

MechanismLab does not own:

- activation patching,
- SAE encode/decode,
- benchmark engines,
- remote model execution,
- scientific truth by dashboard state.

Backends execute interventions. MechanismLab records what ran, checks that
required artifacts exist, and evaluates claim status from artifact-backed
evidence payloads.

## SELF-GROUND Relationship

SELF-GROUND is the first project plugin/compatibility layer. It contributes a
negation-scope claim template, Phase 3 experiment template, and adapter from
existing SELF-GROUND artifacts into generic MechanismLab reports.

The existing detailed SELF-GROUND report remains authoritative for the current
negation experiment. The generic MechanismLab report is written alongside it as
the reusable claim-ledger substrate.

## Integration Boundaries

- TransformerLens is a local execution and patching backend.
- SAELens is a representation backend for SAE load/encode/decode.
- SAEBench/RAVEL is an optional evaluation backend to probe or wrap.
- nnsight/NDIF is an optional remote execution backend.
- pyvene is an optional serializable/trainable intervention backend.
- W&B and MLflow are trackers, not sources of scientific truth.
- DVC is for large artifact/data versioning, not claim status.
- Hydra is for configuration composition, not evidence semantics.

No optional integration is required by the core package.
