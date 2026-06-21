# Migrating SELF-GROUND Toward MechanismLab

This migration is intentionally incremental.

## Already Extracted

- Claim specs: `mechanismlab.core.claims`
- Experiment specs: `mechanismlab.core.experiments`
- Run manifests: `mechanismlab.core.runs`
- Artifact contracts: `mechanismlab.core.artifacts`
- Generic artifact-gated reports: `mechanismlab.reports.claim_report`
- Backend manifests: `mechanismlab.backends`
- Local tracker/store: `mechanismlab.trackers`, `mechanismlab.stores`
- SELF-GROUND adapter: `self_ground.mechanismlab_adapter`

## SELF-GROUND-Specific For Now

- negation pair/task generation,
- token-contrast task validation,
- SAE semantic compatibility details,
- density-matched control implementation for current SAE rankings,
- detailed Phase 3 mechanism report,
- decoded SAE intervention orchestration.

## Future Framework Locations

- Claim ledger helpers may move under `mechanismlab.reports`.
- Artifact gates may move under `mechanismlab.core` or
  `mechanismlab.artifacts`.
- Project templates may move under `mechanismlab.projects`.
- Optional backend adapters may live under `mechanismlab.backends`.

## Not Moved Yet

`self_ground.mechanism_report` is not deleted or replaced. It remains the
detailed project-specific report until the generic MechanismLab report can
express the needed SELF-GROUND-specific evidence without losing scientific
caution.

Residual smoke diagnostics and feature-space proxy paths remain claim-ineligible.
