# Research Log

## 2026-06-25

```yaml
entry_id: R2026-06-25-001
linked_runs:
  - legacy_e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density
  - legacy_e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density
  - legacy_e004_specificity_rescue_matrix
linked_claims:
  - C002
  - C003
  - C004
linked_decisions:
  - D001
open_questions:
  - Should the Pythia-70M-deduped SAE/hook search be retired or redesigned?
copilot_session_id: null
```

### Question

What SELF-GROUND audit logic should become MechLedger's reusable kernel?

### Context

Milestone -1 extracted claim-status, debt, compatibility, calibration,
telemetry, claim-ledger, and Draft Guard prototype logic from SELF-GROUND.

### Hypothesis

The audit/evidence logic can be reused without importing SELF-GROUND's
execution harness or the deleted generic MechanismLab layer.

### Work done

Backfilled E001-E004 claims, consolidated the reuse decision in D001, and
converted representative SELF-GROUND run history into the default run ledger.

### Result

The E002-E004 history remains negative or weakened evidence under MechLedger's
status mapping, while C001 remains single-run path evidence.

### Interpretation

The reusable core is the claim/evidence discipline, not a model execution
framework.

### Decision

D001 accepts extraction of the audit kernel and rejects execution-framework
reuse.

### Open questions

Whether to retire the current Pythia-70M-deduped SAE/hook search or redesign
the objective remains open.
