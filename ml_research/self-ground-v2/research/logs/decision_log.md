# Decision Log

## D001 - Extract SELF-GROUND audit kernel without reviving execution framework

```yaml
decision_id: D001
status: accepted
affected_experiments:
  - E001
  - E002
  - E003
  - E004
affected_claims:
  - C001
  - C002
  - C003
  - C004
decision_type: architecture
copilot_session_id: null
```

Decision:
MechLedger Milestone -1 is a new standalone package in `self-ground-v2/`. The
existing SELF-GROUND repo is treated as a read-only reference and dogfood
corpus. Its `work_logs/` and run artifacts are copied or summarized into
MechLedger records; they are not moved during this pass.

Reusable SELF-GROUND modules are ported only as pure policy logic:
`sae_compat.py`, `sae_metadata.py`, `task_calibration.py`,
`intervention_telemetry.py`, `sae_interventions.py` telemetry concepts, and
`mechanism_report.py` conservative status logic.

`engine_boundary.py` contributes the general boundary principle that
claim-eligible evidence must declare an external backend and that generic or
proxy-only engines cannot promote claims. Its SELF-GROUND-specific constants
are not copied as MechLedger product constants.

`src/mechanismlab/*`, deleted generic report adapters, generic tracker
protocols, and the old framework extraction attempt are not touched or
revived.

Reason:
MechLedger is a flat-file ledger, Draft Guard, run-audit, and scientific-debt
system. It must not become a model execution framework or a generic experiment
tracker. The reusable SELF-GROUND value is the already-tested audit discipline:
evidence ladders, overclaiming barriers, compatibility checks, calibration
rules, telemetry limits, and conservative claim status.

Alternatives considered:
- Refactor all of SELF-GROUND into MechLedger. Rejected because it would import
  TransformerLens/SAELens execution dependencies into the core.
- Recreate the deleted `mechanismlab/*` abstraction layer. Rejected because the
  assessments identify it as framework slop that duplicated real audit outputs
  without adding evidence.
- Build Milestone 0 CLI first. Rejected because PRD Milestone -1 requires the
  extracted logic to survive real SELF-GROUND dogfooding before product wiring.

Evidence:
- `0431_selfground_refactor.md` and `0432_selfground_refactor.md` agree that
  only audit/evidence/debt logic should be extracted.
- Actual module inspection confirms `task_calibration.py` is mostly portable
  dict/Pydantic policy logic, `intervention_telemetry.py` can be rewritten
  without NumPy, `sae_compat.py` mixes pure shape/metadata checks with heavy
  adapters, and `mechanism_report.py` contains the conservative status rules
  to validate against E001-E004 history.

Consequences:
- Core dependencies are limited to `typer`, `pydantic`, and `ruamel.yaml`;
  no console script or `typer.Typer()` exists in Milestone -1.
- Heavy libraries such as `torch`, `transformer_lens`, `sae_lens`, NumPy, and
  SciPy remain outside core.
- Milestone 0 can wrap this kernel in CLI commands only after the dogfood
  report is reviewed.
