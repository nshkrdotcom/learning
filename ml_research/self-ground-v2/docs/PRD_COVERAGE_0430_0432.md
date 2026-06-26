# PRD Coverage Audit for 0430-0432

Source documents:

- `0430_revised_v6.md`
- `0431_selfground_refactor.md`
- `0432_selfground_refactor.md`

This audit is backed by `docs/prd_coverage_0430_0432.json` and tests in
`tests/test_prd_coverage.py`. Status values are limited to:

- `implemented`
- `partially_implemented`
- `deferred_by_prd`
- `intentionally_out_of_scope`
- `missing`
- `ambiguous_or_requires_decision`

| ID | Section | Status | Coverage |
| --- | --- | --- | --- |
| `non_goals_execution_framework` | product-boundary | `implemented` | MechLedger remains a ledger/audit tool, not an execution framework, hosted dashboard, generic ML tracker, or AI reviewer. |
| `flat_file_canonical_storage` | 52 | `implemented` | Canonical state is flat files under `research/` plus `.mechledger/project.json`; SQLite/cache files are disposable. |
| `milestone_50_0` | 50.0 | `implemented` | Milestone -1 extraction and dogfood boundaries are present. |
| `milestone_50_1` | 50.1 | `implemented` | Draft Guard MVP, scaffold, parsers, staged checks, hooks, format, and session close are present. |
| `milestone_50_2` | 50.2 | `implemented` | Run auditor, alias cache, local run files, SDK logging, artifacts, status/next, and disposable index behavior are present. |
| `milestone_50_3` | 50.3 | `implemented` | Experiment prerequisites, claim proposal/review, stale detection, reclassification, and debt waivers are present. |
| `milestone_50_4` | 50.4 | `implemented` | Evidence gates and Tier 2 calibration/telemetry/null/paired-stat surfaces are present. |
| `milestone_50_5_local_surfaces` | 50.5 | `partially_implemented` | Local session/copilot provenance, open questions, external labels, wording reports, dashboard data, and queries are present. |
| `milestone_50_5_dashboard_server` | 50.5 | `deferred_by_prd` | Hosted dashboard server remains deferred; local JSON/query inspection is implemented instead. |
| `milestone_50_6` | 50.6 | `partially_implemented` | Long-term platform metadata validators and export surfaces exist, but they do not compute mechinterp records. |
| `activation_record_schema` | 47.1 | `implemented` | `ActivationRecord` validates the PRD fields and enums as metadata only. |
| `weight_analysis_run_schema` | 47.2 | `implemented` | `WeightAnalysisRun` validates the PRD fields; `WeightAnalysisRecord` is a compatibility alias. |
| `circuit_graph_schema` | 47.3 | `implemented` | `CircuitGraph` validates the PRD fields; `CircuitGraphRecord` is a compatibility alias. |
| `cross_model_comparison_schema` | 47.4 | `implemented` | `CrossModelComparison` validates the PRD fields; `CrossModelComparisonRecord` is a compatibility alias. |
| `extension_platform_records` | 47 | `ambiguous_or_requires_decision` | `FeatureCorrespondenceRecord`, `TrainingDynamicsRecord`, and `RemoteJobMetadataRecord` are supported as strict extension records because the available PRD text does not define full schemas. |
| `session_copilot_provenance` | 43-44 | `implemented` | Session/copilot records are non-authoritative local provenance with human review before accepted artifacts enter `research/`. |
| `local_integrity_sync_redaction_lifecycle` | 42 | `implemented` | Local sync diff/status, integrity records, redaction, pin/gc, and run bundles are implemented without remote merge. |
| `deferred_remote_sync_merge` | 51 | `intentionally_out_of_scope` | Remote sync/merge remains outside the local-first implementation. |
| `dependency_light_boundary` | 0431 | `implemented` | Core remains dependency-light; heavy model/numerical stacks are excluded. |
| `no_self_ground_execution_revival` | 0432 | `implemented` | SELF-GROUND audit/evidence logic is reused without reviving execution code. |

Important boundaries:

- MechLedger does not execute interventions or compute activations, circuits, weights, correspondences, training dynamics, or remote jobs.
- Typed optional platform records are metadata validation and export records only.
- RO-Crate is export-only interoperability metadata, not canonical state.
- Heavy ML dependencies remain outside core.
- Dashboard server, team review queues, remote merge, and LLM review/generation remain deferred or out of scope.
