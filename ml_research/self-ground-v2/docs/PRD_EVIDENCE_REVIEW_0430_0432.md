# PRD Evidence Review for 0430-0432

Reviewed base commit: `4b7001cf96fd5da107a5e26afbceba033c0a74ad`.

Method: risk-weighted spot check plus required high-risk rows. Broad test-file citation alone is not sufficient evidence; reviewed implemented rows explain why a named test asserts the named requirement.

Rows reviewed: 55.
Categories covered: diagnostics, typed_records, coverage_completion_schema, export_bundle_dashboard, architecture_boundaries, staged_hooks, predictions, external_labels, run_auditor, artifact_sweeping.

Evidence quality counts:
- `broad_test_needs_more_specific_assertion`: 4
- `disposition_is_nonimplemented_and_justified`: 4
- `specific_test_asserts_behavior`: 47

Corrected rows: `claim_diagnostics_contract`, `external_labels_metadata_by_default`, `linked_claims_do_not_become_supported_automatically`, `user_supplied_run_id_collision_failure`.
Downgraded rows: none.
Rows still requiring follow-up: none.

| ID | Category | Evidence quality | Result | Specific evidence summary |
| --- | --- | --- | --- | --- |
| `parser_diagnostic_contract` | `diagnostics` | `specific_test_asserts_behavior` | `accepted` | Constructs a malformed claim ledger, runs index --check, and asserts the CLI output preserves file path, line number, failed rule, and suggested fix. |
| `claim_diagnostics_contract` | `diagnostics` | `broad_test_needs_more_specific_assertion` | `corrected` | Creates a malformed claim heading and asserts the diagnostic includes the claim ledger path, line number, claim heading rule, and suggested fix. |
| `decision_diagnostics_contract` | `diagnostics` | `specific_test_asserts_behavior` | `accepted` | Creates duplicate D001 decision headings and asserts path, duplicate line number, object ID, duplicate rule, and suggested fix. |
| `research_log_diagnostics_contract` | `diagnostics` | `specific_test_asserts_behavior` | `accepted` | Writes a malformed research_log.md entry heading, calls parse_research_log, and asserts path, line number, rule, and suggested fix. |
| `experiment_spec_diagnostics_contract` | `diagnostics` | `specific_test_asserts_behavior` | `accepted` | Creates an ExperimentSpec missing status and asserts path, YAML line, experiment ID, required-field rule, and suggested fix. |
| `run_ledger_diagnostics_contract` | `diagnostics` | `specific_test_asserts_behavior` | `accepted` | Writes a CSV run row with invalid status and asserts path, row line number, run ID, failed status rule, and suggested fix. |
| `prediction_diagnostics_contract` | `diagnostics` | `specific_test_asserts_behavior` | `accepted` | Runs prediction lock on an invalid prediction and asserts path, prediction ID, validation rule, failed field, and suggested fix. |
| `external_label_diagnostics_contract` | `external_labels` | `specific_test_asserts_behavior` | `accepted` | Runs labels validate on malformed JSONL metadata and asserts path with line, label ID, validation rule, failed field, and suggested fix. |
| `records_validate_surfaces_record_diagnostics` | `typed_records` | `specific_test_asserts_behavior` | `accepted` | Validates a bad ActivationRecord through the records CLI and asserts path, record type, activation ID, failed enum field, rule marker, and fix. |
| `labels_validate_surfaces_external_label_diagnostics` | `external_labels` | `specific_test_asserts_behavior` | `accepted` | Uses the labels validate command and checks JSONL line number, object ID, failed linked_claims field, validation rule, and suggested fix. |
| `prediction_lock_score_surface_diagnostics` | `predictions` | `specific_test_asserts_behavior` | `accepted` | Exercises prediction lock failure and verifies the surfaced diagnostic keeps path, prediction ID, rule, failed field, and suggested fix. |
| `activation_record_prd_schema` | `typed_records` | `specific_test_asserts_behavior` | `accepted` | Validates a full ActivationRecord fixture through the CLI and records show output, including activation_id and prd_defined_typed schema status. |
| `weight_analysis_run_prd_schema_and_alias` | `typed_records` | `specific_test_asserts_behavior` | `accepted` | Validates both WeightAnalysisRun and WeightAnalysisRecord fixtures, proving the compatibility alias canonicalizes successfully. |
| `record_specific_id` | `typed_records` | `disposition_is_nonimplemented_and_justified` | `accepted` | Asserts WeightAnalysisRecord canonicalizes to WeightAnalysisRun and uses wrapper record_id as the documented specific ID fallback. |
| `extension_records_strict_shared_metadata` | `typed_records` | `specific_test_asserts_behavior` | `accepted` | Validates FeatureCorrespondenceRecord, TrainingDynamicsRecord, and RemoteJobMetadataRecord as extension records with shared metadata only. |
| `ro_crate_platform_record_metadata` | `export_bundle_dashboard` | `specific_test_asserts_behavior` | `accepted` | Exports RO-Crate twice, compares bytes, and asserts platform record @id, recordSpecificId, linked runs, linked decisions, artifact paths, and evidenceRole. |
| `bundle_platform_record_metadata` | `export_bundle_dashboard` | `specific_test_asserts_behavior` | `accepted` | Creates an ActivationRecord, exports manifest-only bundle, and asserts full platform record metadata appears without sweeping tensor bytes. |
| `dashboard_platform_record_metadata` | `export_bundle_dashboard` | `specific_test_asserts_behavior` | `accepted` | Generates dashboard data and asserts counts by record type, counts by schema status, and platform_records_are_evidence false. |
| `metadata_only_artifact_bytes_boundary` | `artifact_sweeping` | `specific_test_asserts_behavior` | `accepted` | Writes tensor bytes at a platform record path, exports with include-artifacts, and asserts the unregistered tensor path is not included in bundle files. |
| `import_boundary_ast` | `architecture_boundaries` | `specific_test_asserts_behavior` | `accepted` | AST-scans every src/mechledger module and fails on heavy ML, network, RDF, graph, and framework imports. |
| `dependency_boundary_pyproject` | `architecture_boundaries` | `specific_test_asserts_behavior` | `accepted` | Parses pyproject.toml dependencies and asserts heavy ML, network client, RDF, graph, and execution stacks are absent. |
| `execution_framework_schema_boundary` | `architecture_boundaries` | `specific_test_asserts_behavior` | `accepted` | AST-scans class and function definitions and rejects PatchSpec, backend abstractions, model loading, intervention, and platform computation names. |
| `boundary_phrase_documentation` | `architecture_boundaries` | `specific_test_asserts_behavior` | `accepted` | Reads README and usage docs and asserts they state non-execution, metadata-only records, heavy dependencies outside core, RO-Crate noncanonical, dashboard server status, and... |
| `positive_metadata_only_test` | `artifact_sweeping` | `specific_test_asserts_behavior` | `accepted` | Positive metadata-only regression proves validation/export records metadata while leaving unregistered tensor bytes out of artifact payloads. |
| `draft_check_staged` | `staged_hooks` | `specific_test_asserts_behavior` | `accepted` | Stages a bad draft, edits the worktree draft to comply, and asserts draft check --staged passes using working-tree staged dependencies. |
| `index_check_staged` | `staged_hooks` | `specific_test_asserts_behavior` | `accepted` | Stages research and project config paths and asserts index --check --staged runs validation and reports index check passed. |
| `no_stash_based_staged_isolation` | `staged_hooks` | `specific_test_asserts_behavior` | `accepted` | The staged draft test would fail under stash/blob-only isolation because the compliant worktree edit is not the staged draft blob. |
| `skip_messages_no_relevant_staged_files` | `staged_hooks` | `specific_test_asserts_behavior` | `accepted` | Stages only notes.txt and asserts both draft and index staged commands exit zero with their documented skip messages. |
| `mechledger_run_command` | `run_auditor` | `specific_test_asserts_behavior` | `accepted` | Runs mechledger run around a Python command and asserts run.json, stdout capture, artifact manifest, debt report, attach, annotate, and run-ledger append behavior. |
| `run_local_artifact_auto_collection` | `run_auditor` | `specific_test_asserts_behavior` | `accepted` | The wrapped command writes into MECHLEDGER_RUN_DIR/artifacts and the test asserts artifact_manifest.json contains auto-collected artifacts. |
| `no_hidden_artifact_sweeping` | `artifact_sweeping` | `specific_test_asserts_behavior` | `accepted` | Places unregistered tensor bytes at a declared platform path and asserts bundle file entries do not include that path. |
| `prediction_feature_matching_from_run_evidence` | `predictions` | `specific_test_asserts_behavior` | `accepted` | Scores predictions against runs where feature identity comes from event metadata, metric metadata, and run metadata, not prediction filenames. |
| `external_labels_metadata_by_default` | `external_labels` | `broad_test_needs_more_specific_assertion` | `corrected` | Imports, lists, shows, links, and exports a label while asserting the claim ledger text remains unchanged after linking. |
| `linked_claims_do_not_become_supported_automatically` | `external_labels` | `broad_test_needs_more_specific_assertion` | `corrected` | Links L001 to C001, verifies the label registry records the link, and asserts the claim ledger is byte-for-byte unchanged. |
| `no_hosted_dashboard_server` | `architecture_boundaries` | `specific_test_asserts_behavior` | `accepted` | AST-scans for web framework imports and asserts the CLI has no dashboard serve or server command registered. |
| `deferred_remote_sync_merge` | `other` | `disposition_is_nonimplemented_and_justified` | `accepted` | Remote sync merge is deferred/outside the local conflict-reporting commands. |
| `out_of_scope_general_model_execution_framework` | `architecture_boundaries` | `disposition_is_nonimplemented_and_justified` | `accepted` | General model execution and intervention frameworks are non-goals. |
| `out_of_scope_citation_verification_statcheck` | `other` | `disposition_is_nonimplemented_and_justified` | `accepted` | Citation verification and statistic recomputation belong to separate tools. |
| `no_execution_framework` | `architecture_boundaries` | `specific_test_asserts_behavior` | `accepted` | Rejects execution framework class/function names such as PatchSpec, ModelBackend, run_model, load_model, and compute_activations in src/mechledger. |
| `no_universal_abstraction_layer` | `architecture_boundaries` | `specific_test_asserts_behavior` | `accepted` | AST import scan blocks TransformerLens, SAELens, NNsight, pyvene, PyTorch, and network-client abstraction dependencies in core modules. |
| `prediction_canonical_hash_excludes_mutable_fields` | `predictions` | `specific_test_asserts_behavior` | `accepted` | Locks a prediction, mutates score/lock fields, and asserts the semantic canonical hash excludes mutable lock and score fields. |
| `prediction_modified_after_lock_detection` | `predictions` | `specific_test_asserts_behavior` | `accepted` | Changes semantic prediction content after lock and asserts lock detects modified_after_lock instead of silently relocking. |
| `prediction_force_relock_behavior` | `predictions` | `specific_test_asserts_behavior` | `accepted` | After modified-after-lock detection, reruns lock with --force and asserts the prediction can be intentionally relocked. |
| `prediction_score_output_fields` | `predictions` | `specific_test_asserts_behavior` | `accepted` | Scores a locked prediction against latest run and asserts scored_against_run_id, sign_match, relative_magnitude_match, tamper_status, and CLI output fields. |
| `labels_validate_import_list_show_link` | `external_labels` | `specific_test_asserts_behavior` | `accepted` | Exercises labels import, list, show, link, and RO-Crate export with a real JSONL label registry. |
| `no_arbitrary_artifact_sweeping` | `artifact_sweeping` | `specific_test_asserts_behavior` | `accepted` | Creates unrelated files while resolving aliases and asserts normal alias/index behavior does not sweep arbitrary project artifacts. |
| `user_supplied_run_id_collision_failure` | `run_auditor` | `broad_test_needs_more_specific_assertion` | `corrected` | Creates an existing run directory, invokes mechledger run with the same user-supplied run ID, and asserts exit code 2 plus the collision message and run ID. |
| `failed_cancelled_promotion_safety` | `run_auditor` | `specific_test_asserts_behavior` | `accepted` | Creates failed and cancelled runs and asserts reclassification cannot promote them to evidence classes. |
| `deterministic_export_output` | `export_bundle_dashboard` | `specific_test_asserts_behavior` | `accepted` | Runs RO-Crate export twice and compares ro-crate-metadata.json bytes to enforce deterministic output. |
| `no_rdf_internal_model` | `architecture_boundaries` | `specific_test_asserts_behavior` | `accepted` | AST import scan rejects rdflib and related non-core graph/RDF imports under src/mechledger. |
| `linked_runs_claims_decisions` | `typed_records` | `specific_test_asserts_behavior` | `accepted` | Shows a WeightAnalysisRecord alias and asserts linked_runs, linked_claims, and linked_decisions are normalized and exposed in records show output. |
| `artifact_paths` | `typed_records` | `specific_test_asserts_behavior` | `accepted` | Shows a WeightAnalysisRecord alias and asserts artifact_paths include result, tensor, and config artifact fields in deterministic order. |
| `record_specific_id_stability` | `typed_records` | `specific_test_asserts_behavior` | `accepted` | Exports aliases and asserts ActivationRecord uses activation_id while WeightAnalysisRecord uses the documented stable record_id fallback. |
| `record_linked_runs_claims_decisions_artifacts` | `typed_records` | `specific_test_asserts_behavior` | `accepted` | Asserts records show includes canonical type, schema status, record-specific ID, linked runs, linked claims, linked decisions, and artifact paths. |
| `documentation_corrections` | `coverage_completion_schema` | `specific_test_asserts_behavior` | `accepted` | Reads README and usage docs and asserts documentation states execution non-goals, metadata-only records, heavy-dependency boundary, RO-Crate noncanonical status, dashboard... |
