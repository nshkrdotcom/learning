from __future__ import annotations

import json
from pathlib import Path

VALID_STATUSES = {
    "implemented",
    "partially_implemented",
    "deferred_by_prd",
    "intentionally_out_of_scope",
    "missing",
    "ambiguous_or_requires_decision",
}

REQUIRED_SURFACE_IDS = [
    "no_execution_framework",
    "no_universal_abstraction_layer",
    "no_generic_ml_experiment_tracker",
    "no_hosted_dashboard",
    "no_ai_reviewer",
    "dependency_light_core",
    "heavy_ml_dependencies_outside_core",
    "no_self_ground_execution_revival",
    "no_deleted_mechanismlab_abstraction_revival",
    "flat_file_canonical_state",
    "mechledger_project_json",
    "committed_research_audit_trail",
    "ignored_local_mechledger_runs",
    "disposable_sqlite_cache",
    "read_only_cache_fallback_behavior",
    "no_sqlite_as_canonical_state",
    "no_remote_merge_sync_canonical_state",
    "mechledger_init",
    "research_scaffold_generation",
    "gitignore_block",
    "claim_ledger_template",
    "decision_log_template",
    "research_log_template",
    "run_ledger_csv_template",
    "experiment_spec_template",
    "literature_paper_portfolio_scaffold",
    "claim_ledger_parser",
    "decision_log_parser",
    "research_log_parser",
    "experiment_spec_parser",
    "run_ledger_parser",
    "external_label_parser_validator",
    "prediction_record_parser_validator",
    "optional_platform_record_parser_validator",
    "parser_diagnostic_contract",
    "claim_heading_grammar",
    "claim_yaml_required_fields",
    "claim_yaml_optional_fields",
    "claim_status_dag",
    "terminal_negative_statuses",
    "claim_block_hash_canonicalization",
    "claim_proposal_generation",
    "claim_review",
    "stale_claim_proposal_detection",
    "forced_stale_review_handling",
    "claim_language_report",
    "markdown_claim_tags",
    "latex_claim_tags",
    "html_comment_claim_tags",
    "unknown_claim_ids",
    "malformed_tags",
    "forbidden_language",
    "required_caveats",
    "unresolved_scientific_debt_surfacing",
    "inline_override_handling",
    "staged_mode_behavior",
    "deterministic_draft_suggestions",
    "pre_commit_config_install",
    "direct_git_hook_fallback",
    "no_mechledger_hooks_pre_commit_dependency",
    "draft_check_staged",
    "index_check_staged",
    "no_stash_based_staged_isolation",
    "skip_messages_no_relevant_staged_files",
    "mechledger_run_command",
    "generated_run_ids",
    "user_supplied_run_id_collision_failure",
    "run_directory_creation",
    "heartbeat_file",
    "command_capture",
    "stdout_stderr_capture",
    "git_state_capture",
    "environment_allowlist_redaction",
    "resource_usage_capture",
    "run_local_artifact_auto_collection",
    "run_ledger_row_proposal",
    "claim_update_proposal",
    "run_class_handling",
    "failed_cancelled_promotion_safety",
    "alias_cache_file",
    "unique_prefix_aliases",
    "experiment_slug_aliases",
    "latest_alias",
    "latest_n_alias",
    "local_sequence_aliases",
    "alias_ambiguity_handling",
    "malformed_alias_cache_handling",
    "missing_alias_cache_fallback_behavior",
    "explicit_artifact_registration",
    "artifact_run_local_auto_collection",
    "artifact_annotation",
    "artifact_manifest_fields",
    "claim_relevance_states",
    "review_status",
    "no_arbitrary_artifact_sweeping",
    "large_artifact_backend_delegation",
    "redacted_artifact_placeholders",
    "baseline_calibration_assessment",
    "positive_control_assessment",
    "empirical_null_assessment",
    "paired_statistic_assessment",
    "typed_control_conditions",
    "seed_sensitivity",
    "telemetry",
    "norm_drift",
    "nonfinite_rows",
    "skipped_rows",
    "candidate_claim_assessment",
    "debt_report_generation",
    "debt_waiver",
    "blocking_vs_non_blocking_debt",
    "draft_guard_debt_awareness",
    "gate_check",
    "calibration_check",
    "telemetry_check",
    "null_run_plan",
    "null_run_register",
    "stats_paired_test_register",
    "threshold_source_reporting",
    "scientific_debt_regeneration",
    "prediction_lock",
    "prediction_score",
    "prediction_canonical_hash_excludes_mutable_fields",
    "prediction_modified_after_lock_detection",
    "prediction_force_relock_behavior",
    "prediction_feature_matching_from_run_evidence",
    "prediction_score_output_fields",
    "experiment_spec_validation",
    "experiment_prerequisites",
    "mechledger_next",
    "ready_blocked_debt_warning_classification",
    "experiment_crystallize",
    "source_runs_preserved",
    "formal_spec_after_exploration_honesty",
    "decision_records",
    "decision_statuses",
    "decision_required_for_debt_waiver",
    "decision_required_for_reclassification",
    "decision_new_from_diff",
    "decision_new_from_declared_surfaces",
    "implicit_surfaces_refused",
    "debt_waiver_remains_visible",
    "appendix_export",
    "bundle_export",
    "manifest_only_bundle",
    "artifact_byte_inclusion_rules",
    "ro_crate_export",
    "deterministic_export_output",
    "ro_crate_export_only_boundary",
    "no_rdf_internal_model",
    "no_hidden_artifact_sweeping",
    "session_start_note_attach_close_list_show",
    "session_review_accept_reject",
    "copilot_list_show_review",
    "sidecar_provenance_file",
    "no_automatic_canonical_mutation_without_review",
    "no_ai_review_truth_verification",
    "questions_list_add_show_resolve",
    "accepted_decision_required_for_resolution",
    "next_surfaces_open_questions",
    "questions_not_blockers_unless_policy_gated",
    "labels_validate_import_list_show_link",
    "external_labels_metadata_by_default",
    "source_attribution_fields",
    "linked_claims_do_not_become_supported_automatically",
    "dashboard_json",
    "query_claims",
    "query_runs",
    "query_debt",
    "query_artifacts",
    "query_decisions",
    "query_experiments",
    "typed_platform_record_counts",
    "records_list_show_inspection_path",
    "no_hosted_dashboard_server",
    "sync_status",
    "sync_diff",
    "no_merge_behavior",
    "integrity_check",
    "integrity_resolve",
    "prediction_edit_tamper_records",
    "claim_proposal_staleness_records",
    "artifact_modification_records",
    "decision_status_staleness_records",
    "run_ledger_proposal_records",
    "redact_run",
    "redact_artifact",
    "redaction_debt",
    "pin_run",
    "gc_dry_run",
    "gc_archive_behavior",
    "per_run_bundle",
    "activation_record",
    "weight_analysis_run",
    "weight_analysis_record_alias",
    "circuit_graph",
    "circuit_graph_record_alias",
    "cross_model_comparison",
    "cross_model_comparison_record_alias",
    "feature_correspondence_record_extension",
    "training_dynamics_record_extension",
    "remote_job_metadata_record_extension",
    "schema_status",
    "record_specific_id",
    "linked_runs_claims_decisions",
    "artifact_paths",
    "metadata_only_boundary",
    "no_platform_record_computation",
    "run_repair_marks_stale_running_run_without_promoting_evidence",
    "run_resume_creates_child_run_with_parent_contract",
    "stale_running_run_indexed_interrupted_without_mutation",
    "non_completed_run_gate_blocks_candidate_support",
    "claim_proposal_uses_evidence_assessment_status_and_debt",
    "missing_external_pointer_records_non_evidence_backend",
    "crystallize_requires_concrete_fields",
    "query_questions_labels_records",
    "question_link_validates_targets",
    "deferred_hosted_dashboard_server",
    "deferred_team_review_queues",
    "deferred_remote_sync_merge",
    "deferred_llm_reviewer_generator",
    "out_of_scope_general_model_execution_framework",
    "out_of_scope_long_term_platform_computation",
    "out_of_scope_large_file_versioning_implementation",
    "out_of_scope_citation_verification_statcheck",
]

REQUIRED_MILESTONE_SECTIONS = {"50.0", "50.1", "50.2", "50.3", "50.4", "50.5", "50.6"}


def _coverage() -> dict:
    return json.loads(Path("docs/prd_coverage_0430_0432.json").read_text(encoding="utf-8"))


def test_prd_coverage_names_source_documents() -> None:
    coverage = _coverage()
    source_documents = set(coverage["source_documents"])

    assert "0430_revised_v6.md" in source_documents
    assert "0431_selfground_refactor.md" in source_documents
    assert "0432_selfground_refactor.md" in source_documents


def test_prd_coverage_status_values_are_valid() -> None:
    coverage = _coverage()

    assert coverage["entries"]
    assert {entry["status"] for entry in coverage["entries"]} <= VALID_STATUSES


def test_prd_coverage_has_milestone_50_entries() -> None:
    coverage = _coverage()
    sections = {entry["doc_section"] for entry in coverage["entries"]}

    for section in REQUIRED_MILESTONE_SECTIONS:
        assert section in sections


def test_prd_coverage_explicitly_marks_deferred_features() -> None:
    coverage = _coverage()

    assert any(entry["status"] == "deferred_by_prd" for entry in coverage["entries"])
    assert any(
        "dashboard server" in entry["title"].lower()
        and entry["status"] == "deferred_by_prd"
        for entry in coverage["entries"]
    )


def test_prd_coverage_entries_have_required_surface_schema() -> None:
    coverage = _coverage()

    for entry in coverage["entries"]:
        assert set(entry) >= {
            "id",
            "title",
            "source_document",
            "doc_section",
            "status",
            "implementation_files",
            "test_files",
            "evidence_notes",
            "remaining_gap",
        }
        assert entry["id"]
        assert entry["title"]
        assert entry["source_document"] in {
            "0430_revised_v6.md",
            "0431_selfground_refactor.md",
            "0432_selfground_refactor.md",
        }
        assert entry["status"] in VALID_STATUSES
        if entry["status"] in {"implemented", "partially_implemented"}:
            assert entry["implementation_files"], entry["id"]
            assert entry["test_files"], entry["id"]
            assert any(not path.startswith("docs/") for path in entry["implementation_files"]), (
                entry["id"]
            )
        if entry["status"] in {
            "partially_implemented",
            "missing",
            "ambiguous_or_requires_decision",
            "deferred_by_prd",
            "intentionally_out_of_scope",
        }:
            assert entry["remaining_gap"], entry["id"]


def test_prd_coverage_has_no_duplicate_ids_and_all_required_surfaces() -> None:
    coverage = _coverage()
    ids = [entry["id"] for entry in coverage["entries"]]

    assert len(ids) == len(set(ids))
    for required_id in REQUIRED_SURFACE_IDS:
        assert ids.count(required_id) == 1, required_id


def test_prd_coverage_uses_concrete_titles_not_milestone_rollups() -> None:
    coverage = _coverage()

    for entry in coverage["entries"]:
        title = entry["title"].strip().lower()
        assert title not in {
            "milestone 0 draft guard mvp",
            "milestone 1 flat-file run auditor",
            "milestone 2 experiment and claim workflow",
            "milestone 3 mechinterp evidence assessment",
            "milestone 4 assistant and local dashboard-adjacent surfaces",
            "milestone 5 long-term mechinterp platform metadata",
        }


def test_prd_coverage_markdown_contains_every_json_id() -> None:
    coverage = _coverage()
    markdown = Path("docs/PRD_COVERAGE_0430_0432.md").read_text(encoding="utf-8")

    for entry in coverage["entries"]:
        assert f"`{entry['id']}`" in markdown


def test_prd_coverage_has_out_of_scope_feature() -> None:
    coverage = _coverage()

    assert any(
        entry["status"] == "intentionally_out_of_scope" for entry in coverage["entries"]
    )


def test_prd_coverage_evidence_notes_are_not_generic_boilerplate() -> None:
    coverage = _coverage()
    banned_phrases = {
        "backed by deterministic flat-file implementation and tests",
    }

    for entry in coverage["entries"]:
        evidence = entry["evidence_notes"].lower()
        assert not any(phrase in evidence for phrase in banned_phrases), entry["id"]
        if entry["status"] == "implemented":
            assert entry["remaining_gap"] == "none", entry["id"]
            assert entry["test_files"], entry["id"]
            assert any(not path.startswith("docs/") for path in entry["implementation_files"]), (
                entry["id"]
            )
        if entry["status"] == "partially_implemented":
            assert entry["remaining_gap"] and entry["remaining_gap"] != "none", entry["id"]
        if entry["status"] in {
            "ambiguous_or_requires_decision",
            "deferred_by_prd",
            "intentionally_out_of_scope",
        }:
            assert not entry["implementation_files"], entry["id"]
            assert entry["remaining_gap"] and entry["remaining_gap"] != "none", entry["id"]


def test_prd_coverage_markdown_has_no_legacy_milestone_acceptance_rows() -> None:
    markdown = Path("docs/PRD_COVERAGE_0430_0432.md").read_text(encoding="utf-8")

    for legacy_id in {
        "milestone_50_1",
        "milestone_50_2",
        "milestone_50_3",
        "milestone_50_4",
    }:
        assert legacy_id not in markdown
