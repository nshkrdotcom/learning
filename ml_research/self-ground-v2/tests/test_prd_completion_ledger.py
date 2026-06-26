from __future__ import annotations

import json
from pathlib import Path

from test_prd_coverage import REQUIRED_SURFACE_IDS, VALID_STATUSES, _coverage

IMPLEMENTED_DISPOSITIONS = {
    "implemented_with_test",
    "already_implemented_with_existing_test",
}
NON_IMPLEMENTED_DISPOSITIONS = {
    "intentionally_deferred_by_prd",
    "intentionally_out_of_scope",
    "ambiguous_or_requires_decision",
    "partially_implemented_with_remaining_gap",
}
VALID_DISPOSITIONS = IMPLEMENTED_DISPOSITIONS | NON_IMPLEMENTED_DISPOSITIONS

REQUIRED_BEHAVIOR_IDS = [
    "completion_ledger_schema",
    "activation_record_prd_schema",
    "weight_analysis_run_prd_schema_and_alias",
    "circuit_graph_prd_schema_and_alias",
    "cross_model_comparison_prd_schema_and_alias",
    "extension_records_strict_shared_metadata",
    "record_diagnostics_missing_record_type",
    "record_diagnostics_unknown_record_type",
    "record_diagnostics_invalid_enum",
    "record_diagnostics_missing_specific_id",
    "record_diagnostics_extra_forbidden_field",
    "record_diagnostics_invalid_shape",
    "record_diagnostics_missing_summary_stat",
    "record_specific_id_stability",
    "record_linked_runs_claims_decisions_artifacts",
    "records_list_show_normalized_metadata",
    "claim_diagnostics_contract",
    "decision_diagnostics_contract",
    "research_log_diagnostics_contract",
    "experiment_spec_diagnostics_contract",
    "run_ledger_diagnostics_contract",
    "prediction_diagnostics_contract",
    "external_label_diagnostics_contract",
    "index_check_surfaces_parser_diagnostics",
    "records_validate_surfaces_record_diagnostics",
    "labels_validate_surfaces_external_label_diagnostics",
    "prediction_lock_score_surface_diagnostics",
    "ro_crate_platform_record_metadata",
    "bundle_platform_record_metadata",
    "dashboard_platform_record_metadata",
    "metadata_only_artifact_bytes_boundary",
    "import_boundary_ast",
    "dependency_boundary_pyproject",
    "execution_framework_schema_boundary",
    "boundary_phrase_documentation",
    "positive_metadata_only_test",
    "documentation_corrections",
]


def _ledger() -> dict:
    return json.loads(Path("docs/prd_completion_ledger_0430_0432.json").read_text())


def test_completion_ledger_has_required_rows_once() -> None:
    ledger = _ledger()
    ids = [row["id"] for row in ledger["rows"]]

    assert len(ids) == len(set(ids))
    for required_id in REQUIRED_SURFACE_IDS + REQUIRED_BEHAVIOR_IDS:
        assert ids.count(required_id) == 1, required_id


def test_completion_ledger_rows_have_contractual_evidence() -> None:
    for row in _ledger()["rows"]:
        assert row["disposition"] in VALID_DISPOSITIONS
        assert "covered by milestone" not in row["requirement"].lower()
        assert "covered by milestone" not in row.get("notes", "").lower()
        if row["disposition"] in IMPLEMENTED_DISPOSITIONS:
            assert row["implementation_files"], row["id"]
            assert row["test_files"], row["id"]
            assert row["test_names"], row["id"]
            assert row["verification_commands"], row["id"]
        else:
            assert row["source_sections"], row["id"]
            assert row["remaining_gap"] and row["remaining_gap"] != "none", row["id"]


def test_completion_ledger_cross_links_to_coverage_and_markdown() -> None:
    ledger = _ledger()
    ledger_ids = {row["id"] for row in ledger["rows"]}
    coverage = _coverage()
    coverage_ids = {entry["id"] for entry in coverage["entries"]}
    markdown = Path("docs/PRD_COMPLETION_LEDGER_0430_0432.md").read_text()

    for surface_id in REQUIRED_SURFACE_IDS:
        assert surface_id in coverage_ids
        assert surface_id in ledger_ids
    for entry in coverage["entries"]:
        if entry["status"] in VALID_STATUSES:
            assert entry["id"] in ledger_ids
    for row in ledger["rows"]:
        assert f"`{row['id']}`" in markdown
