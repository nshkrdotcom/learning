from __future__ import annotations

import json
import re
from pathlib import Path

from test_prd_coverage import REQUIRED_SURFACE_IDS, VALID_STATUSES, _coverage
from test_prd_evidence_review import HIGH_RISK_ROW_IDS

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
EXTENSION_DEFERRED_IDS = {
    "feature_correspondence_record_extension",
    "training_dynamics_record_extension",
    "remote_job_metadata_record_extension",
}
ADVANCED_TECHNIQUE_SOURCE_SECTIONS = {
    "activation_record": {"47.1"},
    "activation_record_prd_schema": {"47.1"},
    "weight_analysis_run": {"47.2"},
    "weight_analysis_record_alias": {"47.2"},
    "weight_analysis_run_prd_schema_and_alias": {"47.2"},
    "circuit_graph": {"47.3"},
    "circuit_graph_record_alias": {"47.3"},
    "circuit_graph_prd_schema_and_alias": {"47.3"},
    "cross_model_comparison": {"47.4"},
    "cross_model_comparison_record_alias": {"47.4"},
    "cross_model_comparison_prd_schema_and_alias": {"47.4"},
}

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
    "run_repair_marks_stale_running_run_without_promoting_evidence",
    "run_resume_creates_child_run_with_parent_contract",
    "stale_running_run_indexed_interrupted_without_mutation",
    "non_completed_run_gate_blocks_candidate_support",
    "claim_proposal_uses_evidence_assessment_status_and_debt",
    "missing_external_pointer_records_non_evidence_backend",
    "crystallize_requires_concrete_fields",
    "query_questions_labels_records",
    "question_link_validates_targets",
]


def _ledger() -> dict:
    return json.loads(Path("docs/prd_completion_ledger_0430_0432.json").read_text())


def _reviewed_ids() -> set[str]:
    path = Path("docs/prd_evidence_review_0430_0432.json")
    if not path.exists():
        return set()
    return {row["id"] for row in json.loads(path.read_text(encoding="utf-8"))["rows"]}


def _coverage_by_id() -> dict[str, dict]:
    return {entry["id"]: entry for entry in _coverage()["entries"]}


def _specific_test_name_exists(row: dict) -> bool:
    stopwords = {
        "and",
        "boundary",
        "check",
        "command",
        "commands",
        "contract",
        "id",
        "ids",
        "metadata",
        "or",
        "record",
        "records",
        "row",
        "rows",
        "schema",
        "specific",
        "surface",
        "surfaces",
        "test",
        "tests",
        "the",
        "validate",
        "validation",
        "with",
    }
    tokens = {
        token
        for token in re.split(r"[_\W]+", row["id"].lower())
        if len(token) > 3 and token not in stopwords
    }
    test_names = " ".join(row.get("test_names", [])).lower()
    return bool(tokens) and any(token in test_names for token in tokens)


def _is_generic_pytest_command(command: str) -> bool:
    stripped = command.strip()
    return stripped in {"uv run pytest", "uv run pytest .", "pytest"}


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
            assert not all(path.startswith("docs/") for path in row["implementation_files"]), (
                row["id"]
            )
            assert not any(
                name.lower() in {"test_placeholder", "test_todo", "todo", "placeholder"}
                for name in row["test_names"]
            ), row["id"]
            assert "covered by milestone" not in " ".join(row["test_names"]).lower()
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


def test_completion_ledger_implemented_rows_have_specific_or_reviewed_evidence() -> None:
    reviewed_ids = _reviewed_ids()

    for row in _ledger()["rows"]:
        if row["disposition"] not in IMPLEMENTED_DISPOSITIONS:
            continue
        assert row["test_names"], row["id"]
        assert not any(name in {"test_placeholder", "test_todo"} for name in row["test_names"])
        assert not any(
            _is_generic_pytest_command(command) for command in row["verification_commands"]
        )
        assert any(not path.startswith("docs/") for path in row["implementation_files"]), row["id"]
        if row["disposition"] == "implemented_with_test":
            assert _specific_test_name_exists(row) or row["id"] in reviewed_ids, row["id"]
        if row["id"] in HIGH_RISK_ROW_IDS:
            assert row["id"] in reviewed_ids, row["id"]


def test_completion_ledger_nonimplemented_rows_are_explicitly_justified() -> None:
    coverage = _coverage_by_id()
    dispositions = {row["disposition"] for row in _ledger()["rows"]}
    coverage_statuses = {entry["status"] for entry in coverage.values()}

    if "deferred_by_prd" in coverage_statuses:
        assert "intentionally_deferred_by_prd" in dispositions
    if "intentionally_out_of_scope" in coverage_statuses:
        assert "intentionally_out_of_scope" in dispositions
    if "ambiguous_or_requires_decision" in coverage_statuses:
        assert "ambiguous_or_requires_decision" in dispositions
    if "partially_implemented" in coverage_statuses:
        assert "partially_implemented_with_remaining_gap" in dispositions

    for row in _ledger()["rows"]:
        if row["disposition"] == "partially_implemented_with_remaining_gap":
            assert row["remaining_gap"] and row["remaining_gap"] != "none", row["id"]
            assert "complete" not in row["remaining_gap"].lower()
            assert "done" not in row["remaining_gap"].lower()
            assert coverage[row["id"]]["status"] in {
                "partially_implemented",
                "ambiguous_or_requires_decision",
            }
        if row["disposition"] == "ambiguous_or_requires_decision":
            assert row["source_sections"], row["id"]
            assert "do not define" in row["remaining_gap"].lower() or "ambiguous" in row[
                "remaining_gap"
            ].lower()
        if row["disposition"] == "intentionally_deferred_by_prd":
            assert row["source_sections"], row["id"]
            assert "deferred" in (row["remaining_gap"] + row["notes"]).lower()
        if row["disposition"] == "intentionally_out_of_scope":
            assert row["source_sections"], row["id"]
            out_of_scope_text = (row["remaining_gap"] + row["notes"]).lower()
            assert "non-goal" in out_of_scope_text or "out of scope" in out_of_scope_text


def test_completion_ledger_has_closed_partial_and_ambiguous_accounting() -> None:
    rows = {row["id"]: row for row in _ledger()["rows"]}
    dispositions = {row["disposition"] for row in rows.values()}

    assert "partially_implemented_with_remaining_gap" not in dispositions
    assert "ambiguous_or_requires_decision" not in dispositions

    for row_id in {"weight_analysis_run", "record_specific_id"}:
        row = rows[row_id]
        assert row["disposition"] in IMPLEMENTED_DISPOSITIONS
        assert row["remaining_gap"] == "none"
        text = (row["notes"] + " " + row["requirement"]).lower()
        assert "record_id" in text
        assert "no separate prd-native" in text
        assert "weightanalysisrun" in text

    for row_id in {
        "feature_correspondence_record_extension",
        "training_dynamics_record_extension",
        "remote_job_metadata_record_extension",
    }:
        row = rows[row_id]
        assert row["disposition"] == "intentionally_deferred_by_prd"
        text = (row["notes"] + " " + row["remaining_gap"]).lower()
        assert "extension" in text
        assert "do not define" in text
        assert "concrete schema" in text
        assert row["implementation_files"] == ["src/mechledger/records.py"]
        assert row["test_files"] == ["tests/test_records.py"]
        assert row["test_names"] == [
            "test_prd_defined_advanced_technique_record_boundary_is_explicit"
        ]


def test_completion_ledger_advanced_technique_sections_are_precise() -> None:
    rows = {row["id"]: row for row in _ledger()["rows"]}

    for row_id, expected_sections in ADVANCED_TECHNIQUE_SOURCE_SECTIONS.items():
        assert set(rows[row_id]["source_sections"]) == expected_sections, row_id

    for row_id in EXTENSION_DEFERRED_IDS:
        row = rows[row_id]
        assert row["source_sections"] == [
            "47. Advanced Technique Schemas (no concrete schema defined)"
        ]


def test_completion_ledger_and_coverage_have_product_surface_parity() -> None:
    ledger_ids = {row["id"] for row in _ledger()["rows"]}
    coverage_ids = set(_coverage_by_id())

    for surface_id in REQUIRED_SURFACE_IDS:
        assert surface_id in ledger_ids
        assert surface_id in coverage_ids
    for entry in _coverage()["entries"]:
        if entry["status"] in {"implemented", "partially_implemented"}:
            assert entry["id"] in ledger_ids
