from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

BASE_COMMIT = "b71ccf19e29bbb4ce58bc534b71452469a792601"

SOURCE_DOCS = {
    "0430_revised_v6.md",
    "0431_selfground_refactor.md",
    "0432_selfground_refactor.md",
}

REQUIRED_CATEGORIES = {
    "diagnostics",
    "typed_records",
    "coverage_completion_schema",
    "export_bundle_dashboard",
    "architecture_boundaries",
    "staged_hooks",
    "predictions",
    "external_labels",
    "run_auditor",
    "artifact_sweeping",
    "qc_proof",
}

HIGH_RISK_ROW_IDS = [
    "parser_diagnostic_contract",
    "claim_diagnostics_contract",
    "decision_diagnostics_contract",
    "research_log_diagnostics_contract",
    "experiment_spec_diagnostics_contract",
    "run_ledger_diagnostics_contract",
    "prediction_diagnostics_contract",
    "external_label_diagnostics_contract",
    "records_validate_surfaces_record_diagnostics",
    "labels_validate_surfaces_external_label_diagnostics",
    "prediction_lock_score_surface_diagnostics",
    "activation_record_prd_schema",
    "weight_analysis_run_prd_schema_and_alias",
    "record_specific_id",
    "extension_records_strict_shared_metadata",
    "ro_crate_platform_record_metadata",
    "bundle_platform_record_metadata",
    "dashboard_platform_record_metadata",
    "metadata_only_artifact_bytes_boundary",
    "import_boundary_ast",
    "dependency_boundary_pyproject",
    "execution_framework_schema_boundary",
    "boundary_phrase_documentation",
    "positive_metadata_only_test",
    "draft_check_staged",
    "index_check_staged",
    "no_stash_based_staged_isolation",
    "skip_messages_no_relevant_staged_files",
    "mechledger_run_command",
    "run_local_artifact_auto_collection",
    "no_hidden_artifact_sweeping",
    "prediction_feature_matching_from_run_evidence",
    "external_labels_metadata_by_default",
    "linked_claims_do_not_become_supported_automatically",
    "no_hosted_dashboard_server",
    "deferred_remote_sync_merge",
    "out_of_scope_general_model_execution_framework",
    "out_of_scope_citation_verification_statcheck",
    "run_repair_marks_stale_running_run_without_promoting_evidence",
    "run_resume_creates_child_run_with_parent_contract",
    "non_completed_run_gate_blocks_candidate_support",
    "claim_proposal_uses_evidence_assessment_status_and_debt",
    "missing_external_pointer_records_non_evidence_backend",
    "crystallize_requires_concrete_fields",
    "query_questions_labels_records",
]

IMPLEMENTED_DISPOSITIONS = {
    "implemented_with_test",
    "already_implemented_with_existing_test",
}

NONIMPLEMENTED_DISPOSITIONS = {
    "intentionally_deferred_by_prd",
    "intentionally_out_of_scope",
    "ambiguous_or_requires_decision",
    "partially_implemented_with_remaining_gap",
}

ALLOWED_EVIDENCE_QUALITIES = {
    "specific_test_asserts_behavior",
    "broad_test_needs_more_specific_assertion",
    "disposition_is_nonimplemented_and_justified",
    "incorrect_or_overclaimed",
}

ALLOWED_REVIEW_RESULTS = {
    "accepted",
    "corrected",
    "downgraded_disposition",
    "requires_followup",
}

VAGUE_EXPLANATION_PHRASES = {
    "covers diagnostics",
    "covers records",
    "covered by test file",
    "nearby coverage",
    "broad test",
    "test file covers this",
    "existing tests cover it",
    "asserts behavior",
}


def _review() -> dict:
    return json.loads(
        Path("docs/prd_evidence_review_0430_0432.json").read_text(encoding="utf-8")
    )


def _ledger_rows() -> dict[str, dict]:
    ledger = json.loads(
        Path("docs/prd_completion_ledger_0430_0432.json").read_text(encoding="utf-8")
    )
    return {row["id"]: row for row in ledger["rows"]}


def test_prd_evidence_review_artifacts_exist_parse_and_name_scope() -> None:
    review_path = Path("docs/prd_evidence_review_0430_0432.json")
    markdown_path = Path("docs/PRD_EVIDENCE_REVIEW_0430_0432.md")

    assert review_path.exists()
    assert markdown_path.exists()
    review = _review()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.strip()
    assert set(review) >= {
        "source_documents",
        "reviewed_commit_base",
        "review_scope",
        "rows",
        "summary",
    }
    assert set(review["source_documents"]) == SOURCE_DOCS
    assert review["reviewed_commit_base"] == BASE_COMMIT
    assert (
        review["review_scope"]["method"]
        == "risk_weighted_spot_check_plus_required_high_risk_rows"
    )
    assert review["review_scope"]["minimum_rows_reviewed"] >= 40
    assert set(review["review_scope"]["required_categories"]) >= REQUIRED_CATEGORIES


def test_prd_evidence_review_rows_are_unique_and_cover_required_high_risk_rows() -> None:
    review = _review()
    ledger_rows = _ledger_rows()
    reviewed_ids = [row["id"] for row in review["rows"]]

    assert len(reviewed_ids) >= 40
    assert len(reviewed_ids) == len(set(reviewed_ids))
    for row_id in reviewed_ids:
        assert row_id in ledger_rows
    for high_risk_id in HIGH_RISK_ROW_IDS:
        if high_risk_id in ledger_rows:
            assert reviewed_ids.count(high_risk_id) == 1, high_risk_id


def test_prd_evidence_review_rows_have_specific_evidence_or_nonimplemented_reason() -> None:
    ledger_rows = _ledger_rows()
    for row in _review()["rows"]:
        ledger = ledger_rows[row["id"]]
        assert row["ledger_disposition"] == ledger["disposition"], row["id"]
        assert row["evidence_quality"] in ALLOWED_EVIDENCE_QUALITIES, row["id"]
        assert row["review_result"] in ALLOWED_REVIEW_RESULTS, row["id"]
        assert row["category"] in REQUIRED_CATEGORIES | {"open_questions", "experiments"}

        if row["ledger_disposition"] in IMPLEMENTED_DISPOSITIONS:
            assert row["cited_tests"], row["id"]
            for cited in row["cited_tests"]:
                assert cited["file"].startswith("tests/"), row["id"]
                assert cited["test_name"].startswith("test_"), row["id"]
                explanation = cited[
                    "why_this_test_actually_covers_the_requirement"
                ].strip()
                lowered = explanation.lower()
                assert len(explanation.split()) >= 12, row["id"]
                assert not any(
                    phrase in lowered for phrase in VAGUE_EXPLANATION_PHRASES
                ), row["id"]
        elif row["ledger_disposition"] in NONIMPLEMENTED_DISPOSITIONS:
            assert row["source_documents"], row["id"]
            assert row["source_sections"], row["id"]
            assert row["remaining_gap"] and row["remaining_gap"] != "none", row["id"]
            assert row.get("nonimplemented_justification"), row["id"]
            assert "complete" in row["nonimplemented_justification"].lower(), row["id"]
        else:
            raise AssertionError(f"Unexpected disposition for {row['id']}")


def test_prd_evidence_review_quality_results_are_consistent_and_counted() -> None:
    review = _review()
    for row in review["rows"]:
        if row["evidence_quality"] == "broad_test_needs_more_specific_assertion":
            assert row["review_result"] in {"corrected", "requires_followup"}, row["id"]
        if row["evidence_quality"] == "incorrect_or_overclaimed":
            assert row["review_result"] in {
                "corrected",
                "downgraded_disposition",
                "requires_followup",
            }, row["id"]
        assert not (
            row["evidence_quality"] == "incorrect_or_overclaimed"
            and row["review_result"] == "accepted"
        ), row["id"]

    counts = Counter(row["review_result"] for row in review["rows"])
    assert review["summary"]["reviewed_rows"] == len(review["rows"])
    assert review["summary"]["accepted"] == counts["accepted"]
    assert review["summary"]["corrected"] == counts["corrected"]
    assert review["summary"]["downgraded_disposition"] == counts["downgraded_disposition"]
    assert review["summary"]["requires_followup"] == counts["requires_followup"]


def test_prd_evidence_review_reflects_record_identity_and_extension_closure() -> None:
    rows = {row["id"]: row for row in _review()["rows"]}

    record_specific = rows["record_specific_id"]
    assert record_specific["ledger_disposition"] in IMPLEMENTED_DISPOSITIONS
    assert record_specific["evidence_quality"] == "specific_test_asserts_behavior"
    assert record_specific["remaining_gap"] == "none"
    assert any(
        "record_id" in cited["why_this_test_actually_covers_the_requirement"].lower()
        and "weightanalysisrun" in cited[
            "why_this_test_actually_covers_the_requirement"
        ].lower()
        for cited in record_specific["cited_tests"]
    )

    for row_id in {
        "feature_correspondence_record_extension",
        "training_dynamics_record_extension",
        "remote_job_metadata_record_extension",
    }:
        row = rows[row_id]
        assert row["ledger_disposition"] == "intentionally_deferred_by_prd"
        assert row["evidence_quality"] == "disposition_is_nonimplemented_and_justified"
        text = (row["remaining_gap"] + " " + row.get("nonimplemented_justification", "")).lower()
        assert "extension" in text
        assert "do not define" in text
        assert "concrete schema" in text


def test_prd_evidence_review_markdown_summarizes_review_and_mentions_every_row() -> None:
    review = _review()
    markdown = Path("docs/PRD_EVIDENCE_REVIEW_0430_0432.md").read_text(encoding="utf-8")
    lowered = markdown.lower()

    assert "review method" in lowered
    assert "reviewed rows count" in lowered
    assert "categories covered" in lowered
    assert "corrected rows" in lowered
    assert "downgraded rows" in lowered
    assert "rows requiring follow-up" in lowered
    assert "broad test-file citation alone is not sufficient evidence" in lowered
    for row in review["rows"]:
        assert f"`{row['id']}`" in markdown
