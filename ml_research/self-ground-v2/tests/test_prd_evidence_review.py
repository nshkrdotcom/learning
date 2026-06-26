from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

BASE_COMMIT = "4b7001cf96fd5da107a5e26afbceba033c0a74ad"

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
]

VAGUE_EXPLANATIONS = {
    "covers diagnostics",
    "covers records",
    "covered by test file",
    "nearby coverage",
    "broad test",
}


def _review() -> dict:
    return json.loads(
        Path("docs/prd_evidence_review_0430_0432.json").read_text(encoding="utf-8")
    )


def _ledger() -> dict:
    return json.loads(
        Path("docs/prd_completion_ledger_0430_0432.json").read_text(encoding="utf-8")
    )


def test_prd_evidence_review_artifacts_exist_and_parse() -> None:
    review_path = Path("docs/prd_evidence_review_0430_0432.json")
    markdown_path = Path("docs/PRD_EVIDENCE_REVIEW_0430_0432.md")

    assert review_path.exists()
    assert markdown_path.exists()
    review = _review()
    assert review["reviewed_commit_base"] == BASE_COMMIT
    assert review["review_scope"]["minimum_rows_reviewed"] >= 40
    assert len(review["rows"]) >= review["review_scope"]["minimum_rows_reviewed"]


def test_prd_evidence_review_covers_required_high_risk_rows_once() -> None:
    review = _review()
    ledger_ids = {row["id"] for row in _ledger()["rows"]}
    reviewed_ids = [row["id"] for row in review["rows"]]

    assert len(reviewed_ids) == len(set(reviewed_ids))
    for row_id in reviewed_ids:
        assert row_id in ledger_ids
    for high_risk_id in HIGH_RISK_ROW_IDS:
        if high_risk_id in ledger_ids:
            assert reviewed_ids.count(high_risk_id) == 1, high_risk_id


def test_prd_evidence_review_rows_have_specific_evidence_or_justification() -> None:
    implemented = {"implemented_with_test", "already_implemented_with_existing_test"}
    for row in _review()["rows"]:
        assert row["evidence_quality"] in {
            "specific_test_asserts_behavior",
            "broad_test_needs_more_specific_assertion",
            "disposition_is_nonimplemented_and_justified",
            "incorrect_or_overclaimed",
        }
        if row["ledger_disposition"] in implemented:
            assert row["cited_tests"], row["id"]
            for cited in row["cited_tests"]:
                assert cited["file"].startswith("tests/"), row["id"]
                assert cited["test_name"].startswith("test_"), row["id"]
                explanation = cited["why_this_test_actually_covers_the_requirement"].strip()
                assert explanation, row["id"]
                assert explanation.lower() not in VAGUE_EXPLANATIONS, row["id"]
                assert len(explanation.split()) >= 10, row["id"]
        else:
            assert row["remaining_gap"] and row["remaining_gap"] != "none", row["id"]


def test_prd_evidence_review_quality_results_are_consistent() -> None:
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

    counts = Counter(row["review_result"] for row in review["rows"])
    assert review["summary"]["reviewed_rows"] == len(review["rows"])
    assert review["summary"]["accepted"] == counts["accepted"]
    assert review["summary"]["corrected"] == counts["corrected"]
    assert review["summary"]["downgraded_disposition"] == counts["downgraded_disposition"]
    assert review["summary"]["requires_followup"] == counts["requires_followup"]


def test_prd_evidence_review_markdown_mentions_every_reviewed_row() -> None:
    markdown = Path("docs/PRD_EVIDENCE_REVIEW_0430_0432.md").read_text(encoding="utf-8")

    assert "broad test-file citation alone is not sufficient evidence" in markdown.lower()
    for row in _review()["rows"]:
        assert f"`{row['id']}`" in markdown
