from __future__ import annotations

import textwrap

from mechledger.draft_guard import check_draft_file
from mechledger.parsers import parse_claim_ledger


def _ledger(tmp_path):
    path = tmp_path / "claim_ledger.md"
    path.write_text(
        textwrap.dedent(
            """
            ### C003 - Selected SAE features increase target contrast

            ```yaml
            claim_id: C003
            status: single_run_evidence
            allowed:
              - observed
              - preliminary
            forbidden:
              - proves that
              - is the mechanism
              - identifies the mechanism
            required_caveats:
              - single run
            debt_flags:
              - missing_empirical_null
              - singleton_seed
            ```
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return parse_claim_ledger(path)


def test_draft_guard_blocks_forbidden_language_in_sentence_window(tmp_path):
    draft = tmp_path / "draft.md"
    draft.write_text(
        "The ablation proves that selected features explain the behavior. [CLAIM:C003]\n",
        encoding="utf-8",
    )

    result = check_draft_file(draft, _ledger(tmp_path))

    assert result.has_blocking_findings
    assert [violation.violation_type for violation in result.violations] == [
        "forbidden_language",
        "missing_required_caveat",
        "unresolved_scientific_debt",
    ]
    assert result.violations[0].severity == "blocking"


def test_draft_guard_supports_markdown_latex_and_comment_claim_tags(tmp_path):
    draft = tmp_path / "draft.tex"
    draft.write_text(
        textwrap.dedent(
            r"""
            This single run observed a local effect. \claim{C003}
            This single run observed a local effect. <!-- CLAIM:C003 -->
            """
        ).lstrip(),
        encoding="utf-8",
    )

    result = check_draft_file(draft, _ledger(tmp_path))

    assert [reference.claim_id for reference in result.references] == ["C003", "C003"]
    assert not any(v.violation_type == "unknown_claim" for v in result.violations)


def test_draft_guard_records_visible_inline_override_without_hiding_it(tmp_path):
    draft = tmp_path / "draft.md"
    draft.write_text(
        textwrap.dedent(
            """
            This single run proves that the scoring harness is wired. [CLAIM:C003]
            <!-- mechledger-disable forbidden_language: reviewed in D013 -->
            """
        ).lstrip(),
        encoding="utf-8",
    )

    result = check_draft_file(draft, _ledger(tmp_path))

    assert result.overrides[0].violation_type == "forbidden_language"
    assert result.overrides[0].reason == "reviewed in D013"
    forbidden = [
        violation
        for violation in result.violations
        if violation.violation_type == "forbidden_language"
    ][0]
    assert forbidden.suppressed_by_override
    assert not result.has_blocking_findings


def test_draft_guard_reports_unknown_and_malformed_tags(tmp_path):
    draft = tmp_path / "draft.md"
    draft.write_text(
        "Valid syntax with bad ID [CLAIM:C999]. Malformed [CLAIM:].\n",
        encoding="utf-8",
    )

    result = check_draft_file(draft, _ledger(tmp_path))

    assert [violation.violation_type for violation in result.violations] == [
        "unknown_claim",
        "malformed_claim_tag",
    ]
    assert result.has_blocking_findings
