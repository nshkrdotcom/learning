from __future__ import annotations

from pathlib import Path

from mechledger.core.claim_ledger import parse_claim_ledger
from mechledger.draftguard_proto import check_markdown_text, extract_sentence_window


def test_extracts_sentence_window_only_near_claim_tag() -> None:
    text = (
        "Outside sentence proves that nothing relevant. "
        "First context. Second context. Tagged sentence [CLAIM:C004]. "
        "After one. After two. Outside again proves that nothing."
    )

    window = extract_sentence_window(text, text.index("[CLAIM:C004]"))

    assert "Tagged sentence" in window
    assert "Outside again" not in window


def test_real_good_sentence_passes_and_real_bad_sentence_flags() -> None:
    ledger = parse_claim_ledger("research/logs/claim_ledger.md")
    good = Path("tests/fixtures/real/good_sentence.md").read_text(encoding="utf-8")
    bad = Path("tests/fixtures/real/bad_sentence.md").read_text(encoding="utf-8")

    good_result = check_markdown_text(good, ledger)
    bad_result = check_markdown_text(bad, ledger)

    assert good_result.flagged == []
    assert [finding.violation_type for finding in bad_result.flagged] == ["forbidden_language"]
    assert "the negation mechanism" in bad_result.flagged[0].matched_phrase
