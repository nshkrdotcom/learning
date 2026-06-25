from __future__ import annotations

import textwrap

import pytest

from mechledger.core.claim_ledger import (
    ClaimLedgerParseError,
    canonical_claim_hash,
    parse_claim_ledger,
)

BASE = {
    "claim_id": "C900",
    "title": "Hash fixture",
    "status": "single_run_evidence",
    "allowed": ["observed", "preliminary"],
    "forbidden": ["proves that", "is the mechanism"],
    "required_caveats": ["single run"],
    "debt_flags": ["missing_empirical_null"],
    "ordered_examples": ["first", "second"],
}


def test_order_insensitive_fields_and_key_order_do_not_affect_hash() -> None:
    variant = {
        "forbidden": ["is the mechanism", "proves that", "proves that"],
        "allowed": ["preliminary", "observed"],
        "ordered_examples": ["first", "second"],
        "status": "single_run_evidence",
        "title": "Hash fixture",
        "claim_id": "C900",
        "required_caveats": ["single run"],
        "debt_flags": ["missing_empirical_null"],
    }

    assert canonical_claim_hash(BASE) == canonical_claim_hash(variant)


def test_non_order_insensitive_list_order_affects_hash() -> None:
    variant = {**BASE, "ordered_examples": ["second", "first"]}

    assert canonical_claim_hash(BASE) != canonical_claim_hash(variant)


def test_yaml_quoting_and_freeform_prose_do_not_affect_hash(tmp_path) -> None:
    first = tmp_path / "claim_ledger_a.md"
    second = tmp_path / "claim_ledger_b.md"
    first.write_text(
        textwrap.dedent(
            """
            ### C900 - Hash fixture

            ```yaml
            claim_id: C900
            title: "Hash fixture"
            status: single_run_evidence
            allowed: ["observed", "preliminary"]
            forbidden:
              - "proves that"
              - "is the mechanism"
            ordered_examples:
              - first
              - second
            ```

            Freeform prose A.
            """
        ).lstrip(),
        encoding="utf-8",
    )
    second.write_text(
        textwrap.dedent(
            """
            ### C900 - Hash fixture


            ```yaml
            forbidden:
              - 'is the mechanism'
              - 'proves that'
            allowed:
              - preliminary
              - observed
            status: single_run_evidence
            title: Hash fixture
            claim_id: C900
            ordered_examples:
              - first
              - second
            ```

            Different prose B.
            """
        ).lstrip(),
        encoding="utf-8",
    )

    assert (
        parse_claim_ledger(first).claims["C900"].block_hash
        == parse_claim_ledger(second).claims["C900"].block_hash
    )


def test_claim_parser_error_includes_path_line_object_rule_and_fix(tmp_path) -> None:
    path = tmp_path / "claim_ledger.md"
    path.write_text(
        textwrap.dedent(
            """
            ### C001 - Missing forbidden

            ```yaml
            claim_id: C001
            status: unsupported
            allowed: []
            ```
            """
        ).lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ClaimLedgerParseError) as exc:
        parse_claim_ledger(path)

    message = str(exc.value)
    assert "claim_ledger.md:1 C001" in message
    assert "Rule: claim.yaml.required_field" in message
    assert "Suggested fix:" in message


def test_backfilled_real_claim_ledger_parses() -> None:
    ledger = parse_claim_ledger("research/logs/claim_ledger.md")

    assert sorted(ledger.claims) == ["C001", "C002", "C003", "C004"]
    assert ledger.claims["C004"].status == "failed_or_weakened"
