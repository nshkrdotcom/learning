from __future__ import annotations

import textwrap

import pytest

from mechledger.hashing import canonical_claim_hash
from mechledger.parsers import (
    LedgerParseError,
    parse_claim_ledger,
    parse_decision_log,
    parse_experiment_spec,
)


def test_claim_ledger_parses_yaml_blocks_and_hashes_order_insensitive_lists(tmp_path):
    path = tmp_path / "claim_ledger.md"
    path.write_text(
        textwrap.dedent(
            """
            ### C003 - Selected SAE features increase target contrast

            ```yaml
            claim_id: C003
            title: Selected SAE features increase target contrast
            status: single_run_evidence
            allowed:
              - preliminary
              - observed
            forbidden:
              - proves that
              - is the mechanism
            required_caveats:
              - single run
            debt_flags:
              - missing_empirical_null
            linked_experiments:
              - E001
            linked_runs:
              - 20260625T120301Z_e001_verify_intervention_path_k7p4qd
            linked_decisions:
              - D009
            copilot_session_id: null
            ```

            Evidence:
            - One finite run.
            """
        ).lstrip(),
        encoding="utf-8",
    )

    ledger = parse_claim_ledger(path)

    assert ledger.claims["C003"].status == "single_run_evidence"
    assert ledger.claims["C003"].line == 1
    assert ledger.claims["C003"].debt_flags == ["missing_empirical_null"]
    same_claim = {
        "claim_id": "C003",
        "title": "Selected SAE features increase target contrast",
        "status": "single_run_evidence",
        "allowed": ["observed", "preliminary"],
        "forbidden": ["is the mechanism", "proves that"],
        "required_caveats": ["single run"],
        "debt_flags": ["missing_empirical_null"],
        "linked_experiments": ["E001"],
        "linked_runs": ["20260625T120301Z_e001_verify_intervention_path_k7p4qd"],
        "linked_decisions": ["D009"],
    }
    assert ledger.claims["C003"].block_hash == canonical_claim_hash(same_claim)


def test_claim_ledger_rejects_missing_required_language_policy(tmp_path):
    path = tmp_path / "claim_ledger.md"
    path.write_text(
        textwrap.dedent(
            """
            ### C001 - Bad claim

            ```yaml
            claim_id: C001
            status: unsupported
            allowed: []
            ```
            """
        ).lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(LedgerParseError) as error:
        parse_claim_ledger(path)

    assert "claim.yaml.required_field" in str(error.value)
    assert "forbidden" in str(error.value)
    assert "claim_ledger.md:1 C001" in str(error.value)


def test_claim_ledger_rejects_duplicate_claim_ids(tmp_path):
    path = tmp_path / "claim_ledger.md"
    path.write_text(
        textwrap.dedent(
            """
            ### C001 - First

            ```yaml
            claim_id: C001
            status: unsupported
            allowed: []
            forbidden: []
            ```

            ### C001 - Second

            ```yaml
            claim_id: C001
            status: unsupported
            allowed: []
            forbidden: []
            ```
            """
        ).lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(LedgerParseError, match="claim.id.duplicate"):
        parse_claim_ledger(path)


def test_decision_log_parser_enforces_status_and_identity(tmp_path):
    path = tmp_path / "decision_log.md"
    path.write_text(
        textwrap.dedent(
            """
            ## D009 - Replace baseline

            ```yaml
            decision_id: D009
            status: accepted
            affected_experiments:
              - E001
            affected_claims:
              - C003
            copilot_session_id: null
            ```
            """
        ).lstrip(),
        encoding="utf-8",
    )

    log = parse_decision_log(path)

    assert log.decisions["D009"].status == "accepted"
    assert log.decisions["D009"].affected_claims == ["C003"]


def test_experiment_spec_parser_validates_required_yaml_and_headings(tmp_path):
    path = tmp_path / "E001_token_contrast.md"
    path.write_text(
        textwrap.dedent(
            """
            # E001: Phase 3 Token-Contrast Evaluation

            ```yaml
            experiment_id: E001
            status: planned
            claim_targets:
              - C003
            prerequisites:
              - type: decision_accepted
                id: D009
            config_files:
              - configs/e001.yaml
            expected_artifacts:
              - runs/e001/results.jsonl
            ```

            ## Status
            planned
            ## Research question
            TODO
            ## Hypothesis
            TODO
            ## Metrics
            specificity_gap
            ## Controls
            matched non-negation
            ## Success criterion
            positive specificity gap
            ## Failure criterion
            null result
            """
        ).lstrip(),
        encoding="utf-8",
    )

    spec = parse_experiment_spec(path)

    assert spec.experiment_id == "E001"
    assert spec.claim_targets == ["C003"]
    assert spec.prerequisites[0]["type"] == "decision_accepted"


def test_experiment_spec_warns_when_prose_prerequisites_are_not_machine_readable(tmp_path):
    path = tmp_path / "E001_token_contrast.md"
    path.write_text(
        textwrap.dedent(
            """
            # E001: Phase 3 Token-Contrast Evaluation

            ## Status
            planned
            ## Research question
            TODO
            ## Hypothesis
            TODO
            ## Metrics
            specificity_gap
            ## Controls
            matched non-negation
            ## Success criterion
            positive specificity gap
            ## Failure criterion
            null result
            ## Prerequisites
            E000 must be complete.
            """
        ).lstrip(),
        encoding="utf-8",
    )

    spec = parse_experiment_spec(path)

    assert spec.machine_warnings == [
        "Experiment E001 has prose prerequisites but no machine-readable YAML prerequisites."
    ]
