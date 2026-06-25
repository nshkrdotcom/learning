from __future__ import annotations

import pytest

from mechledger.core.claim_status import (
    ClaimStatus,
    IncomparableStatusError,
    SelfGroundEvidenceStatus,
    conservative_status_from_summary,
    map_self_ground_status,
    status_at_least,
)


def test_status_at_least_uses_dag_not_flat_ordinal() -> None:
    assert status_at_least(ClaimStatus.CANDIDATE_CLAIM, ClaimStatus.SINGLE_RUN_EVIDENCE)
    assert status_at_least(ClaimStatus.CAUSAL_SUPPORT, ClaimStatus.EXPLORATORY_SIGNAL)
    assert not status_at_least(ClaimStatus.SINGLE_RUN_EVIDENCE, ClaimStatus.CANDIDATE_CLAIM)


def test_incomparable_branches_raise_cli_error_equivalent() -> None:
    with pytest.raises(IncomparableStatusError):
        status_at_least(ClaimStatus.EXPLORATORY_SIGNAL, ClaimStatus.CORRELATIONAL_SUPPORT)


def test_terminal_statuses_only_match_themselves() -> None:
    assert status_at_least(ClaimStatus.CONTRADICTED, ClaimStatus.CONTRADICTED)
    assert not status_at_least(ClaimStatus.CONTRADICTED, ClaimStatus.UNSUPPORTED)
    assert not status_at_least(ClaimStatus.REPLICATED_EVIDENCE, ClaimStatus.CONTRADICTED)


def test_self_ground_status_mapping_is_explicitly_conservative() -> None:
    assert map_self_ground_status(SelfGroundEvidenceStatus.BLOCKED) == ClaimStatus.UNSUPPORTED
    assert (
        map_self_ground_status(SelfGroundEvidenceStatus.INSUFFICIENT_EVIDENCE)
        == ClaimStatus.FAILED_OR_WEAKENED
    )
    assert (
        map_self_ground_status(SelfGroundEvidenceStatus.CANDIDATE_EVIDENCE)
        == ClaimStatus.CANDIDATE_CLAIM
    )
    assert (
        map_self_ground_status(SelfGroundEvidenceStatus.STRONG_CANDIDATE_EVIDENCE)
        == ClaimStatus.CAUSAL_SUPPORT
    )


def test_conservative_status_logic_matches_e002_e003_e004_history() -> None:
    shared = {
        "engine_backend_valid": True,
        "required_artifacts_present": True,
        "all_rows_skipped": False,
        "nonfinite_rate": 0.0,
        "compatibility_passed": True,
        "diagnostic_only": False,
        "task_validation_passed": True,
        "valid_tasks": 69,
        "family_count": 3,
        "random_control_count": 3,
        "density_matched_control_count": 3,
        "has_ablate_and_amplify": True,
        "relative_norm_drift": 0.1,
        "norm_drift_warning_rate": 0.0,
        "has_skipped_rows": False,
    }

    e002 = {
        **shared,
        "candidate_checks_passed": False,
        "specificity_gap": -0.02048155466715495,
    }
    e003 = {
        **shared,
        "candidate_checks_passed": False,
        "specificity_gap": -0.09110175699427508,
    }
    e004_best = {
        **shared,
        "candidate_checks_passed": False,
        "specificity_gap": 0.13617621988490008,
    }

    assert conservative_status_from_summary(e002) == SelfGroundEvidenceStatus.INSUFFICIENT_EVIDENCE
    assert conservative_status_from_summary(e003) == SelfGroundEvidenceStatus.INSUFFICIENT_EVIDENCE
    assert (
        conservative_status_from_summary(e004_best)
        == SelfGroundEvidenceStatus.INSUFFICIENT_EVIDENCE
    )
