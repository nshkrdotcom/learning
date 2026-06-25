from __future__ import annotations

from enum import StrEnum


class ClaimStatus(StrEnum):
    UNSUPPORTED = "unsupported"
    ENGINEERING_VERIFIED = "engineering_verified"
    SINGLE_RUN_EVIDENCE = "single_run_evidence"
    EXPLORATORY_SIGNAL = "exploratory_signal"
    CORRELATIONAL_SUPPORT = "correlational_support"
    CANDIDATE_CLAIM = "candidate_claim"
    CAUSAL_SUPPORT = "causal_support"
    REPLICATED_EVIDENCE = "replicated_evidence"
    FAILED_OR_WEAKENED = "failed_or_weakened"
    CONTRADICTED = "contradicted"
    RETIRED = "retired"


class SelfGroundEvidenceStatus(StrEnum):
    BLOCKED = "blocked"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CANDIDATE_EVIDENCE = "candidate_evidence"
    STRONG_CANDIDATE_EVIDENCE = "strong_candidate_evidence"


TERMINAL_STATUSES = {
    ClaimStatus.FAILED_OR_WEAKENED,
    ClaimStatus.CONTRADICTED,
    ClaimStatus.RETIRED,
}

SUPPORT_DAG: dict[ClaimStatus, set[ClaimStatus]] = {
    ClaimStatus.UNSUPPORTED: {ClaimStatus.ENGINEERING_VERIFIED},
    ClaimStatus.ENGINEERING_VERIFIED: {
        ClaimStatus.SINGLE_RUN_EVIDENCE,
        ClaimStatus.EXPLORATORY_SIGNAL,
        ClaimStatus.CORRELATIONAL_SUPPORT,
    },
    ClaimStatus.SINGLE_RUN_EVIDENCE: {ClaimStatus.CANDIDATE_CLAIM},
    ClaimStatus.EXPLORATORY_SIGNAL: {ClaimStatus.CANDIDATE_CLAIM},
    ClaimStatus.CORRELATIONAL_SUPPORT: {ClaimStatus.CANDIDATE_CLAIM},
    ClaimStatus.CANDIDATE_CLAIM: {ClaimStatus.CAUSAL_SUPPORT},
    ClaimStatus.CAUSAL_SUPPORT: {ClaimStatus.REPLICATED_EVIDENCE},
}


class IncomparableStatusError(ValueError):
    """Raised when PRD §11 requires a CLI-error-equivalent comparison failure."""


def status_at_least(actual: ClaimStatus | str, required: ClaimStatus | str) -> bool:
    actual = ClaimStatus(actual)
    required = ClaimStatus(required)
    if actual in TERMINAL_STATUSES or required in TERMINAL_STATUSES:
        return actual == required
    if actual == required:
        return True
    if actual in _descendants(required):
        return True
    if required in _descendants(actual):
        return False
    raise IncomparableStatusError(
        f"claim status {actual.value!r} is incomparable with {required.value!r}"
    )


def _descendants(status: ClaimStatus) -> set[ClaimStatus]:
    seen: set[ClaimStatus] = set()
    frontier = list(SUPPORT_DAG.get(status, set()))
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(SUPPORT_DAG.get(current, set()))
    return seen


def map_self_ground_status(status: SelfGroundEvidenceStatus | str) -> ClaimStatus:
    """Conservative bridge from SELF-GROUND's four report states to PRD §11.

    SELF-GROUND `insufficient_evidence` often means evidence weakened the target
    feature claim under controls, so Milestone -1 maps it to
    `failed_or_weakened` for E002-E004 dogfood rather than silently calling it
    merely unsupported.
    """

    status = SelfGroundEvidenceStatus(status)
    return {
        SelfGroundEvidenceStatus.BLOCKED: ClaimStatus.UNSUPPORTED,
        SelfGroundEvidenceStatus.INSUFFICIENT_EVIDENCE: ClaimStatus.FAILED_OR_WEAKENED,
        SelfGroundEvidenceStatus.CANDIDATE_EVIDENCE: ClaimStatus.CANDIDATE_CLAIM,
        SelfGroundEvidenceStatus.STRONG_CANDIDATE_EVIDENCE: ClaimStatus.CAUSAL_SUPPORT,
    }[status]


def conservative_status_from_summary(summary: dict[str, object]) -> SelfGroundEvidenceStatus:
    """Pure seed of SELF-GROUND mechanism-report promotion logic.

    This accepts already-computed JSON-shaped summary fields. It does not read
    artifacts or import SELF-GROUND.
    """

    if summary.get("engine_backend_valid") is False:
        return SelfGroundEvidenceStatus.BLOCKED
    if summary.get("required_artifacts_present") is False:
        return SelfGroundEvidenceStatus.BLOCKED
    if summary.get("all_rows_skipped") is True:
        return SelfGroundEvidenceStatus.BLOCKED
    if float(summary.get("nonfinite_rate") or 0.0) > 0.0:
        return SelfGroundEvidenceStatus.BLOCKED
    if summary.get("compatibility_passed") is False and not summary.get("diagnostic_only"):
        return SelfGroundEvidenceStatus.BLOCKED
    if summary.get("task_validation_passed") is False:
        return SelfGroundEvidenceStatus.BLOCKED
    if summary.get("diagnostic_only") is True:
        return SelfGroundEvidenceStatus.INSUFFICIENT_EVIDENCE
    if not summary.get("candidate_checks_passed"):
        return SelfGroundEvidenceStatus.INSUFFICIENT_EVIDENCE
    if float(summary.get("specificity_gap") or 0.0) <= 0.0:
        return SelfGroundEvidenceStatus.INSUFFICIENT_EVIDENCE
    if summary.get("has_skipped_rows"):
        return SelfGroundEvidenceStatus.INSUFFICIENT_EVIDENCE
    if (
        int(summary.get("valid_tasks") or 0) >= 30
        and int(summary.get("family_count") or 0) >= 3
        and int(summary.get("random_control_count") or 0) >= 3
        and int(summary.get("density_matched_control_count") or 0) >= 3
        and bool(summary.get("has_ablate_and_amplify"))
        and float(summary.get("collateral_ratio") or 999.0) <= 0.5
        and float(summary.get("relative_norm_drift") or 0.0) <= 0.5
        and float(summary.get("norm_drift_warning_rate") or 0.0) == 0.0
    ):
        return SelfGroundEvidenceStatus.STRONG_CANDIDATE_EVIDENCE
    return SelfGroundEvidenceStatus.CANDIDATE_EVIDENCE
