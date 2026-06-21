from __future__ import annotations

from enum import StrEnum


class ClaimStatus(StrEnum):
    PLANNED = "planned"
    BLOCKED = "blocked"
    ENGINEERING_VERIFIED = "engineering_verified"
    SINGLE_RUN_EFFECT = "single_run_effect"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CANDIDATE_EVIDENCE = "candidate_evidence"
    REPLICATED_EVIDENCE = "replicated_evidence"
    WEAKENED = "weakened"
    FALSIFIED = "falsified"
    DEPRECATED = "deprecated"
