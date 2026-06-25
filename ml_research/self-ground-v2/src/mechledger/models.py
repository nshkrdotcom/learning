from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

CLAIM_STATUSES = {
    "unsupported",
    "engineering_verified",
    "single_run_evidence",
    "exploratory_signal",
    "correlational_support",
    "candidate_claim",
    "causal_support",
    "replicated_evidence",
    "failed_or_weakened",
    "contradicted",
    "retired",
}

SUPPORT_EDGES = {
    "unsupported": {"engineering_verified"},
    "engineering_verified": {
        "single_run_evidence",
        "exploratory_signal",
        "correlational_support",
    },
    "single_run_evidence": {"candidate_claim"},
    "exploratory_signal": {"candidate_claim"},
    "correlational_support": {"candidate_claim"},
    "candidate_claim": {"causal_support"},
    "causal_support": {"replicated_evidence"},
}

TERMINAL_CLAIM_STATUSES = {"failed_or_weakened", "contradicted", "retired"}
DECISION_STATUSES = {"proposed", "accepted", "superseded", "rejected"}
EXPERIMENT_STATUSES = {
    "draft",
    "planned",
    "ready",
    "running",
    "completed",
    "completed_and_reviewed",
    "blocked",
    "retired",
}
RUN_CLASSES = {
    "scratch",
    "notebook_exploration",
    "path_validation",
    "smoke_test",
    "diagnostic",
    "calibration",
    "benchmark",
    "serious_evidence_run",
    "paper_candidate",
    "replication",
    "published_result",
}
RUN_STATUSES = {"created", "running", "completed", "interrupted", "blocked", "failed", "cancelled"}
DEBT_TYPES = {
    "missing_positive_control",
    "failed_positive_control",
    "missing_baseline_calibration",
    "missing_empirical_null",
    "insufficient_null_seeds",
    "missing_paired_statistic",
    "missing_matched_controls",
    "high_norm_drift",
    "nonfinite_rows",
    "all_rows_skipped",
    "metadata_mismatch_override",
    "singleton_seed",
    "diagnostic_run_only",
    "smoke_test_only",
    "unreviewed_notebook_run",
    "redacted_supporting_evidence",
    "unjustified_threshold_default",
    "custom",
}
DEBT_SEVERITIES = {"info", "warning", "serious", "blocking"}
DEBT_STATUSES = {"open", "resolved", "waived", "superseded"}

RUN_LEDGER_COLUMNS = [
    "date",
    "run_id",
    "git_commit",
    "phase",
    "purpose",
    "hypothesis",
    "command",
    "model",
    "hook_point",
    "sae_release",
    "sae_id",
    "ranking_dir",
    "out_dir",
    "seed",
    "per_family",
    "top_k_features",
    "baseline_mode",
    "operations",
    "status",
    "blocker",
    "key_metric_1",
    "key_metric_2",
    "artifact_paths",
    "decision",
]


class IncomparableStatusError(ValueError):
    pass


def claim_status_at_least(actual: str, required: str) -> bool:
    if actual not in CLAIM_STATUSES or required not in CLAIM_STATUSES:
        raise ValueError(f"unknown claim status comparison: {actual!r}, {required!r}")
    if actual in TERMINAL_CLAIM_STATUSES or required in TERMINAL_CLAIM_STATUSES:
        return actual == required
    if actual == required:
        return True

    reachable = {required}
    frontier = [required]
    while frontier:
        current = frontier.pop()
        for next_status in SUPPORT_EDGES.get(current, set()):
            if next_status not in reachable:
                reachable.add(next_status)
                frontier.append(next_status)
    if actual in reachable:
        return True

    reverse_reachable = {actual}
    frontier = [actual]
    while frontier:
        current = frontier.pop()
        for next_status in SUPPORT_EDGES.get(current, set()):
            if next_status not in reverse_reachable:
                reverse_reachable.add(next_status)
                frontier.append(next_status)
    if required in reverse_reachable:
        return False

    raise IncomparableStatusError(f"claim status {actual!r} is incomparable with {required!r}")


@dataclass(slots=True)
class ClaimRecord:
    claim_id: str
    title: str
    status: str
    allowed: list[str]
    forbidden: list[str]
    required_caveats: list[str]
    debt_flags: list[str]
    linked_experiments: list[str]
    linked_runs: list[str]
    linked_decisions: list[str]
    raw_yaml: dict[str, Any]
    block_hash: str
    file: Path
    line: int
    scope: str | None = None
    owner: str | None = None
    updated_at: str | None = None
    tags: list[str] = field(default_factory=list)
    copilot_session_id: str | None = None


@dataclass(slots=True)
class ClaimLedger:
    path: Path
    claims: dict[str, ClaimRecord]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DecisionRecord:
    decision_id: str
    title: str
    status: str
    raw_yaml: dict[str, Any]
    file: Path
    line: int
    affected_experiments: list[str] = field(default_factory=list)
    affected_claims: list[str] = field(default_factory=list)
    decision_type: str | None = None
    copilot_session_id: str | None = None


@dataclass(slots=True)
class DecisionLog:
    path: Path
    decisions: dict[str, DecisionRecord]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExperimentSpec:
    experiment_id: str
    title: str
    status: str
    file: Path
    line: int
    raw_yaml: dict[str, Any] | None
    claim_targets: list[str] = field(default_factory=list)
    source_runs: list[str] = field(default_factory=list)
    prerequisites: list[dict[str, Any]] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    expected_artifacts: list[str] = field(default_factory=list)
    machine_warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DraftReference:
    file: Path
    line: int
    claim_id: str
    raw_tag: str
    offset: int


@dataclass(slots=True)
class DraftOverride:
    file: Path
    line: int | None
    claim_id: str
    violation_type: str
    reason: str
    raw_comment: str


@dataclass(slots=True)
class DraftViolation:
    file: Path
    line: int | None
    claim_id: str
    claim_status: str | None
    violation_type: Literal[
        "unknown_claim",
        "missing_claim_status",
        "forbidden_language",
        "missing_required_caveat",
        "unresolved_scientific_debt",
        "status_overreach",
        "stale_claim",
        "malformed_claim_tag",
    ]
    severity: Literal["info", "warning", "blocking"]
    message: str
    suggested_rewrite: str | None = None
    suppressed_by_override: bool = False


@dataclass(slots=True)
class DraftCheckResult:
    file: Path
    references: list[DraftReference]
    violations: list[DraftViolation]
    overrides: list[DraftOverride]

    @property
    def has_blocking_findings(self) -> bool:
        return any(
            violation.severity == "blocking" and not violation.suppressed_by_override
            for violation in self.violations
        )


@dataclass(slots=True)
class ArtifactRecord:
    artifact_id: str
    path: str
    original_path: str
    resolved_path: str
    artifact_type: str
    content_hash: str | None
    content_hash_status: Literal["computed", "external_unverified"]
    artifact_storage_backend: Literal["git", "dvc", "git_annex", "external"]
    byte_size: int | None
    claim_relevance: Literal["none", "diagnostic", "supporting", "contradicting", "required"]
    review_status: Literal["unannotated", "annotated", "ignored", "missing"]
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "original_path": self.original_path,
            "resolved_path": self.resolved_path,
            "artifact_type": self.artifact_type,
            "content_hash": self.content_hash,
            "content_hash_status": self.content_hash_status,
            "artifact_storage_backend": self.artifact_storage_backend,
            "byte_size": self.byte_size,
            "claim_relevance": self.claim_relevance,
            "review_status": self.review_status,
            "description": self.description,
        }


@dataclass(slots=True)
class ScientificDebtRecord:
    debt_id: str
    debt_type: str
    severity: str
    claim_id: str | None
    run_id: str | None
    experiment_id: str | None
    evidence_paths: list[str]
    message: str
    required_resolution: str | None
    status: str
    waiver_decision_id: str | None
    created_at: str
    resolved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "debt_id": self.debt_id,
            "debt_type": self.debt_type,
            "severity": self.severity,
            "claim_id": self.claim_id,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "evidence_paths": self.evidence_paths,
            "message": self.message,
            "required_resolution": self.required_resolution,
            "status": self.status,
            "waiver_decision_id": self.waiver_decision_id,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }


@dataclass(slots=True)
class ScientificDebtReport:
    report_id: str
    run_id: str
    experiment_id: str | None
    generated_at: str
    evaluated_assessments: list[str]
    debts: list[ScientificDebtRecord]
    blockers: list[ScientificDebtRecord]
    warnings: list[ScientificDebtRecord]
    threshold_sources: list[dict[str, Any]]
    clean_candidate_support: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "generated_at": self.generated_at,
            "evaluated_assessments": self.evaluated_assessments,
            "debts": [debt.to_dict() for debt in self.debts],
            "blockers": [debt.to_dict() for debt in self.blockers],
            "warnings": [debt.to_dict() for debt in self.warnings],
            "threshold_sources": self.threshold_sources,
            "clean_candidate_support": self.clean_candidate_support,
            "summary": self.summary,
        }
