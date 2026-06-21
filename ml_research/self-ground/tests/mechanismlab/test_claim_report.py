from __future__ import annotations

from mechanismlab.core import ArtifactContract, ClaimSpec, EvidenceThresholds, ExperimentSpec
from mechanismlab.core.status import ClaimStatus
from mechanismlab.reports import build_claim_report


def _claim() -> ClaimSpec:
    return ClaimSpec(
        claim_id="claim.generic",
        claim_type="unit",
        title="Generic claim",
        claim_text="A generic claim.",
    )


def _experiment() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="experiment.generic",
        claim_id="claim.generic",
        hypothesis="generic hypothesis",
        required_artifacts=["artifact.json"],
    )


def test_missing_artifacts_block_report(tmp_path) -> None:
    report = build_claim_report(
        run_dir=tmp_path,
        claim=_claim(),
        experiment=_experiment(),
        artifact_contract=ArtifactContract(required=["artifact.json"]),
        evidence_payload={"effect_abs": 1.0},
    )

    assert report.status == ClaimStatus.BLOCKED
    assert "missing required artifacts" in report.blocker_reason


def test_compatibility_failure_blocks_report(tmp_path) -> None:
    (tmp_path / "artifact.json").write_text("{}")

    report = build_claim_report(
        run_dir=tmp_path,
        artifact_contract=ArtifactContract(required=["artifact.json"]),
        evidence_payload={"compatibility": {"compatible": False}, "effect_abs": 1.0},
    )

    assert report.status == ClaimStatus.BLOCKED
    assert report.blocker_reason == "compatibility failed"


def test_nonzero_effect_without_controls_is_single_run_effect(tmp_path) -> None:
    (tmp_path / "artifact.json").write_text("{}")

    report = build_claim_report(
        run_dir=tmp_path,
        artifact_contract=ArtifactContract(required=["artifact.json"]),
        evidence_payload={
            "compatibility": {"compatible": True},
            "effect_abs": 0.5,
            "n_tasks": 2,
            "n_controls": 0,
        },
    )

    assert report.status == ClaimStatus.SINGLE_RUN_EFFECT


def test_controls_failing_threshold_are_insufficient(tmp_path) -> None:
    (tmp_path / "artifact.json").write_text("{}")

    report = build_claim_report(
        run_dir=tmp_path,
        artifact_contract=ArtifactContract(required=["artifact.json"]),
        evidence_payload={
            "compatibility": {"compatible": True},
            "effect_abs": 0.5,
            "n_tasks": 2,
            "n_controls": 1,
            "controls_passed": False,
        },
    )

    assert report.status == ClaimStatus.INSUFFICIENT_EVIDENCE


def test_passing_generic_control_criteria_are_candidate(tmp_path) -> None:
    (tmp_path / "artifact.json").write_text("{}")

    report = build_claim_report(
        run_dir=tmp_path,
        artifact_contract=ArtifactContract(required=["artifact.json"]),
        evidence_payload={
            "compatibility": {"compatible": True},
            "effect_abs": 0.5,
            "n_tasks": 2,
            "n_controls": 1,
            "controls_passed": True,
        },
    )

    assert report.status == ClaimStatus.CANDIDATE_EVIDENCE


def test_replication_count_can_promote_replicated_evidence(tmp_path) -> None:
    (tmp_path / "artifact.json").write_text("{}")

    report = build_claim_report(
        run_dir=tmp_path,
        artifact_contract=ArtifactContract(required=["artifact.json"]),
        thresholds=EvidenceThresholds(min_replications_for_replicated=2),
        evidence_payload={
            "compatibility": {"compatible": True},
            "effect_abs": 0.5,
            "n_tasks": 2,
            "n_controls": 1,
            "controls_passed": True,
            "replications": 2,
        },
    )

    assert report.status == ClaimStatus.REPLICATED_EVIDENCE
