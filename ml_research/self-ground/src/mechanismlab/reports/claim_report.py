from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mechanismlab.core.artifacts import ArtifactContract, write_model
from mechanismlab.core.claims import ClaimSpec
from mechanismlab.core.evidence import EvidenceThresholds
from mechanismlab.core.experiments import ExperimentSpec
from mechanismlab.core.runs import RunManifest
from mechanismlab.core.status import ClaimStatus


class ArtifactStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    required: bool
    present: bool
    path: str | None = None
    reason: str | None = None


class EvidenceSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | int | bool | str | None
    passed: bool | None = None
    threshold: float | int | bool | str | None = None
    details: str = ""


class ClaimReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "mechanismlab.claim_report.v1"
    claim_id: str | None
    experiment_id: str | None
    run_id: str | None
    project: str | None
    status: ClaimStatus
    recommended_claim: str
    blocker_reason: str | None = None
    artifact_status: list[ArtifactStatus]
    evidence_signals: list[EvidenceSignal] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _manifest(run_dir: Path) -> RunManifest | None:
    path = run_dir / "mechanismlab_run_manifest.json"
    if not path.exists():
        return None
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _artifact_status(run_dir: Path, contract: ArtifactContract | None) -> list[ArtifactStatus]:
    if contract is None:
        return []
    statuses: list[ArtifactStatus] = []
    for name in contract.required:
        path = run_dir / name
        statuses.append(
            ArtifactStatus(
                name=name,
                required=True,
                present=path.exists(),
                path=str(path) if path.exists() else None,
                reason=None if path.exists() else "missing required artifact",
            )
        )
    for name in contract.optional:
        path = run_dir / name
        statuses.append(
            ArtifactStatus(
                name=name,
                required=False,
                present=path.exists(),
                path=str(path) if path.exists() else None,
            )
        )
    return statuses


def _finite_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _evidence_signal(
    name: str,
    value: float | int | bool | str | None,
    *,
    passed: bool | None = None,
    threshold: float | int | bool | str | None = None,
    details: str = "",
) -> EvidenceSignal:
    return EvidenceSignal(
        name=name,
        value=value,
        passed=passed,
        threshold=threshold,
        details=details,
    )


def _status_from_evidence(
    *,
    artifact_status: list[ArtifactStatus],
    thresholds: EvidenceThresholds,
    payload: dict[str, Any],
) -> tuple[ClaimStatus, str | None, list[EvidenceSignal], list[str]]:
    missing = [item.name for item in artifact_status if item.required and not item.present]
    signals: list[EvidenceSignal] = []
    limitations: list[str] = []
    if missing:
        return (
            ClaimStatus.BLOCKED,
            f"missing required artifacts: {', '.join(missing)}",
            signals,
            limitations,
        )
    if bool(payload.get("falsified")):
        return ClaimStatus.FALSIFIED, None, signals, limitations
    if bool(payload.get("weakened")):
        return ClaimStatus.WEAKENED, None, signals, limitations

    compatibility = payload.get("compatibility") or {}
    if thresholds.require_compatibility and compatibility.get("compatible") is False:
        return ClaimStatus.BLOCKED, "compatibility failed", signals, limitations

    effect_abs = _finite_float(payload.get("effect_abs", payload.get("effect", 0.0)))
    n_tasks = int(_finite_float(payload.get("n_tasks", 0)))
    n_controls = int(_finite_float(payload.get("n_controls", 0)))
    controls_passed = bool(payload.get("controls_passed", n_controls > 0))
    warning_rate = payload.get("warning_rate")
    warning_rate_value = None if warning_rate is None else _finite_float(warning_rate)
    replications = int(_finite_float(payload.get("replications", 0)))

    effect_passed = effect_abs >= thresholds.min_effect_abs
    tasks_passed = n_tasks >= thresholds.min_tasks_for_candidate
    controls_count_passed = n_controls >= thresholds.min_controls_for_candidate
    warning_passed = (
        True
        if thresholds.max_warning_rate_for_candidate is None or warning_rate_value is None
        else warning_rate_value <= thresholds.max_warning_rate_for_candidate
    )
    signals.extend(
        [
            _evidence_signal(
                "effect_abs",
                effect_abs,
                passed=effect_passed,
                threshold=thresholds.min_effect_abs,
                details="Absolute effect magnitude.",
            ),
            _evidence_signal(
                "n_tasks",
                n_tasks,
                passed=tasks_passed,
                threshold=thresholds.min_tasks_for_candidate,
                details="Evaluated task count.",
            ),
            _evidence_signal(
                "n_controls",
                n_controls,
                passed=controls_count_passed,
                threshold=thresholds.min_controls_for_candidate,
                details="Control set count.",
            ),
            _evidence_signal(
                "controls_passed",
                controls_passed,
                passed=controls_passed,
                threshold=True,
                details="Project-supplied control check.",
            ),
            _evidence_signal(
                "warning_rate",
                warning_rate_value,
                passed=warning_passed,
                threshold=thresholds.max_warning_rate_for_candidate,
                details="Warning rate if supplied by the project evaluator.",
            ),
        ]
    )

    if not effect_passed:
        return ClaimStatus.INSUFFICIENT_EVIDENCE, None, signals, limitations
    if thresholds.require_controls_for_candidate and n_controls == 0:
        limitations.append("Nonzero effect is present without control evidence.")
        return ClaimStatus.SINGLE_RUN_EFFECT, None, signals, limitations
    if not (tasks_passed and controls_count_passed and controls_passed and warning_passed):
        return ClaimStatus.INSUFFICIENT_EVIDENCE, None, signals, limitations
    if replications >= thresholds.min_replications_for_replicated:
        signals.append(
            _evidence_signal(
                "replications",
                replications,
                passed=True,
                threshold=thresholds.min_replications_for_replicated,
                details="Independent replication count.",
            )
        )
        return ClaimStatus.REPLICATED_EVIDENCE, None, signals, limitations
    return ClaimStatus.CANDIDATE_EVIDENCE, None, signals, limitations


def _recommended_claim(status: ClaimStatus) -> str:
    if status == ClaimStatus.BLOCKED:
        return "No claim is supported because required evidence gates are blocked."
    if status == ClaimStatus.SINGLE_RUN_EFFECT:
        return (
            "A single-run effect is present, but controls are insufficient for "
            "candidate evidence."
        )
    if status == ClaimStatus.CANDIDATE_EVIDENCE:
        return "The generic artifact gates support candidate evidence under configured thresholds."
    if status == ClaimStatus.REPLICATED_EVIDENCE:
        return "The generic artifact gates support replicated evidence under configured thresholds."
    if status == ClaimStatus.FALSIFIED:
        return "The supplied evidence marks the claim as falsified."
    if status == ClaimStatus.WEAKENED:
        return "The supplied evidence weakens the claim."
    return "The run does not support promotion beyond insufficient evidence."


def build_claim_report(
    *,
    run_dir: Path,
    claim: ClaimSpec | None = None,
    experiment: ExperimentSpec | None = None,
    artifact_contract: ArtifactContract | None = None,
    thresholds: EvidenceThresholds | None = None,
    evidence_payload: dict[str, Any] | None = None,
) -> ClaimReport:
    run_path = Path(run_dir)
    payload = evidence_payload or {}
    threshold_config = thresholds or EvidenceThresholds()
    manifest = _manifest(run_path)
    artifact_status = _artifact_status(run_path, artifact_contract)
    status, blocker_reason, signals, limitations = _status_from_evidence(
        artifact_status=artifact_status,
        thresholds=threshold_config,
        payload=payload,
    )
    unsupported = list(payload.get("unsupported_claims", []))
    experiment_id = (
        experiment.experiment_id
        if experiment
        else (manifest.experiment_id if manifest else None)
    )
    return ClaimReport(
        claim_id=claim.claim_id if claim else (manifest.claim_id if manifest else None),
        experiment_id=experiment_id,
        run_id=manifest.run_id if manifest else str(run_path.name),
        project=claim.project if claim else (manifest.project if manifest else None),
        status=status,
        recommended_claim=_recommended_claim(status),
        blocker_reason=blocker_reason,
        artifact_status=artifact_status,
        evidence_signals=signals,
        limitations=limitations + list(payload.get("limitations", [])),
        unsupported_claims=unsupported,
        metadata={
            "evidence_payload": payload,
            "thresholds": threshold_config.model_dump(mode="json"),
        },
    )


def write_claim_report_markdown(report: ClaimReport, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MechanismLab Claim Report",
        "",
        f"- claim id: `{report.claim_id}`",
        f"- experiment id: `{report.experiment_id}`",
        f"- run id: `{report.run_id}`",
        f"- project: `{report.project}`",
        f"- status: `{report.status.value}`",
        f"- recommended claim: {report.recommended_claim}",
    ]
    if report.blocker_reason:
        lines.extend(["", "## Blocker", "", report.blocker_reason])
    lines.extend(["", "## Artifact Status", ""])
    for item in report.artifact_status:
        lines.append(
            f"- {item.name}: required=`{item.required}` present=`{item.present}`"
            + (f" reason={item.reason}" if item.reason else "")
        )
    lines.extend(["", "## Evidence Signals", ""])
    for signal in report.evidence_signals:
        lines.append(
            f"- {signal.name}: value=`{signal.value}` passed=`{signal.passed}` "
            f"threshold=`{signal.threshold}`"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend([f"- {item}" for item in report.limitations] or ["not available"])
    lines.extend(["", "## Unsupported Claims", ""])
    lines.extend([f"- {item}" for item in report.unsupported_claims] or ["not available"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_claim_report(
    report: ClaimReport,
    json_path: Path,
    markdown_path: Path | None = None,
) -> None:
    write_model(report, json_path)
    if markdown_path is not None:
        write_claim_report_markdown(report, markdown_path)
