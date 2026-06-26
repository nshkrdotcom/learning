from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from mechledger.core.claim_ledger import ClaimRecord, parse_claim_ledger
from mechledger.core.claim_status import IncomparableStatusError, status_at_least
from mechledger.core.decision_log import DecisionRecord, parse_decision_log
from mechledger.core.experiment_spec import ExperimentSpec
from mechledger.core.run_ledger import parse_run_ledger
from mechledger.project import Project


class PrerequisiteConsequence(StrEnum):
    BLOCKING = "blocking"
    SCIENTIFIC_DEBT = "scientific_debt"
    WARNING = "warning"


class PrerequisiteFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consequence: PrerequisiteConsequence
    file: str
    line: int | None
    experiment_id: str
    prerequisite_type: str
    rule: str
    message: str
    suggested_fix: str | None = None
    input_error: bool = False

    def format(self) -> str:
        if self.consequence == PrerequisiteConsequence.SCIENTIFIC_DEBT:
            label = "SCIENTIFIC_DEBT"
        else:
            label = self.consequence.value.upper()
        location = self.file
        if self.line is not None:
            location += f":{self.line}"
        text = (
            f"{label} {location} {self.experiment_id}\n"
            f"Prerequisite: {self.prerequisite_type}\n"
            f"Rule: {self.rule}\n"
            f"{self.message}"
        )
        if self.suggested_fix:
            text += f"\nSuggested fix: {self.suggested_fix}"
        return text


class PrerequisiteEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    findings: list[PrerequisiteFinding]

    @property
    def input_errors(self) -> list[PrerequisiteFinding]:
        return [finding for finding in self.findings if finding.input_error]

    @property
    def blockers(self) -> list[PrerequisiteFinding]:
        return [
            finding
            for finding in self.findings
            if finding.consequence == PrerequisiteConsequence.BLOCKING
            and not finding.input_error
        ]

    @property
    def debt_or_warnings(self) -> list[PrerequisiteFinding]:
        return [
            finding
            for finding in self.findings
            if finding.consequence
            in {PrerequisiteConsequence.SCIENTIFIC_DEBT, PrerequisiteConsequence.WARNING}
        ]

    @property
    def is_clean(self) -> bool:
        return not self.findings


class ProjectPrerequisiteContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    decisions: dict[str, DecisionRecord]
    claims: dict[str, ClaimRecord]
    completed_experiments: set[str]
    reviewed_experiments: set[str]
    artifact_ids: set[str]
    artifact_paths: set[str]
    project: Project


def load_prerequisite_context(project: Project) -> ProjectPrerequisiteContext:
    decisions = parse_decision_log(project.root / project.config.default_decision_log).decisions
    claims = parse_claim_ledger(project.root / project.config.default_claim_ledger).claims
    run_ledger = parse_run_ledger(project.root / project.config.default_run_ledger)
    completed = {
        row.get("phase", "")
        for row in run_ledger.rows
        if row.get("status") in {"completed", "insufficient_evidence", "not_claim_run"}
    }
    reviewed = {row.get("phase", "") for row in run_ledger.rows if row.get("decision")}
    artifact_ids: set[str] = set()
    artifact_paths: set[str] = set()
    _merge_local_run_state(
        project,
        decisions,
        completed,
        reviewed,
        artifact_ids,
        artifact_paths,
    )
    return ProjectPrerequisiteContext(
        decisions=decisions,
        claims=claims,
        completed_experiments=completed,
        reviewed_experiments=reviewed,
        artifact_ids=artifact_ids,
        artifact_paths=artifact_paths,
        project=project,
    )


def evaluate_experiment_prerequisites(
    spec: ExperimentSpec,
    context: ProjectPrerequisiteContext,
) -> PrerequisiteEvaluation:
    findings: list[PrerequisiteFinding] = []
    for prereq in spec.prerequisites:
        kind = str(prereq.get("type") or "")
        consequence = _consequence(prereq)
        finding = _evaluate_one(spec, context, prereq, kind, consequence)
        if finding is not None:
            findings.append(finding)
    return PrerequisiteEvaluation(experiment_id=spec.experiment_id, findings=findings)


def _evaluate_one(
    spec: ExperimentSpec,
    context: ProjectPrerequisiteContext,
    prereq: dict[str, Any],
    kind: str,
    consequence: PrerequisiteConsequence,
) -> PrerequisiteFinding | None:
    if kind == "decision_accepted":
        decision_id = str(prereq.get("id") or "")
        decision = context.decisions.get(decision_id)
        if decision is not None and decision.status == "accepted":
            return None
        return _finding(
            spec,
            kind,
            consequence,
            "experiment.prerequisite.decision_accepted",
            f"Decision {decision_id or '<missing>'} is not accepted.",
            "add an accepted decision record or change this prerequisite.",
        )
    if kind == "experiment_completed":
        experiment_id = str(prereq.get("id") or "")
        if experiment_id in context.completed_experiments:
            return None
        return _finding(
            spec,
            kind,
            consequence,
            "experiment.prerequisite.experiment_completed",
            f"Experiment {experiment_id or '<missing>'} is not completed.",
            "append a reviewed run ledger row or remove the prerequisite.",
        )
    if kind == "experiment_completed_and_reviewed":
        experiment_id = str(prereq.get("id") or "")
        if (
            experiment_id in context.completed_experiments
            and experiment_id in context.reviewed_experiments
        ):
            return None
        return _finding(
            spec,
            kind,
            consequence,
            "experiment.prerequisite.experiment_completed_and_reviewed",
            f"Experiment {experiment_id or '<missing>'} is not completed and reviewed.",
            "complete the experiment and link an accepted decision/review.",
        )
    if kind == "claim_status_at_least":
        claim_id = str(prereq.get("id") or "")
        required = str(prereq.get("status") or "")
        claim = context.claims.get(claim_id)
        if claim is None:
            return _finding(
                spec,
                kind,
                consequence,
                "experiment.prerequisite.claim_exists",
                f"Claim {claim_id or '<missing>'} does not exist.",
                "add the claim or correct the prerequisite id.",
            )
        try:
            ok = status_at_least(claim.status, required)
        except (IncomparableStatusError, ValueError) as exc:
            return _finding(
                spec,
                kind,
                consequence,
                "experiment.prerequisite.claim_status_incomparable",
                f"Claim status comparison is invalid or incomparable: {exc}",
                "use a comparable claim-status DAG requirement.",
                input_error=True,
            )
        if ok:
            return None
        return _finding(
            spec,
            kind,
            consequence,
            "experiment.prerequisite.claim_status_at_least",
            f"Claim {claim_id} is {claim.status.value}, below required {required}.",
            "update the claim with reviewed evidence or lower the prerequisite.",
        )
    if kind == "artifact_exists":
        raw_path = str(prereq.get("path") or "")
        artifact_id = str(prereq.get("artifact_id") or "")
        if artifact_id and artifact_id in context.artifact_ids:
            return None
        if raw_path and _artifact_path_exists(context, raw_path):
            return None
        described = artifact_id or raw_path or "<missing>"
        return _finding(
            spec,
            kind,
            consequence,
            "experiment.prerequisite.artifact_exists",
            f"Required artifact does not exist: {described}.",
            "create/register the artifact or change the prerequisite path.",
        )
    return _finding(
        spec,
        kind or "<missing>",
        consequence,
        "experiment.prerequisite.unknown_type",
        f"Unknown prerequisite type: {kind or '<missing>'}.",
        "use a supported prerequisite type.",
        input_error=True,
    )


def _consequence(prereq: dict[str, Any]) -> PrerequisiteConsequence:
    raw = prereq.get("consequence") or PrerequisiteConsequence.BLOCKING.value
    try:
        return PrerequisiteConsequence(str(raw))
    except ValueError:
        return PrerequisiteConsequence.BLOCKING


def _finding(
    spec: ExperimentSpec,
    kind: str,
    consequence: PrerequisiteConsequence,
    rule: str,
    message: str,
    suggested_fix: str,
    *,
    input_error: bool = False,
) -> PrerequisiteFinding:
    return PrerequisiteFinding(
        consequence=consequence,
        file=str(Path(spec.file)),
        line=spec.line,
        experiment_id=spec.experiment_id,
        prerequisite_type=kind,
        rule=rule,
        message=message,
        suggested_fix=suggested_fix,
        input_error=input_error,
    )


def _merge_local_run_state(
    project: Project,
    decisions: dict[str, DecisionRecord],
    completed: set[str],
    reviewed: set[str],
    artifact_ids: set[str],
    artifact_paths: set[str],
) -> None:
    if not project.runs_dir.exists():
        return
    for run_dir in sorted(path for path in project.runs_dir.iterdir() if path.is_dir()):
        run_json_path = run_dir / "run.json"
        if run_json_path.exists():
            run_data = _read_json_object(run_json_path)
            experiment_id = str(run_data.get("experiment_id") or "")
            if experiment_id and run_data.get("status") == "completed":
                completed.add(experiment_id)
            if experiment_id and _has_accepted_transition(run_dir, decisions):
                reviewed.add(experiment_id)
        manifest = _read_json_object(run_dir / "artifact_manifest.json")
        for artifact in manifest.get("artifacts", []):
            artifact_id = str(artifact.get("artifact_id") or "")
            if artifact_id:
                artifact_ids.add(artifact_id)
            for key in ("original_path", "project_relative_path", "resolved_path"):
                value = artifact.get(key)
                if value:
                    artifact_paths.add(str(value))


def _has_accepted_transition(
    run_dir: Path,
    decisions: dict[str, DecisionRecord],
) -> bool:
    transition_path = run_dir / "run_class_transition.json"
    if not transition_path.exists():
        return False
    transitions = _read_json_value(transition_path)
    if isinstance(transitions, dict):
        transitions = [transitions]
    if not isinstance(transitions, list):
        return False
    for transition in transitions:
        decision = decisions.get(str(transition.get("decision_id") or ""))
        if decision is not None and decision.status == "accepted":
            return True
    return False


def _artifact_path_exists(context: ProjectPrerequisiteContext, raw_path: str) -> bool:
    project_path = context.project.root / raw_path
    if project_path.exists():
        return True
    if Path(str(project_path) + ".redacted").exists():
        return True
    resolved = str(project_path.resolve())
    return (
        raw_path in context.artifact_paths
        or resolved in context.artifact_paths
        or f"{raw_path}.redacted" in context.artifact_paths
        or f"{resolved}.redacted" in context.artifact_paths
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = _read_json_value(path)
    return payload if isinstance(payload, dict) else {}


def _read_json_value(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
