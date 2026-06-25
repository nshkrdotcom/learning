from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mechledger.models import IncomparableStatusError, claim_status_at_least
from mechledger.parsers import parse_claim_ledger, parse_decision_log, parse_experiment_spec


@dataclass(slots=True)
class ExperimentAction:
    experiment_id: str
    title: str
    status: str
    unmet: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExperimentGroups:
    ready: list[ExperimentAction] = field(default_factory=list)
    blocked: list[ExperimentAction] = field(default_factory=list)
    not_started: list[ExperimentAction] = field(default_factory=list)
    completed: list[ExperimentAction] = field(default_factory=list)
    retired: list[ExperimentAction] = field(default_factory=list)


def classify_experiments(project_root: str | Path) -> ExperimentGroups:
    root = Path(project_root)
    claims = parse_claim_ledger(root / "research" / "logs" / "claim_ledger.md")
    decisions = parse_decision_log(root / "research" / "logs" / "decision_log.md")
    run_rows = _run_ledger_rows(root / "research" / "logs" / "run_ledger.csv")
    specs = [
        parse_experiment_spec(path)
        for path in sorted((root / "research" / "experiments").glob("*.md"))
        if not path.name.startswith("TEMPLATE")
    ]
    groups = ExperimentGroups()
    completed_ids = {
        spec.experiment_id
        for spec in specs
        if spec.status in {"completed", "completed_and_reviewed"}
    }
    reviewed_ids = {spec.experiment_id for spec in specs if spec.status == "completed_and_reviewed"}
    for spec in specs:
        action = ExperimentAction(spec.experiment_id, spec.title, spec.status)
        if spec.status == "retired":
            groups.retired.append(action)
            continue
        if spec.status in {"completed", "completed_and_reviewed"}:
            groups.completed.append(action)
            continue
        for prerequisite in spec.prerequisites:
            action.unmet.extend(
                _unmet_prerequisite(
                    prerequisite, root, claims, decisions, completed_ids, reviewed_ids, run_rows
                )
            )
        if action.unmet:
            groups.blocked.append(action)
        elif spec.status in {"planned", "ready", "draft"}:
            groups.ready.append(action)
        else:
            groups.not_started.append(action)
    return groups


def _unmet_prerequisite(
    prerequisite: dict[str, Any],
    root: Path,
    claims,
    decisions,
    completed_ids: set[str],
    reviewed_ids: set[str],
    run_rows: list[dict[str, str]],
) -> list[str]:
    kind = prerequisite.get("type")
    object_id = prerequisite.get("id")
    if kind == "decision_accepted":
        decision = decisions.decisions.get(str(object_id))
        if decision is None or decision.status != "accepted":
            return [f"decision {object_id} is not accepted"]
    if kind == "claim_status_at_least":
        claim = claims.claims.get(str(object_id))
        required = str(prerequisite.get("status"))
        if claim is None:
            return [f"claim {object_id} is missing"]
        try:
            if not claim_status_at_least(claim.status, required):
                return [f"claim {object_id} is {claim.status}, not at least {required}"]
        except IncomparableStatusError as exc:
            return [str(exc)]
    if kind == "experiment_completed" and object_id not in completed_ids:
        return [f"experiment {object_id} is not completed"]
    if kind == "experiment_completed_and_reviewed" and object_id not in reviewed_ids:
        return [f"experiment {object_id} is not completed_and_reviewed"]
    if kind == "artifact_exists":
        path = root / str(prerequisite.get("path", ""))
        if not path.exists():
            return [f"artifact {path} is missing"]
    if kind == "run_class_exists":
        experiment = str(prerequisite.get("experiment"))
        run_class = str(prerequisite.get("run_class"))
        if not any(
            row.get("phase") == experiment and row.get("operations") == run_class
            for row in run_rows
        ):
            return [f"run class {run_class} for {experiment} is missing"]
    return []


def _run_ledger_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
