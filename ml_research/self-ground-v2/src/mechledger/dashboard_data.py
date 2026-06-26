from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mechledger.inspection import collect_project, write_json
from mechledger.open_questions import list_questions
from mechledger.project import Project


def dashboard_summary(project: Project) -> dict[str, Any]:
    snapshot = collect_project(project)
    questions = list_questions(project)
    return {
        "project_id": project.config.project_id,
        "claims_by_status": _count(claim.status.value for claim in snapshot.claims.values()),
        "unresolved_debt_by_severity": _count(
            str(debt.get("severity"))
            for debt in snapshot.debts
            if debt.get("status") == "open"
        ),
        "debt_by_type": _count(
            str(debt.get("debt_type"))
            for debt in snapshot.debts
            if debt.get("status") == "open"
        ),
        "runs_by_status": _count(
            str(run.get("run", {}).get("status") or "unknown")
            for run in snapshot.runs.values()
        ),
        "runs_by_experiment": _count(
            str(run.get("run", {}).get("experiment_id") or "unlinked")
            for run in snapshot.runs.values()
        ),
        "artifacts_by_review_status": _count(
            str(artifact.get("review_status") or "unknown")
            for artifact in snapshot.artifact_metadata
        ),
        "artifacts_by_claim_relevance": _count(
            str(artifact.get("claim_relevance") or "none")
            for artifact in snapshot.artifact_metadata
        ),
        "decisions_by_status": _count(
            decision.status for decision in snapshot.decisions.values()
        ),
        "experiments_by_readiness": _count(
            spec.status for spec in snapshot.experiments.values()
        ),
        "open_questions": [
            question for question in questions if question.get("status") == "open"
        ],
        "warnings": sorted(snapshot.warnings),
    }


def write_dashboard_data(project: Project, out: Path) -> dict[str, Any]:
    payload = dashboard_summary(project)
    write_json(out, payload)
    return payload


def query_rows(project: Project, kind: str) -> list[dict[str, Any]]:
    snapshot = collect_project(project)
    if kind == "claims":
        return [
            {
                "claim_id": claim.claim_id,
                "status": claim.status.value,
                "scope": claim.scope,
                "linked_runs": claim.linked_runs,
                "linked_experiments": claim.linked_experiments,
            }
            for claim in sorted(snapshot.claims.values(), key=lambda item: item.claim_id)
        ]
    if kind == "runs":
        return [
            {
                "run_id": run_id,
                "status": record["run"].get("status"),
                "run_class": record["run"].get("run_class"),
                "experiment_id": record["run"].get("experiment_id"),
            }
            for run_id, record in sorted(snapshot.runs.items())
        ]
    if kind == "debt":
        return sorted(
            snapshot.debts,
            key=lambda item: (str(item.get("run_id")), str(item.get("debt_id"))),
        )
    if kind == "artifacts":
        return sorted(
            snapshot.artifact_metadata,
            key=lambda item: (str(item.get("run_id")), str(item.get("artifact_id"))),
        )
    if kind == "decisions":
        return [
            {
                "decision_id": decision.decision_id,
                "status": decision.status,
                "decision_type": decision.decision_type,
                "affected_claims": decision.affected_claims,
                "affected_experiments": decision.affected_experiments,
            }
            for decision in sorted(
                snapshot.decisions.values(), key=lambda item: item.decision_id
            )
        ]
    if kind == "experiments":
        return [
            {
                "experiment_id": spec.experiment_id,
                "title": spec.title,
                "status": spec.status,
                "claim_targets": spec.claim_targets,
                "source_runs": spec.source_runs,
            }
            for spec in sorted(snapshot.experiments.values(), key=lambda item: item.experiment_id)
        ]
    raise ValueError(f"Unknown query kind: {kind}")


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    status: str | None = None,
    claim: str | None = None,
    experiment: str | None = None,
    run: str | None = None,
    severity: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        row_status = row.get("status", row.get("review_status"))
        if status and str(row_status) != status:
            continue
        if severity and str(row.get("severity")) != severity:
            continue
        if claim and not _row_has(
            row,
            claim,
            "claim_id",
            "linked_claims",
            "claim_targets",
            "affected_claims",
        ):
            continue
        if experiment and not _row_has(
            row,
            experiment,
            "experiment_id",
            "linked_experiments",
            "affected_experiments",
        ):
            continue
        if run and not _row_has(row, run, "run_id", "linked_runs", "source_runs"):
            continue
        filtered.append(row)
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def rows_json(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def rows_text(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        primary = (
            row.get("claim_id")
            or row.get("run_id")
            or row.get("debt_id")
            or row.get("artifact_id")
            or row.get("decision_id")
            or row.get("experiment_id")
            or row
        )
        lines.append(str(primary))
    return "\n".join(lines) + ("\n" if lines else "")


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _row_has(row: dict[str, Any], value: str, *fields: str) -> bool:
    for field in fields:
        item = row.get(field)
        if item == value:
            return True
        if isinstance(item, list) and value in item:
            return True
    return False
