from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mechledger.core.claim_ledger import parse_claim_ledger
from mechledger.core.debt import DebtSeverity, DebtStatus
from mechledger.debt_report import generate_scientific_debt_report
from mechledger.project import Project, now_utc


def write_claim_update_proposal(
    project: Project,
    run_id: str,
    *,
    regenerate: bool = True,
) -> Path:
    run_dir = project.runs_dir / run_id
    path = run_dir / "claim_update_proposal.json"
    if path.exists() and not regenerate:
        return path

    ledger = parse_claim_ledger(project.root / project.config.default_claim_ledger)
    target_claim_id = _target_claim_id(run_dir, ledger.claims)
    target_claim = ledger.claims.get(target_claim_id) if target_claim_id else None
    report = generate_scientific_debt_report(project, run_id)
    evidence = _read_json(run_dir / "evidence_assessment.json")
    recommended = evidence.get("recommended_claim_status") or None
    open_debts = [
        debt for debt in report.debts if getattr(debt, "status", None) == DebtStatus.OPEN
    ]
    blocking_issues = [
        f"{debt.debt_id}: {debt.message}"
        for debt in open_debts
        if getattr(debt, "severity", None) in {DebtSeverity.BLOCKING, DebtSeverity.SERIOUS}
    ]
    proposal = {
        "proposal_id": f"CP-{run_id}",
        "run_id": run_id,
        "generated_at": now_utc(),
        "target_claim_id": target_claim_id,
        "current_claim_status_at_generation": target_claim.status.value
        if target_claim
        else None,
        "proposed_status": recommended,
        "proposed_direction": _proposal_direction(recommended, blocking_issues),
        "expected_claim_ledger_hash": _hash_file(
            project.root / project.config.default_claim_ledger
        ),
        "expected_claim_block_hash": target_claim.block_hash if target_claim else None,
        "supporting_metric_names": _metric_names(run_dir, only_if_supports=recommended),
        "contradicting_metric_names": _failed_condition_ids(evidence, recommended),
        "supporting_artifact_paths": _artifact_paths(run_dir, {"supporting", "required"}),
        "contradicting_artifact_paths": _artifact_paths(run_dir, {"contradicting"}),
        "scientific_debt_ids": [debt.debt_id for debt in open_debts],
        "blocking_issues": blocking_issues,
        "required_human_checks": _human_checks(recommended, blocking_issues),
        "proposed_markdown_patch_path": str(run_dir / "claim_update_proposal.md"),
        "review_status": "pending",
        "reviewed_at": None,
        "reviewed_by": None,
        "force_applied": False,
    }
    path.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(run_dir / "claim_update_proposal.md", proposal)
    return path


def _target_claim_id(run_dir: Path, claims: dict[str, Any]) -> str | None:
    run_json = _read_json(run_dir / "run.json")
    experiment_id = run_json.get("experiment_id")
    for claim in claims.values():
        if run_json.get("run_id") in claim.linked_runs:
            return claim.claim_id
    if experiment_id:
        for claim in claims.values():
            if experiment_id in claim.linked_experiments:
                return claim.claim_id
    return next(iter(claims), None)


def _proposal_direction(recommended: str | None, blocking_issues: list[str]) -> str:
    if recommended in {"candidate_claim", "causal_support", "replicated_evidence"}:
        return "supports"
    if recommended in {"failed_or_weakened", "contradicted"}:
        return "weakens"
    if blocking_issues:
        return "blocks"
    return "neutral"


def _metric_names(run_dir: Path, *, only_if_supports: str | None) -> list[str]:
    if only_if_supports != "candidate_claim":
        return []
    names = []
    for row in _jsonl_rows(run_dir / "metrics.jsonl"):
        metric_name = row.get("metric_name")
        if metric_name:
            names.append(str(metric_name))
    return sorted(set(names))


def _failed_condition_ids(evidence: dict[str, Any], recommended: str | None) -> list[str]:
    if recommended not in {"failed_or_weakened", "contradicted"}:
        return []
    conditions = evidence.get("conditions") or {}
    return sorted(
        condition_id
        for condition_id, condition in conditions.items()
        if isinstance(condition, dict) and condition.get("passed") is False
    )


def _artifact_paths(run_dir: Path, relevances: set[str]) -> list[str]:
    manifest = _read_json(run_dir / "artifact_manifest.json")
    paths = []
    for artifact in manifest.get("artifacts", []):
        if artifact.get("claim_relevance") in relevances:
            paths.append(
                artifact.get("project_relative_path")
                or artifact.get("original_path")
                or artifact.get("artifact_id")
            )
    return sorted(str(path) for path in paths if path)


def _human_checks(recommended: str | None, blocking_issues: list[str]) -> list[str]:
    checks = ["Review evidence manually; MechLedger is not a claim truth oracle."]
    if recommended == "candidate_claim":
        checks.append("Confirm candidate support matches the claim scope before accepting.")
    if blocking_issues:
        checks.append("Resolve or waive blocking/serious scientific debt before promotion.")
    return checks


def _write_markdown(path: Path, proposal: dict[str, Any]) -> None:
    lines = [
        f"# Claim Update Proposal for {proposal['run_id']}",
        "",
        f"- Target claim: {proposal.get('target_claim_id') or 'none'}",
        f"- Proposed status: {proposal.get('proposed_status') or 'none'}",
        f"- Proposed direction: {proposal.get('proposed_direction')}",
        "",
        "## Scientific Debt",
    ]
    if proposal["scientific_debt_ids"]:
        lines.extend(f"- {debt_id}" for debt_id in proposal["scientific_debt_ids"])
    else:
        lines.append("- none")
    if proposal["blocking_issues"]:
        lines.extend(["", "## Blocking / Serious Issues"])
        lines.extend(f"- {issue}" for issue in proposal["blocking_issues"])
    lines.extend(
        [
            "",
            "No claim ledger mutation is applied automatically. Human review is required.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
