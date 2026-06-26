from __future__ import annotations

from pathlib import Path

from mechledger.core.claim_ledger import ClaimRecord, parse_claim_ledger
from mechledger.draftguard import DraftCheckResult, check_draft_files
from mechledger.project import Project


def draft_suggestion_report(project: Project, files: list[Path]) -> str:
    ledger_path = project.resolve(project.config.default_claim_ledger)
    result = check_draft_files(files, claim_ledger_path=ledger_path)
    claims = parse_claim_ledger(ledger_path).claims
    lines = [
        "# MechLedger Draft Suggestions",
        "",
        "This deterministic report does not rewrite prose with AI, prove semantic "
        "correctness, or suppress Draft Guard violations.",
        "",
        "## Diagnostics",
    ]
    _append_diagnostics(lines, result)
    lines.extend(["", "## Safe Language Checklist"])
    for claim_id in sorted({violation.claim_id for violation in result.violations}):
        claim = claims.get(claim_id)
        if claim is None:
            continue
        _append_claim_policy(lines, claim)
    return "\n".join(lines).rstrip() + "\n"


def write_draft_suggestions(project: Project, files: list[Path], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(draft_suggestion_report(project, files), encoding="utf-8")
    return out


def claim_language_report(
    project: Project,
    *,
    claim_ids: list[str] | None = None,
    all_claims: bool = False,
) -> str:
    claims = parse_claim_ledger(project.resolve(project.config.default_claim_ledger)).claims
    if all_claims:
        selected = sorted(claims.values(), key=lambda item: item.claim_id)
    else:
        selected = []
        for claim_id in claim_ids or []:
            claim = claims.get(claim_id)
            if claim is None:
                raise ValueError(f"Unknown claim: {claim_id}")
            selected.append(claim)
    if not selected:
        raise ValueError("Use --claim C001 or --all.")
    lines = [
        "# Claim Language Report",
        "",
        "This deterministic report is a reviewer aid. It does not verify scientific "
        "truth, citations, or untagged claims.",
    ]
    for claim in selected:
        _append_claim_policy(lines, claim)
    return "\n".join(lines).rstrip() + "\n"


def write_claim_language_report(
    project: Project,
    out: Path,
    *,
    claim_ids: list[str] | None = None,
    all_claims: bool = False,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        claim_language_report(project, claim_ids=claim_ids, all_claims=all_claims),
        encoding="utf-8",
    )
    return out


def _append_diagnostics(lines: list[str], result: DraftCheckResult) -> None:
    if not result.violations:
        lines.append("- no Draft Guard diagnostics")
        return
    for violation in sorted(
        result.violations,
        key=lambda item: (str(item.file), item.line or 0, item.claim_id, item.violation_type),
    ):
        lines.append(
            f"- `{violation.violation_type}` [CLAIM:{violation.claim_id}] "
            f"{violation.file}:{violation.line}: {violation.message}"
        )
        if violation.suggested_rewrite:
            lines.append(f"  Suggested fix: {violation.suggested_rewrite}")
        if violation.window:
            lines.append(f"  Window: {violation.window}")


def _append_claim_policy(lines: list[str], claim: ClaimRecord) -> None:
    lines.extend(
        [
            "",
            f"## {claim.claim_id} - {claim.title or claim.heading_title}",
            "",
            f"- status: `{claim.status.value}`",
            f"- allowed phrases: {', '.join(claim.allowed) or 'none'}",
            f"- forbidden phrases: {', '.join(claim.forbidden) or 'none'}",
            f"- required caveats: {', '.join(claim.required_caveats) or 'none'}",
            f"- unresolved debt: {', '.join(claim.debt_flags) or 'none'}",
        ]
    )
