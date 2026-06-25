from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from mechledger.alias import resolve_run_id
from mechledger.core.claim_ledger import parse_claim_ledger
from mechledger.core.decision_log import parse_decision_log
from mechledger.core.run_ledger import DEFAULT_RUN_LEDGER_COLUMNS, parse_run_ledger
from mechledger.debt_report import write_scientific_debt_report
from mechledger.project import Project, now_utc


def append_run_ledger(project: Project, alias: str, *, yes: bool = False) -> str:
    run_id = resolve_run_id(project, alias)
    proposal = project.runs_dir / run_id / "run_ledger_row.csv"
    if not proposal.exists():
        raise FileNotFoundError(f"Run ledger proposal missing: {proposal}")
    with proposal.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != DEFAULT_RUN_LEDGER_COLUMNS:
            raise ValueError("run ledger proposal header is incompatible with project ledger")
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError("run ledger proposal must contain exactly one row")
    ledger_path = project.root / project.config.default_run_ledger
    ledger = parse_run_ledger(ledger_path)
    if any(row["run_id"] == run_id for row in ledger.rows):
        raise ValueError(f"run_id {run_id} already exists in {ledger_path}")
    if not yes:
        raise PermissionError("confirmation required; rerun with --yes to append")
    with ledger_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_RUN_LEDGER_COLUMNS)
        writer.writerow(rows[0])
    return run_id


def crystallize_experiment(
    project: Project, run_aliases: list[str], experiment_id: str, title: str
) -> Path:
    run_ids = [resolve_run_id(project, alias) for alias in run_aliases]
    slug = "".join(char.lower() if char.isalnum() else "_" for char in title).strip("_")
    slug = "_".join(part for part in slug.split("_") if part)[:60]
    path = project.root / "research/experiments" / f"{experiment_id}_{slug}.md"
    if path.exists():
        raise FileExistsError(f"ExperimentSpec already exists: {path}")
    source_runs = "\n".join(f"  - {run_id}" for run_id in run_ids)
    path.write_text(
        f"""# {experiment_id}: {title}

```yaml
experiment_id: {experiment_id}
status: draft
claim_targets: []
source_runs:
{source_runs}
prerequisites: []
config_files: []
expected_artifacts: []
```

## Status
draft

## Research question
TODO: state the question that emerged from the source runs.

## Hypothesis
TODO: state the post-hoc hypothesis without promoting claims.

## Model / SAE / Hook
TODO

## Task
TODO

## Mechanism objects
TODO

## Claim format
TODO

## Intervention
TODO

## Metrics
TODO

## Baselines
TODO

## Controls
TODO

## Success criterion
TODO

## Failure criterion
TODO

## Prerequisites
TODO

## Expected artifacts
TODO

## Evidence seed
{chr(10).join(f"- {run_id}" for run_id in run_ids)}

## Notes
Crystallized after exploratory/source runs; no claim promotion applied.
""",
        encoding="utf-8",
    )
    return path


def propose_claim(project: Project, alias: str, *, regenerate: bool = False) -> Path:
    run_id = resolve_run_id(project, alias)
    run_dir = project.runs_dir / run_id
    path = run_dir / "claim_update_proposal.json"
    if path.exists() and not regenerate:
        return path
    ledger = parse_claim_ledger(project.root / project.config.default_claim_ledger)
    target_claim_id = next(iter(ledger.claims), None)
    expected_block_hash = ledger.claims[target_claim_id].block_hash if target_claim_id else None
    proposal = {
        "proposal_id": f"CP-{run_id}",
        "run_id": run_id,
        "generated_at": now_utc(),
        "target_claim_id": target_claim_id,
        "current_claim_status_at_generation": ledger.claims[target_claim_id].status.value
        if target_claim_id
        else None,
        "proposed_status": None,
        "proposed_direction": "neutral",
        "expected_claim_ledger_hash": _hash_file(
            project.root / project.config.default_claim_ledger
        ),
        "expected_claim_block_hash": expected_block_hash,
        "supporting_metric_names": [],
        "contradicting_metric_names": [],
        "supporting_artifact_paths": [],
        "contradicting_artifact_paths": [],
        "scientific_debt_ids": [],
        "blocking_issues": [],
        "required_human_checks": [
            "Review evidence manually; MechLedger is not a claim truth oracle."
        ],
        "proposed_markdown_patch_path": str(run_dir / "claim_update_proposal.md"),
        "review_status": "pending",
        "reviewed_at": None,
        "reviewed_by": None,
        "force_applied": False,
    }
    path.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "claim_update_proposal.md").write_text(
        f"# Claim Update Proposal for {run_id}\n\nNo automatic claim promotion proposed.\n",
        encoding="utf-8",
    )
    return path


def review_claim(project: Project, alias: str, *, apply: bool = False, yes: bool = False) -> str:
    run_id = resolve_run_id(project, alias)
    proposal_path = project.runs_dir / run_id / "claim_update_proposal.json"
    if not proposal_path.exists():
        propose_claim(project, run_id)
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    ledger = parse_claim_ledger(project.root / project.config.default_claim_ledger)
    target = proposal.get("target_claim_id")
    stale = False
    if target and target in ledger.claims:
        stale = ledger.claims[target].block_hash != proposal.get("expected_claim_block_hash")
    if apply and stale:
        raise ValueError("claim proposal is stale; regenerate or force review after checking diff")
    if apply and not yes:
        raise PermissionError("confirmation required; rerun with --yes")
    return "stale" if stale else "current"


def waive_debt(project: Project, debt_id: str, decision_id: str) -> Path:
    decisions = parse_decision_log(project.root / project.config.default_decision_log)
    decision = decisions.decisions.get(decision_id)
    if decision is None or decision.status != "accepted":
        raise ValueError(f"Decision {decision_id} must exist and have status accepted.")
    for report_path in project.runs_dir.glob("*/scientific_debt_report.json"):
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        changed = False
        for debt in payload.get("debts", []):
            if debt["debt_id"] == debt_id:
                debt["status"] = "waived"
                debt["waiver_decision_id"] = decision_id
                debt["resolved_at"] = now_utc()
                changed = True
        if changed:
            report_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            from mechledger.core.debt import ScientificDebtReport

            report = ScientificDebtReport.model_validate(
                {k: payload[k] for k in ScientificDebtReport.model_fields if k in payload}
            )
            write_scientific_debt_report(report_path.parent, report)
            return report_path
    raise FileNotFoundError(f"Debt {debt_id} not found in local debt reports.")


def session_close(project: Project, *, accept: bool = False, since: str | None = None) -> Path:
    drafts = project.mechledger_dir / "session_drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    stamp = now_utc().replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
    path = drafts / f"{stamp}.md"
    changed = _changed_research_files(project, since)
    path.write_text(
        "# Session Close Draft\n\n"
        f"Generated at: {now_utc()}\n\n"
        "## Changed research files\n"
        + "".join(f"- {item}\n" for item in changed)
        + "\n## Open questions\n\n- TODO\n",
        encoding="utf-8",
    )
    if accept:
        log_path = project.root / project.config.default_research_log
        entry = (
            f"\n## {now_utc()[:10]}\n\n```yaml\nentry_id: R{now_utc()[:10]}-session\n"
            "linked_runs: []\nlinked_claims: []\nlinked_decisions: []\nopen_questions: []\n"
            "copilot_session_id: null\n```\n\n### Question\n\n### Context\n\n### Hypothesis\n\n"
            "### Work done\nSession close accepted.\n\n### Result\n\n### Interpretation\n\n"
            "### Decision\n\n### Open questions\n\n"
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(entry)
        (project.mechledger_dir / "last_session_close.json").write_text(
            json.dumps({"last_session_close_at": now_utc()}, indent=2) + "\n",
            encoding="utf-8",
        )
        path.unlink()
    return path


def decision_new_from_diff(project: Project) -> Path:
    decisions = parse_decision_log(project.root / project.config.default_decision_log)
    next_num = len(decisions.decisions) + 1
    decision_id = f"D{next_num:03d}"
    path = project.root / project.config.default_decision_log
    diff_summary = _git_diff_name_only(project.root)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## {decision_id} - Review declared-surface diff\n\n"
            "```yaml\n"
            f"decision_id: {decision_id}\nstatus: proposed\naffected_experiments: []\n"
            "affected_claims: []\ndecision_type: methodology\ncopilot_session_id: null\n```\n\n"
            "Decision:\nTODO\n\nReason:\nTODO\n\nEvidence:\n"
            + "".join(f"- {item}\n" for item in diff_summary)
        )
    return path


def _changed_research_files(project: Project, since: str | None) -> list[str]:
    if since:
        return [
            str(path.relative_to(project.root))
            for path in project.root.glob("research/**/*")
            if path.is_file()
        ]
    return _git_diff_name_only(project.root) or [
        str(path.relative_to(project.root))
        for path in project.root.glob("research/**/*")
        if path.is_file()
    ]


def _git_diff_name_only(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in result.stdout.splitlines() if line.startswith("research/")]


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
