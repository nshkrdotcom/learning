import csv
import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mwb.cli import app
from mwb.project import Project, ProjectManager


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def write_valid_ledgers(project: Project) -> None:
    logs = project.root / "research" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "claim_ledger.md").write_text(
        """# Claim Ledger

### C001 - Negation feature is associated with target contrast

```yaml
claim_id: C001
title: Negation feature is associated with target contrast
status: single_run_evidence
scope: Pythia-70M layer 2 negation bundle
allowed:
  - "is associated with"
forbidden:
  - "is the mechanism"
required_caveats:
  - "single run"
debt_flags:
  - control_leaky
linked_runs:
  - run_phase14
linked_decisions:
  - D001
copilot_session_id: null
```

Evidence:
- One diagnostic run exists.

Contradicting evidence:
- Controls moved too much.
""",
        encoding="utf-8",
    )
    (logs / "decision_log.md").write_text(
        """# Decision Log

## D001 - Keep control leakage visible

```yaml
decision_id: D001
status: accepted
affected_claims:
  - C001
decision_type: methodology
copilot_session_id: null
```

Decision:
Do not promote mechanism wording while controls are leaky.
""",
        encoding="utf-8",
    )
    (logs / "research_log.md").write_text(
        """# Research Log

## 2026-06-26

```yaml
entry_id: R2026-06-26-001
linked_runs:
  - run_phase14
linked_claims:
  - C001
linked_decisions:
  - D001
open_questions:
  - Which controls discriminate sentiment from negation?
copilot_session_id: null
```

### Question

### Context

### Hypothesis

### Work done

### Result

### Interpretation

### Decision

### Open questions
""",
        encoding="utf-8",
    )
    with (logs / "run_ledger.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "date": "2026-06-26",
                "run_id": "run_phase14",
                "git_commit": "abc123",
                "phase": "phase14",
                "purpose": "validate ledger indexing",
                "hypothesis": "C001",
                "command": "uv run mwb graph rebuild",
                "model": "EleutherAI/pythia-70m-deduped",
                "hook_point": "blocks.2.hook_resid_post",
                "sae_release": "pythia-70m-deduped-res-sm",
                "sae_id": "blocks.2.hook_resid_post",
                "ranking_dir": "",
                "out_dir": ".mechanism/runs/run_phase14",
                "seed": "0",
                "per_family": "",
                "top_k_features": "",
                "baseline_mode": "",
                "operations": "diagnostic",
                "status": "insufficient_evidence",
                "blocker": "control_leaky",
                "key_metric_1": "target_delta=0.7",
                "key_metric_2": "control_delta=0.6",
                "artifact_paths": ".mechanism/runs/run_phase14",
                "decision": "D001",
            }
        )


def write_run_and_card(project: Project) -> dict[str, str]:
    refs = {
        "run": "run_phase14",
        "card": "mc_phase14",
        "claim": "C014",
    }
    run_dir = project.mechanism_dir / "runs" / refs["run"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_ref": refs["run"],
                "source_kind": "phase14_fixture",
                "source_hypothesis_ref": "hyp_phase14",
                "status": "insufficient_evidence",
                "claim_bearing": False,
                "evidence_posture": "diagnostic_insufficient",
                "created_at": "2026-06-26T00:00:00Z",
                "backend_capabilities": {"source_backend": "fixture"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "control_metrics.json").write_text(
        json.dumps({"target_delta": 0.7, "matched_control_delta": 0.6}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "blocker_report.json").write_text(
        json.dumps(
            {
                "wb_ref": "blocker_phase14",
                "run_ref": refs["run"],
                "blockers": ["control_leaky"],
                "primary_blocker": "control_leaky",
                "blocking_metrics": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    card = {
        "wb_ref": refs["card"],
        "wb_type": "MechanismCard",
        "title": "MechanismCard: phase 14",
        "status": "insufficient_evidence",
        "evidence_tier": "association",
        "run_ref": refs["run"],
        "claim_ref": refs["claim"],
        "allowed_language": ["is associated with"],
        "blocked_language": ["is the mechanism"],
        "artifact_refs": [],
        "metadata": {
            "claim_ref": refs["claim"],
            "blockers": ["control_leaky"],
            "scientific_debt": [
                {
                    "debt_ref": "debt_phase14_control_leaky",
                    "kind": "controls",
                    "blocker": "control_leaky",
                    "description": "Controls moved too much.",
                }
            ],
        },
        "parents": [refs["run"]],
    }
    (run_dir / "mechanism_card.json").write_text(
        json.dumps(card, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cards_dir = project.mechanism_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    (cards_dir / f"{refs['card']}.json").write_text(
        json.dumps(card, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return refs


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


def test_ledger_validate_parses_and_indexes_research_ledgers(
    tmp_path: Path, monkeypatch
) -> None:
    from mwb.ledgers import validate_ledgers
    from mwb.sqlite_index import fetch_payload

    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    write_valid_ledgers(project)
    monkeypatch.chdir(tmp_path)

    report = validate_ledgers(project)

    assert report["status"] == "ok"
    assert report["counts"] == {
        "claims": 1,
        "decisions": 1,
        "research_log_entries": 1,
        "run_ledger_rows": 1,
    }
    assert fetch_payload(project.sqlite_path, "claims", "C001")["status"] == "single_run_evidence"
    assert fetch_payload(project.sqlite_path, "decisions", "D001")["status"] == "accepted"
    assert fetch_payload(project.sqlite_path, "runs", "run_phase14")["ledger_kind"] == "run_ledger"
    assert (
        fetch_payload(project.sqlite_path, "research_log_entries", "R2026-06-26-001")[
            "entry_id"
        ]
        == "R2026-06-26-001"
    )


def test_ledger_cli_validate_and_rebuild_preserve_ledgers(
    tmp_path: Path, monkeypatch
) -> None:
    from mwb.sqlite_index import fetch_payload, rebuild_sqlite_index

    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    write_valid_ledgers(project)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["ledger", "validate"])
    rebuilt = tmp_path / "rebuilt.sqlite"
    rebuild_report = rebuild_sqlite_index(project, output_path=rebuilt)

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["counts"]["claims"] == 1
    assert rebuild_report["counts"]["claims"] == 1
    assert rebuild_report["counts"]["decisions"] == 1
    assert rebuild_report["counts"]["research_log_entries"] == 1
    assert fetch_payload(rebuilt, "claims", "C001")["claim_id"] == "C001"


def test_ledger_proposals_are_human_reviewable_files(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    refs = write_run_and_card(project)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    run_result = runner.invoke(app, ["ledger", "propose-run", refs["run"]])
    claim_result = runner.invoke(app, ["ledger", "propose-claim", refs["card"]])

    assert run_result.exit_code == 0, run_result.output
    assert claim_result.exit_code == 0, claim_result.output
    run_proposal = project.mechanism_dir / "runs" / refs["run"] / "run_ledger_row.csv"
    claim_proposal_md = project.mechanism_dir / "proposals" / "claims" / f"{refs['claim']}.md"
    claim_proposal_json = project.mechanism_dir / "proposals" / "claims" / f"{refs['claim']}.json"
    assert run_proposal.exists()
    assert claim_proposal_md.exists()
    assert claim_proposal_json.exists()
    with run_proposal.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["run_id"] == refs["run"]
    assert rows[0]["status"] == "insufficient_evidence"
    assert f"### {refs['claim']} - MechanismCard: phase 14" in claim_proposal_md.read_text()
    assert json.loads(claim_proposal_json.read_text())["claim_id"] == refs["claim"]


def test_project_init_creates_git_visible_research_scaffold(tmp_path: Path) -> None:
    init_git_repo(tmp_path)

    ProjectManager.init(tmp_path, name="self-ground")

    assert (tmp_path / "research" / "logs" / "claim_ledger.md").exists()
    assert (tmp_path / "research" / "logs" / "run_ledger.csv").exists()
    assert (tmp_path / "research" / "logs" / "decision_log.md").exists()
    assert (tmp_path / "research" / "logs" / "research_log.md").exists()
