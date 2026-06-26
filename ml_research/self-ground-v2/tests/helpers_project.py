from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from mechledger.cli import app
from mechledger.project import run_ledger_header

runner = CliRunner()


def init_project(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    assert result.exit_code == 0, result.output


def write_claim_ledger(tmp_path: Path, *, status: str = "candidate_claim") -> None:
    (tmp_path / "research/logs/claim_ledger.md").write_text(
        f"""# Claim Ledger

### C001 - Target feature claim

```yaml
claim_id: C001
status: {status}
allowed:
  - preliminary evidence
forbidden:
  - proves causally
required_caveats:
  - single-run evidence
debt_flags:
  - missing_empirical_null
linked_experiments: [E001]
linked_runs: [RUN_E001]
linked_decisions: [D001]
scope: negation prompts
```
""",
        encoding="utf-8",
    )


def write_decision_log(tmp_path: Path, *, status: str = "accepted") -> None:
    (tmp_path / "research/logs/decision_log.md").write_text(
        f"""# Decision Log

## D001 - Method review

```yaml
decision_id: D001
status: {status}
affected_experiments: [E001]
affected_claims: [C001]
decision_type: methodology
copilot_session_id: null
```
""",
        encoding="utf-8",
    )


def write_research_log(tmp_path: Path) -> None:
    (tmp_path / "research/logs/research_log.md").write_text(
        """# Research Log

## 2026-06-25

```yaml
entry_id: R2026-06-25-001
linked_runs: [RUN_E001]
linked_claims: [C001]
linked_decisions: [D001]
open_questions:
  - Does the feature survive density-matched controls?
copilot_session_id: null
```

### Question
Does the intervention isolate the claimed feature?
### Context
Registered run and claim review.
### Hypothesis
Target delta should exceed matched control delta.
### Work done
Captured run metadata.
### Result
Preliminary evidence only.
### Interpretation
Debt remains visible.
### Decision
Use D001 for methodology review.
### Open questions
Does the feature survive density-matched controls?
""",
        encoding="utf-8",
    )


def write_experiment(tmp_path: Path) -> None:
    (tmp_path / "research/experiments/E001_test.md").write_text(
        """# E001: Test experiment

```yaml
experiment_id: E001
status: planned
claim_targets: [C001]
source_runs: [RUN_E001]
prerequisites:
  - type: decision_accepted
    id: D001
config_files: []
expected_artifacts: [results/artifact.json]
```

## Status
planned
## Research question
x
## Hypothesis
x
## Model / SAE / Hook
x
## Task
x
## Mechanism objects
x
## Claim format
x
## Intervention
x
## Metrics
x
## Baselines
x
## Controls
x
## Success criterion
x
## Failure criterion
x
## Prerequisites
x
## Expected artifacts
x
## Notes
x
""",
        encoding="utf-8",
    )


def write_run_ledger(tmp_path: Path) -> None:
    (tmp_path / "research/logs/run_ledger.csv").write_text(
        run_ledger_header()
        + "\n"
        + "2026-06-25,RUN_E001,abc,phase,purpose,hypothesis,cmd,pythia,hook,,,,,"
        + ",,,baseline,ablate,completed,,specificity_gap=0.3,,artifacts/result.json,D001\n",
        encoding="utf-8",
    )


def create_run(
    tmp_path: Path,
    *,
    run_id: str = "RUN_E001",
    metrics: dict[str, Any] | None = None,
    artifact: bool = True,
) -> Path:
    run_dir = tmp_path / ".mechledger/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "experiment_id": "E001",
                "run_class": "serious_evidence_run",
                "status": "completed",
                "started_at": "2026-06-25T00:00:00Z",
                "finished_at": "2026-06-25T00:00:01Z",
                "exit_code": 0,
                "command": "python run.py",
                "argv": ["python", "run.py"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metric_payload = metrics or {"specificity_gap": 0.3, "target_delta": 0.4}
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for metric_name, value in metric_payload.items():
            handle.write(json.dumps({"metric_name": metric_name, "value": value}) + "\n")
    (run_dir / "events.jsonl").write_text(
        json.dumps({"event_type": "completed", "metadata": {"feature_id": "sae_123"}}) + "\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "artifacts/result.json"
    artifact_path.parent.mkdir(exist_ok=True)
    artifact_path.write_text('{"ok": true}\n', encoding="utf-8")
    artifact_record = {
        "artifact_id": "A001",
        "original_path": "artifacts/result.json",
        "resolved_path": str(artifact_path),
        "project_relative_path": "artifacts/result.json",
        "artifact_type": "json",
        "content_hash": "sha256:test",
        "content_hash_status": "computed",
        "artifact_storage_backend": "git",
        "claim_relevance": "required",
        "review_status": "annotated",
        "description": "result artifact",
        "byte_size": artifact_path.stat().st_size,
        "auto_collected": False,
    }
    manifest = {"artifacts": [artifact_record] if artifact else []}
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "artifacts.jsonl").write_text(
        (json.dumps(artifact_record, sort_keys=True) + "\n") if artifact else "",
        encoding="utf-8",
    )
    (run_dir / "scientific_debt_report.json").write_text(
        json.dumps(
            {
                "report_id": f"SDR-{run_id}",
                "run_id": run_id,
                "experiment_id": "E001",
                "generated_at": "2026-06-25T00:00:00Z",
                "evaluated_assessments": ["empirical_null"],
                "threshold_sources": [],
                "clean_candidate_support": False,
                "summary": "open debt remains",
                "debts": [
                    {
                        "debt_id": "DPT006",
                        "debt_type": "missing_empirical_null",
                        "severity": "serious",
                        "claim_id": "C001",
                        "run_id": run_id,
                        "experiment_id": "E001",
                        "evidence_paths": [],
                        "message": "Missing empirical null.",
                        "required_resolution": "Register null evidence.",
                        "status": "open",
                        "waiver_decision_id": None,
                        "created_at": "2026-06-25T00:00:00Z",
                        "assessment_id": "empirical_null",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / ".mechledger/alias_cache.txt").write_text(
        f"{run_id}\t2026-06-25T00:00:00Z\tE001\t{run_id}\n", encoding="utf-8"
    )
    return run_dir


def populate_project(tmp_path: Path, *, claim_status: str = "candidate_claim") -> None:
    init_project(tmp_path)
    write_claim_ledger(tmp_path, status=claim_status)
    write_decision_log(tmp_path)
    write_research_log(tmp_path)
    write_experiment(tmp_path)
    write_run_ledger(tmp_path)
    create_run(tmp_path)
