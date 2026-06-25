from __future__ import annotations

import json
from pathlib import Path

from mechledger.io import ensure_file, utc_now
from mechledger.models import RUN_LEDGER_COLUMNS

GITIGNORE_BLOCK = """\
# MechLedger local run records and indexes
.mechledger/runs/
.mechledger/index.sqlite
.mechledger/cache/
.mechledger/tmp/
.mechledger/copilot/
.mechledger/session_drafts/

# Optional MechLedger bundles
*.tar.zst
"""


def init_project(project_root: str | Path, project_name: str | None = None) -> None:
    root = Path(project_root)
    for relative in [
        "research/experiments",
        "research/logs",
        "research/literature",
        "research/paper",
        "research/portfolio",
        ".mechledger/runs",
        ".mechledger/cache",
        ".mechledger/tmp",
    ]:
        (root / relative).mkdir(parents=True, exist_ok=True)
    project = {
        "schema_version": "0.1.0",
        "project_name": project_name or root.name,
        "initialized_at": utc_now(),
    }
    project_path = root / ".mechledger" / "project.json"
    if not project_path.exists():
        project_path.write_text(
            json.dumps(project, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _append_gitignore(root / ".gitignore")
    ensure_file(root / "research" / "logs" / "claim_ledger.md", _claim_ledger_template())
    ensure_file(root / "research" / "logs" / "decision_log.md", _decision_log_template())
    ensure_file(root / "research" / "logs" / "research_log.md", _research_log_template())
    ensure_file(root / "research" / "logs" / "run_ledger.csv", ",".join(RUN_LEDGER_COLUMNS) + "\n")
    ensure_file(
        root / "research" / "experiments" / "TEMPLATE_experiment.md", _experiment_template()
    )
    ensure_file(root / "research" / "literature" / "prior_art_matrix.md", "# Prior Art Matrix\n\n")
    ensure_file(root / "research" / "literature" / "external_labels.md", "# External Labels\n\n")
    ensure_file(root / "research" / "paper" / "draft.md", "# Draft\n\n")
    ensure_file(root / "research" / "portfolio" / "WRITEUP_PLAN.md", "# Writeup Plan\n\n")


def _append_gitignore(path: Path) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if ".mechledger/runs/" in existing:
        return
    separator = "" if not existing or existing.endswith("\n") else "\n"
    path.write_text(existing + separator + GITIGNORE_BLOCK, encoding="utf-8")


def _claim_ledger_template() -> str:
    return """\
# Claim Ledger

### C001 - Initial tagged claim language is guarded

```yaml
claim_id: C001
title: Initial tagged claim language is guarded
status: unsupported
allowed:
  - observed
  - preliminary
forbidden:
  - proves that
  - is the mechanism
  - identifies the mechanism
required_caveats: []
debt_flags: []
linked_experiments: []
linked_runs: []
linked_decisions: []
copilot_session_id: null
```

Evidence:
- None yet.

Contradicting evidence:
- None yet.

Required next evidence:
- Add a real run and review the claim.

Notes:
This starter claim exists so Draft Guard can be exercised immediately.
"""


def _decision_log_template() -> str:
    return """\
# Decision Log

## D001 - Initialize MechLedger

```yaml
decision_id: D001
status: accepted
affected_experiments: []
affected_claims:
  - C001
decision_type: methodology
copilot_session_id: null
```

Decision:
Use MechLedger flat files to keep claims, runs, and draft language linked.

Reason:
The repository should make scientific debt and evidence provenance visible.
"""


def _research_log_template() -> str:
    today = utc_now()[:10]
    return f"""\
# Research Log

## {today}

```yaml
entry_id: R{today}-001
linked_runs: []
linked_claims: []
linked_decisions: []
open_questions: []
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
"""


def _experiment_template() -> str:
    return """\
# E001: Replace with experiment title

```yaml
experiment_id: E001
status: planned
claim_targets:
  - C001
source_runs: []
prerequisites: []
config_files: []
expected_artifacts: []
```

## Status

## Research question

## Hypothesis

## Model / SAE / Hook

## Task

## Mechanism objects

## Claim format

## Intervention

## Metrics

## Baselines

## Controls

## Success criterion

## Failure criterion

## Prerequisites

## Expected artifacts

## Notes
"""
