from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1.0"

MECHLEDGER_GITIGNORE_BLOCK = """# MechLedger local run records and indexes
.mechledger/runs/
.mechledger/alias_cache.txt
.mechledger/index.sqlite
.mechledger/cache/
.mechledger/tmp/
.mechledger/copilot/
.mechledger/session_drafts/

# Optional MechLedger bundles
*.tar.zst
"""


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    project_id: str
    created_at: str
    schema_version: str = SCHEMA_VERSION
    research_root: str = "research"
    default_claim_ledger: str = "research/logs/claim_ledger.md"
    default_decision_log: str = "research/logs/decision_log.md"
    default_research_log: str = "research/logs/research_log.md"
    default_run_ledger: str = "research/logs/run_ledger.csv"
    draft_guard: dict[str, Any] = Field(
        default_factory=lambda: {
            "missing_caveat_severity": "warning",
            "unresolved_debt_severity": "warning",
            "allow_overrides": True,
            "stale_claim_hash": "warning",
        }
    )
    cache_policy: dict[str, Any] = Field(
        default_factory=lambda: {"temp_fallback": True, "stale_heartbeat_seconds": 120}
    )


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path
    config_path: Path
    config: ProjectConfig

    @property
    def mechledger_dir(self) -> Path:
        return self.root / ".mechledger"

    @property
    def runs_dir(self) -> Path:
        return self.mechledger_dir / "runs"

    def resolve(self, configured_path: str) -> Path:
        return self.root / configured_path


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def command_cwd() -> Path:
    return Path(os.environ.get("PWD") or os.getcwd()).resolve()


def find_project(start: Path | None = None, *, allow_uninitialized: bool = False) -> Project:
    start = (start or command_cwd()).resolve()
    for candidate in [start, *start.parents]:
        config_path = candidate / ".mechledger/project.json"
        if config_path.exists():
            config = ProjectConfig.model_validate(
                json.loads(config_path.read_text(encoding="utf-8"))
            )
            return Project(root=candidate, config_path=config_path, config=config)
    if allow_uninitialized:
        root = git_root(start) or start
        config = ProjectConfig(
            project_id=f"mechledger-{uuid.uuid4().hex[:12]}",
            created_at=now_utc(),
        )
        return Project(root=root, config_path=root / ".mechledger/project.json", config=config)
    raise FileNotFoundError(
        "MechLedger project not initialized. Suggested fix: run `mechledger init`."
    )


def git_root(start: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(result.stdout.strip()).resolve()


def init_project(
    root: Path | None = None,
    *,
    force: bool = False,
    overwrite_template: bool = False,
) -> tuple[Project, list[str]]:
    root = (root or find_project(allow_uninitialized=True).root).resolve()
    project = find_project(root, allow_uninitialized=True)
    created: list[str] = []
    skipped: list[str] = []
    updated: list[str] = []

    for directory in [
        root / "research/experiments",
        root / "research/logs",
        root / "research/literature",
        root / "research/paper",
        root / "research/portfolio",
        root / ".mechledger/cache",
        root / ".mechledger/tmp",
        root / ".mechledger/runs",
        root / ".mechledger/session_drafts",
    ]:
        if directory.exists():
            skipped.append(str(directory.relative_to(root)))
        else:
            directory.mkdir(parents=True, exist_ok=True)
            created.append(str(directory.relative_to(root)))

    if project.config_path.exists() and not force:
        skipped.append(".mechledger/project.json")
        config = ProjectConfig.model_validate(
            json.loads(project.config_path.read_text(encoding="utf-8"))
        )
        project = Project(root=root, config_path=project.config_path, config=config)
    else:
        project.config_path.parent.mkdir(parents=True, exist_ok=True)
        project.config_path.write_text(
            json.dumps(project.config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        created.append(".mechledger/project.json")

    templates = {
        "research/experiments/TEMPLATE_experiment.md": experiment_template(),
        "research/logs/claim_ledger.md": "# Claim Ledger\n\n",
        "research/logs/decision_log.md": "# Decision Log\n\n",
        "research/logs/research_log.md": "# Research Log\n\n",
        "research/logs/run_ledger.csv": run_ledger_header() + "\n",
        "research/literature/prior_art_matrix.md": "# Prior Art Matrix\n\n",
        "research/literature/external_labels.md": "# External Labels\n\n",
        "research/paper/draft.md": "# Draft\n\n",
        "research/portfolio/WRITEUP_PLAN.md": "# Writeup Plan\n\n",
    }
    for rel, content in templates.items():
        path = root / rel
        may_overwrite = force or (overwrite_template and "TEMPLATE_" in path.name)
        if path.exists() and not may_overwrite:
            skipped.append(rel)
            continue
        path.write_text(content, encoding="utf-8")
        created.append(rel)

    gitignore_result = update_gitignore(root)
    if gitignore_result:
        updated.append(gitignore_result)

    messages = [f"created {item}" for item in created]
    messages.extend(f"skipped {item}" for item in skipped)
    messages.extend(f"updated {item}" for item in updated)
    return find_project(root), messages


def update_gitignore(root: Path) -> str | None:
    path = root / ".gitignore"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if ".mechledger/runs/" in existing and "*.tar.zst" in existing:
        return None
    if existing and not existing.endswith("\n"):
        existing += "\n"
    path.write_text(existing + "\n" + MECHLEDGER_GITIGNORE_BLOCK, encoding="utf-8")
    return ".gitignore"


def run_ledger_header() -> str:
    return (
        "date,run_id,git_commit,phase,purpose,hypothesis,command,model,hook_point,"
        "sae_release,sae_id,ranking_dir,out_dir,seed,per_family,top_k_features,"
        "baseline_mode,operations,status,blocker,key_metric_1,key_metric_2,"
        "artifact_paths,decision"
    )


def experiment_template() -> str:
    return """# E000: Template experiment

```yaml
experiment_id: E000
status: draft
claim_targets: []
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
