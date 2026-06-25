from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mechledger.artifacts import annotate_artifact, attach_artifact
from mechledger.draft_guard import check_draft_file
from mechledger.evidence import calibration_check, gate_check, telemetry_check
from mechledger.formatter import format_project
from mechledger.hooks import install_direct_hook, install_pre_commit_config
from mechledger.indexer import check_project, index_project
from mechledger.next_actions import classify_experiments
from mechledger.parsers import LedgerParseError, parse_claim_ledger, parse_experiment_spec
from mechledger.paths import find_project_root
from mechledger.run_capture import capture_command
from mechledger.scaffold import init_project
from mechledger.session import close_session
from mechledger.status import project_status

app = typer.Typer(help="MechLedger research integrity CLI.")
draft_app = typer.Typer(help="Draft Guard commands.")
artifact_app = typer.Typer(help="Artifact commands.")
gate_app = typer.Typer(help="Evidence gate commands.")
calibration_app = typer.Typer(help="Calibration commands.")
telemetry_app = typer.Typer(help="Telemetry commands.")
experiment_app = typer.Typer(help="Experiment commands.")
session_app = typer.Typer(help="Session commands.")
app.add_typer(draft_app, name="draft")
app.add_typer(artifact_app, name="artifact")
app.add_typer(gate_app, name="gate")
app.add_typer(calibration_app, name="calibration")
app.add_typer(telemetry_app, name="telemetry")
app.add_typer(experiment_app, name="experiment")
app.add_typer(session_app, name="session")


@app.command()
def init(
    project_name: Annotated[str | None, typer.Option("--project-name")] = None,
    install_pre_commit: Annotated[bool, typer.Option("--install-pre-commit")] = False,
) -> None:
    root = Path.cwd()
    init_project(root, project_name=project_name)
    if install_pre_commit:
        install_pre_commit_config(root)
    typer.echo(f"Initialized MechLedger project at {root}")


@app.command("install-hooks")
def install_hooks(direct: Annotated[bool, typer.Option("--direct")] = False) -> None:
    root = find_project_root()
    if direct:
        path = install_direct_hook(root)
        typer.echo(f"Installed direct Git hook into {path}.")
        typer.echo(
            "This hook is not committed and must be installed separately by each researcher."
        )
        typer.echo("For portable hooks, use the default pre-commit framework mode.")
    else:
        path = install_pre_commit_config(root)
        typer.echo(f"Updated {path}. Run `pre-commit install` to activate hooks.")


@draft_app.command("check")
def draft_check(
    paths: Annotated[list[Path] | None, typer.Argument()] = None,
    staged: Annotated[bool, typer.Option("--staged")] = False,
) -> None:
    root = find_project_root()
    if staged:
        staged_paths = _staged_paths(root)
        relevant = [
            root / path
            for path in staged_paths
            if path.endswith((".md", ".markdown", ".tex"))
            or path == "research/logs/claim_ledger.md"
            or path == ".mechledger/project.json"
        ]
        if not relevant:
            typer.echo("MechLedger: no staged draft or claim files changed; skipping.")
            raise typer.Exit(0)
        paths = relevant
    if not paths:
        paths = list((root / "research" / "paper").glob("*.md")) + list(
            (root / "research" / "paper").glob("*.tex")
        )
    try:
        ledger = parse_claim_ledger(root / "research" / "logs" / "claim_ledger.md")
    except LedgerParseError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    blocking = 0
    warnings = 0
    overrides = 0
    for path in paths:
        result = check_draft_file(path, ledger)
        overrides += len(result.overrides)
        for violation in result.violations:
            if violation.severity == "blocking" and not violation.suppressed_by_override:
                blocking += 1
            elif not violation.suppressed_by_override:
                warnings += 1
            typer.echo(
                f"{violation.severity.upper()} {violation.file}:{violation.line} "
                f"[CLAIM:{violation.claim_id}] {violation.violation_type}: {violation.message}"
            )
    typer.echo(f"Draft Guard: {blocking} blocking, {warnings} warnings, {overrides} overrides")
    raise typer.Exit(1 if blocking else 0)


@app.command()
def index(
    check: Annotated[bool, typer.Option("--check")] = False,
    staged: Annotated[bool, typer.Option("--staged")] = False,
) -> None:
    root = find_project_root()
    if staged:
        staged_paths = _staged_paths(root)
        if not any(
            path.startswith("research/") or path == ".mechledger/project.json"
            for path in staged_paths
        ):
            typer.echo("MechLedger: no staged research/indexed files changed; skipping.")
            raise typer.Exit(0)
    if check:
        result = check_project(root)
        if result.ok:
            typer.echo("MechLedger index check passed.")
            raise typer.Exit(0)
        for error in result.errors:
            typer.echo(error, err=True)
        raise typer.Exit(1)
    idx = index_project(root)
    claim_count = sum(idx.claim_count_by_status.values())
    typer.echo(
        f"Indexed {claim_count} claims, "
        f"{idx.experiment_count} experiments, {idx.run_count} runs."
    )


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    experiment: Annotated[str | None, typer.Option("--experiment")] = None,
    run_class: Annotated[str, typer.Option("--class")] = "scratch",
    purpose: Annotated[str | None, typer.Option("--purpose")] = None,
    hypothesis: Annotated[str | None, typer.Option("--hypothesis")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
) -> None:
    command = list(ctx.args)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        typer.echo("Usage error: pass a command after `--`.", err=True)
        raise typer.Exit(2)
    captured = capture_command(
        command,
        project_root=find_project_root(),
        experiment_id=experiment,
        run_class=run_class,
        purpose=purpose,
        hypothesis=hypothesis,
        run_id=run_id,
    )
    typer.echo(f"Created run: {captured.run_id}")
    typer.echo(f"Directory: {captured.run_dir}")
    typer.echo(f"Status: {'completed' if captured.exit_code == 0 else 'failed'}")


@app.command()
def attach(
    run_id: str,
    path: Path,
    type: Annotated[str | None, typer.Option("--type")] = None,
    claim_relevance: Annotated[str, typer.Option("--claim-relevance")] = "none",
    description: Annotated[str | None, typer.Option("--description")] = None,
    allow_missing: Annotated[bool, typer.Option("--allow-missing")] = False,
) -> None:
    root = find_project_root()
    run_dir = root / ".mechledger" / "runs" / run_id
    artifact = attach_artifact(
        run_dir,
        path,
        artifact_type=type,
        claim_relevance=claim_relevance,
        description=description,
        allow_missing=allow_missing,
    )
    typer.echo(f"Registered artifact {artifact.artifact_id}")


@artifact_app.command("annotate")
def artifact_annotate(
    run_id: str,
    artifact_id: str,
    claim_relevance: Annotated[str, typer.Option("--claim-relevance")],
    description: Annotated[str | None, typer.Option("--description")] = None,
) -> None:
    root = find_project_root()
    run_dir = root / ".mechledger" / "runs" / run_id
    artifact = annotate_artifact(
        run_dir, artifact_id, claim_relevance=claim_relevance, description=description
    )
    typer.echo(f"Annotated artifact {artifact.artifact_id}")


@gate_app.command("check")
def gate_check_cmd(run_id: str) -> None:
    report = gate_check(find_project_root(), run_id)
    typer.echo(report.summary)
    raise typer.Exit(1 if report.blockers else 0)


@calibration_app.command("check")
def calibration_check_cmd(run_id: str) -> None:
    report = calibration_check(find_project_root(), run_id)
    typer.echo(report.summary)
    raise typer.Exit(1 if report.blockers else 0)


@telemetry_app.command("check")
def telemetry_check_cmd(run_id: str) -> None:
    report = telemetry_check(find_project_root(), run_id)
    typer.echo(report.summary)
    raise typer.Exit(1 if report.blockers else 0)


@app.command()
def status() -> None:
    typer.echo(project_status(find_project_root()))


@app.command("next")
def next_cmd() -> None:
    groups = classify_experiments(find_project_root())
    if groups.ready:
        typer.echo("READY")
        for item in groups.ready:
            typer.echo(f"  {item.experiment_id} - {item.title}")
    if groups.blocked:
        typer.echo("BLOCKED")
        for item in groups.blocked:
            typer.echo(f"  {item.experiment_id} - {item.title}")
            for unmet in item.unmet:
                typer.echo(f"    - {unmet}")
    raise typer.Exit(0 if groups.ready else (1 if groups.blocked else 0))


@app.command("format")
def format_cmd(write: Annotated[bool, typer.Option("--write")] = False) -> None:
    changes = format_project(find_project_root(), write=write)
    if not changes:
        typer.echo("MechLedger format: no changes.")
        return
    for diff in changes.values():
        typer.echo(diff)


@experiment_app.command("validate")
def experiment_validate(paths: list[Path]) -> None:
    for path in paths:
        parse_experiment_spec(path)
        typer.echo(f"valid: {path}")


@experiment_app.command("crystallize")
def experiment_crystallize(
    runs: Annotated[list[str], typer.Option("--runs")],
    id: Annotated[str, typer.Option("--id")],
    title: Annotated[str, typer.Option("--title")],
) -> None:
    root = find_project_root()
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in title).strip("_")
    path = root / "research" / "experiments" / f"{id}_{slug}.md"
    yaml_runs = "\n".join(f"  - {run}" for run in runs)
    path.write_text(
        f"""# {id}: {title}

```yaml
experiment_id: {id}
status: draft
claim_targets: []
source_runs:
{yaml_runs}
prerequisites: []
config_files: []
expected_artifacts: []
```

## Status
draft
## Research question
TODO
## Hypothesis
TODO
## Metrics
TODO
## Controls
TODO
## Success criterion
TODO
## Failure criterion
TODO
## Notes
Crystallized from exploratory runs after those runs were created.
""",
        encoding="utf-8",
    )
    typer.echo(f"Created {path}")


@session_app.command("close")
def session_close(accept: Annotated[bool, typer.Option("--accept")] = False) -> None:
    path = close_session(find_project_root(), accept=accept)
    typer.echo(f"Wrote {path}")


def _staged_paths(root: Path) -> list[str]:
    import subprocess

    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]
