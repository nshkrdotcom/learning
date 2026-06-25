from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from mechledger.alias import resolve_run_id
from mechledger.artifacts import annotate_artifact, register_artifact
from mechledger.core.claim_ledger import parse_claim_ledger
from mechledger.core.decision_log import parse_decision_log
from mechledger.core.experiment_spec import parse_experiment_spec
from mechledger.core.run_ledger import parse_run_ledger
from mechledger.debt_report import generate_scientific_debt_report
from mechledger.draftguard import check_draft_files
from mechledger.formatter import format_project
from mechledger.hooks import install_direct_hook, install_precommit_config
from mechledger.indexer import rebuild_index, validate_project
from mechledger.project import command_cwd, find_project, init_project
from mechledger.run_auditor import capture_run
from mechledger.workflows import (
    append_run_ledger,
    crystallize_experiment,
    decision_new_from_diff,
    propose_claim,
    review_claim,
    session_close,
    waive_debt,
)

ALLOW_EXTRA_ARGS = {"allow_extra_args": True, "ignore_unknown_options": True}

app = typer.Typer(no_args_is_help=True, help="MechLedger research-integrity CLI.")
draft_app = typer.Typer(help="Check tagged draft claims.")
session_app = typer.Typer(help="Close and summarize research sessions.")
artifact_app = typer.Typer(help="Annotate registered artifacts.")
run_ledger_app = typer.Typer(help="Review and append run ledger proposals.")
experiment_app = typer.Typer(help="Validate and crystallize ExperimentSpecs.")
claim_app = typer.Typer(help="Generate and review claim proposals.")
debt_app = typer.Typer(help="Waive visible scientific debt.")
decision_app = typer.Typer(help="Create decision records.")
gate_app = typer.Typer(help="Generate scientific-debt reports.")

app.add_typer(draft_app, name="draft")
app.add_typer(session_app, name="session")
app.add_typer(artifact_app, name="artifact")
app.add_typer(run_ledger_app, name="run-ledger")
app.add_typer(experiment_app, name="experiment")
app.add_typer(claim_app, name="claim")
app.add_typer(debt_app, name="debt")
app.add_typer(decision_app, name="decision")
app.add_typer(gate_app, name="gate")


@app.command()
def init(
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite existing scaffold files.")
    ] = False,
    overwrite_template: Annotated[
        bool, typer.Option("--overwrite-template", help="Overwrite generated templates only.")
    ] = False,
    install_pre_commit: Annotated[
        bool, typer.Option("--install-pre-commit", help="Also install pre-commit config.")
    ] = False,
) -> None:
    try:
        project, messages = init_project(
            command_cwd(), force=force, overwrite_template=overwrite_template
        )
        for message in messages:
            typer.echo(message)
        if install_pre_commit:
            path = install_precommit_config(project.root)
            typer.echo(f"updated {path.relative_to(project.root)}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@draft_app.command("check")
def draft_check(
    files: Annotated[list[Path] | None, typer.Argument(help="Draft files to check.")] = None,
    claim_ledger: Annotated[
        Path | None, typer.Option("--claim-ledger", help="Claim ledger path.")
    ] = None,
    staged: Annotated[bool, typer.Option("--staged", help="Check relevant staged paths.")] = False,
    output_format: Annotated[
        str, typer.Option("--format", help="Output format: text or json.")
    ] = "text",
    warnings_as_errors: Annotated[
        bool, typer.Option("--warnings-as-errors", help="Treat warnings as blocking.")
    ] = False,
    allow_overrides: Annotated[bool, typer.Option("--allow-overrides/--no-allow-overrides")] = True,
) -> None:
    try:
        project = find_project()
        if staged:
            staged_files = _staged_files(project.root)
            relevant = [
                project.root / path
                for path in staged_files
                if path.endswith((".md", ".markdown", ".tex"))
                or path == "research/logs/claim_ledger.md"
                or path == ".mechledger/project.json"
            ]
            if not relevant:
                typer.echo("MechLedger: no staged draft or claim files changed; skipping.")
                raise typer.Exit(0)
            files = [path for path in relevant if path.suffix in {".md", ".markdown", ".tex"}]
        paths = [Path(path) for path in (files or _default_drafts(project))]
        ledger_path = claim_ledger or project.root / project.config.default_claim_ledger
        result = check_draft_files(
            paths, claim_ledger_path=ledger_path, allow_overrides=allow_overrides
        )
        if warnings_as_errors:
            for violation in result.violations:
                if violation.severity.value == "warning" and not violation.suppressed_by_override:
                    violation.severity = type(violation.severity).BLOCKING
        if output_format == "json":
            typer.echo(result.to_json(), nl=False)
        elif output_format == "text":
            typer.echo(result.to_text(), nl=False)
        else:
            _fail("Unknown --format value. Use text or json.", code=2)
        if any(
            item.violation_type in {"unknown_claim", "malformed_claim_tag"}
            for item in result.violations
        ):
            raise typer.Exit(2)
        raise typer.Exit(1 if result.unsuppressed_blocking else 0)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@app.command()
def index(
    check: Annotated[
        bool, typer.Option("--check", help="Validate without rebuilding SQLite.")
    ] = False,
    staged: Annotated[
        bool, typer.Option("--staged", help="Check staged relevant files only.")
    ] = False,
) -> None:
    try:
        project = find_project()
        if staged:
            staged_files = _staged_files(project.root)
            if not any(
                path.startswith("research/") or path == ".mechledger/project.json"
                for path in staged_files
            ):
                typer.echo("MechLedger: no staged research/indexed files changed; skipping.")
                raise typer.Exit(0)
        if check:
            errors, _ = validate_project(project)
            if errors:
                typer.echo("\n\n".join(errors))
                raise typer.Exit(1)
            typer.echo("MechLedger index check passed.")
            raise typer.Exit(0)
        db_path, errors = rebuild_index(project)
        if errors:
            typer.echo("\n\n".join(errors))
            raise typer.Exit(1)
        typer.echo(f"Rebuilt index: {db_path}")
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@app.command("format")
def format_command(
    write: Annotated[bool, typer.Option("--write", help="Apply safe formatting changes.")] = False,
) -> None:
    try:
        project = find_project()
        changed, diff = format_project(project, write=write)
        if diff:
            typer.echo(diff, nl=False)
        if changed and not write:
            raise typer.Exit(1)
        typer.echo("MechLedger format complete.")
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@app.command("install-hooks")
def install_hooks(
    direct: Annotated[
        bool, typer.Option("--direct", help="Install local .git/hooks hook.")
    ] = False,
) -> None:
    try:
        project = find_project(allow_uninitialized=True)
        if direct:
            path = install_direct_hook(project.root)
            typer.echo(f"Installed direct Git hook into {path.relative_to(project.root)}.")
            typer.echo(
                "This hook is not committed and must be installed separately by each researcher."
            )
            typer.echo("For portable hooks, use the default pre-commit framework mode.")
        else:
            path = install_precommit_config(project.root)
            typer.echo(f"Updated {path.relative_to(project.root)}.")
            typer.echo("Run `pre-commit install` to activate the hooks.")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@session_app.command("close")
def session_close_command(
    accept: Annotated[
        bool, typer.Option("--accept", help="Append the draft to research_log.md.")
    ] = False,
    since: Annotated[str | None, typer.Option("--since", help="Session start timestamp.")] = None,
) -> None:
    try:
        project = find_project()
        path = session_close(project, accept=accept, since=since)
        if accept:
            typer.echo("Accepted session close and updated research log.")
        else:
            typer.echo(f"Wrote session draft: {path.relative_to(project.root)}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@session_app.command("cleanup")
def session_cleanup() -> None:
    try:
        project = find_project()
        drafts = sorted((project.mechledger_dir / "session_drafts").glob("*.md"))
        typer.echo(f"Retained {len(drafts)} abandoned session drafts.")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@app.command(context_settings=ALLOW_EXTRA_ARGS)
def run(
    ctx: typer.Context,
    experiment: Annotated[str | None, typer.Option("--experiment")] = None,
    run_class: Annotated[str, typer.Option("--class")] = "scratch",
    purpose: Annotated[str | None, typer.Option("--purpose")] = None,
    hypothesis: Annotated[str | None, typer.Option("--hypothesis")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    hook_point: Annotated[str | None, typer.Option("--hook-point")] = None,
    sae_release: Annotated[str | None, typer.Option("--sae-release")] = None,
    sae_id: Annotated[str | None, typer.Option("--sae-id")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
) -> None:
    try:
        project = find_project()
        argv = list(ctx.args)
        if argv and argv[0] == "--":
            argv = argv[1:]
        run_id_out, exit_code = capture_run(
            project,
            argv,
            experiment_id=experiment,
            run_class=run_class,
            purpose=purpose,
            hypothesis=hypothesis,
            run_id=run_id,
            metadata={
                "model": model,
                "hook_point": hook_point,
                "sae_release": sae_release,
                "sae_id": sae_id,
                "seed": seed,
            },
        )
        run_dir = project.runs_dir / run_id_out
        status = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["status"]
        typer.echo(f"Created run: {run_id_out}")
        typer.echo(f"Directory: {run_dir.relative_to(project.root)}/")
        typer.echo(f"Status: {status}")
        typer.echo(
            f"Run ledger proposal: {(run_dir / 'run_ledger_row.csv').relative_to(project.root)}"
        )
        debt_report = (run_dir / "scientific_debt_report.md").relative_to(project.root)
        typer.echo(f"Scientific debt report: {debt_report}")
        manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("artifacts"):
            typer.echo("Run captured. No artifacts were registered or auto-collected.")
            typer.echo(
                f"Use `mechledger attach {run_id_out} PATH`, write outputs to "
                f"`{run_dir.relative_to(project.root)}/artifacts/`, "
                "or call `run.log_artifact(...)`."
            )
        typer.echo("\nNext:")
        typer.echo(f"  mechledger attach {run_id_out} <artifact>")
        typer.echo(f"  mechledger artifact annotate {run_id_out} <artifact-id>")
        typer.echo(f"  mechledger run-ledger append {run_id_out}")
        typer.echo(f"  mechledger claim propose --run {run_id_out}")
        typer.echo(f"  mechledger claim review {run_id_out}")
        raise typer.Exit(exit_code if exit_code else 0)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@app.command()
def attach(
    run_id: str,
    path: Path,
    claim_relevance: Annotated[
        str,
        typer.Option("--claim-relevance", help="none|diagnostic|supporting|contradicting|required"),
    ] = "none",
    artifact_type: Annotated[str | None, typer.Option("--type")] = None,
    description: Annotated[str | None, typer.Option("--description")] = None,
    allow_missing: Annotated[bool, typer.Option("--allow-missing")] = False,
) -> None:
    try:
        project = find_project()
        canonical = resolve_run_id(project, run_id)
        artifact = register_artifact(
            project,
            canonical,
            path,
            claim_relevance=claim_relevance,
            description=description or artifact_type,
            allow_missing=allow_missing,
        )
        typer.echo(f"Attached {artifact['artifact_id']} to {canonical}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@artifact_app.command("annotate")
def artifact_annotate(
    run_id: str,
    artifact_id: str,
    claim_relevance: Annotated[str, typer.Option("--claim-relevance")],
    description: Annotated[str | None, typer.Option("--description")] = None,
) -> None:
    try:
        project = find_project()
        canonical = resolve_run_id(project, run_id)
        artifact = annotate_artifact(
            project,
            canonical,
            artifact_id,
            claim_relevance=claim_relevance,
            description=description,
        )
        typer.echo(f"Annotated {artifact['artifact_id']} on {canonical}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@run_ledger_app.command("append")
def run_ledger_append(
    run_id: str,
    yes: Annotated[
        bool, typer.Option("--yes", help="Append without interactive confirmation.")
    ] = False,
) -> None:
    try:
        project = find_project()
        canonical = append_run_ledger(project, run_id, yes=yes)
        typer.echo(f"Appended run ledger row for {canonical}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@experiment_app.command("validate")
def experiment_validate(paths: list[Path]) -> None:
    try:
        for path in paths:
            spec = parse_experiment_spec(path)
            typer.echo(f"valid {spec.experiment_id}: {path}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@experiment_app.command("crystallize")
def experiment_crystallize(
    runs: Annotated[list[str], typer.Option("--runs")],
    experiment_id: Annotated[str, typer.Option("--id")],
    title: Annotated[str, typer.Option("--title")],
) -> None:
    try:
        project = find_project()
        path = crystallize_experiment(project, runs, experiment_id, title)
        typer.echo(f"Created {path.relative_to(project.root)}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@claim_app.command("propose")
def claim_propose(
    run_id: Annotated[str, typer.Option("--run")],
    regenerate: Annotated[bool, typer.Option("--regenerate")] = False,
) -> None:
    try:
        project = find_project()
        path = propose_claim(project, run_id, regenerate=regenerate)
        typer.echo(f"Claim proposal: {path.relative_to(project.root)}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@claim_app.command("review")
def claim_review(
    run_id: str,
    apply: Annotated[bool, typer.Option("--apply")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
) -> None:
    try:
        project = find_project()
        state = review_claim(project, run_id, apply=apply, yes=yes)
        typer.echo(f"Claim proposal state: {state}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@debt_app.command("waive")
def debt_waive(debt_id: str, decision: Annotated[str, typer.Option("--decision")]) -> None:
    try:
        project = find_project()
        path = waive_debt(project, debt_id, decision)
        typer.echo(f"Waived {debt_id} in {path.relative_to(project.root)}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@decision_app.command("new")
def decision_new(
    from_diff: Annotated[bool, typer.Option("--from-diff")] = False,
) -> None:
    try:
        project = find_project()
        if not from_diff:
            _fail("Only `decision new --from-diff` is implemented in this pass.", code=3)
        path = decision_new_from_diff(project)
        typer.echo(f"Appended proposed decision to {path.relative_to(project.root)}")
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@gate_app.command("check")
def gate_check(run_id: str) -> None:
    try:
        project = find_project()
        canonical = resolve_run_id(project, run_id)
        report = generate_scientific_debt_report(project, canonical)
        typer.echo(report.summary)
        raise typer.Exit(1 if report.blockers else 0)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@app.command()
def status() -> None:
    try:
        project = find_project()
        errors, data = validate_project(project)
        if errors:
            typer.echo("\n\n".join(errors))
            raise typer.Exit(2)
        claim_counts: dict[str, int] = {}
        for claim in data["claims"].values():
            claim_counts[claim.status.value] = claim_counts.get(claim.status.value, 0) + 1
        debt_counts = _debt_counts(project)
        typer.echo(f"Project: {project.root.name}")
        typer.echo(f"Root: {project.root}")
        typer.echo(f"Schema: {project.config.schema_version}")
        typer.echo("Index: disposable")
        typer.echo("\nClaims:")
        for status_name, count in sorted(claim_counts.items()):
            typer.echo(f"  {status_name}: {count}")
        typer.echo("\nExperiments:")
        typer.echo(f"  total: {len(data['experiments'])}")
        typer.echo("\nSince last session close:")
        typer.echo(f"  runs: {len(data['local_runs'])}")
        typer.echo("\nScientific Debt:")
        for severity, count in sorted(debt_counts.items()):
            typer.echo(f"  {severity}: {count}")
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@app.command("next")
def next_command() -> None:
    try:
        project = find_project()
        decisions = parse_decision_log(project.root / project.config.default_decision_log)
        claims = parse_claim_ledger(project.root / project.config.default_claim_ledger)
        run_ledger = parse_run_ledger(project.root / project.config.default_run_ledger)
        completed_experiments = {
            row.get("phase") for row in run_ledger.rows if row.get("status") == "completed"
        }
        ready: list[str] = []
        blocked: list[str] = []
        for path in sorted((project.root / "research/experiments").glob("*.md")):
            if path.name.startswith("TEMPLATE_"):
                continue
            spec = parse_experiment_spec(path)
            blockers = _experiment_blockers(
                spec, decisions.decisions, claims.claims, completed_experiments, project
            )
            if blockers:
                blocked.append(f"{spec.experiment_id} - {spec.title}: {', '.join(blockers)}")
            elif spec.status in {"planned", "draft"}:
                ready.append(f"{spec.experiment_id} - {spec.title}")
        if ready:
            typer.echo("READY")
            for item in ready:
                typer.echo(f"  {item}")
        if blocked:
            typer.echo("BLOCKED")
            for item in blocked:
                typer.echo(f"  {item}")
        raise typer.Exit(0 if ready or not blocked else 1)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


def _default_drafts(project) -> list[Path]:
    root = project.root / "research/paper"
    paths = []
    for pattern in ("**/*.md", "**/*.markdown", "**/*.tex"):
        paths.extend(sorted(root.glob(pattern)))
    return paths


def _staged_files(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return result.stdout.splitlines()


def _debt_counts(project) -> dict[str, int]:
    counts: dict[str, int] = {}
    for report_path in project.runs_dir.glob("*/scientific_debt_report.json"):
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        for debt in payload.get("debts", []):
            if debt.get("status") == "open":
                severity = debt.get("severity", "unknown")
                counts[severity] = counts.get(severity, 0) + 1
    return counts


def _experiment_blockers(spec, decisions, claims, completed_experiments, project) -> list[str]:
    blockers: list[str] = []
    for prereq in spec.prerequisites:
        kind = prereq.get("type")
        if kind == "decision_accepted":
            decision = decisions.get(prereq.get("id"))
            if decision is None or decision.status != "accepted":
                blockers.append(f"{prereq.get('id')} accepted")
        elif kind in {"experiment_completed", "experiment_completed_and_reviewed"}:
            if prereq.get("id") not in completed_experiments:
                blockers.append(f"{prereq.get('id')} completed")
        elif kind == "claim_status_at_least":
            claim = claims.get(prereq.get("id"))
            if claim is None:
                blockers.append(f"{prereq.get('id')} exists")
        elif kind == "artifact_exists":
            if not (project.root / str(prereq.get("path"))).exists():
                blockers.append(f"{prereq.get('path')} exists")
    return blockers


def _fail(message: str, *, code: int) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code)
