from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from ruamel.yaml import YAML

from mechledger.alias import resolve_run_id
from mechledger.artifacts import annotate_artifact, register_artifact, resolve_artifact_path
from mechledger.copilot import (
    accept_copilot_output,
    find_copilot_output,
    list_copilot_outputs,
    reject_copilot_output,
)
from mechledger.core.experiment_spec import parse_experiment_spec
from mechledger.dashboard_data import (
    filter_rows,
    query_rows,
    rows_json,
    rows_text,
    write_dashboard_data,
)
from mechledger.debt_report import generate_scientific_debt_report
from mechledger.draftguard import check_draft_files
from mechledger.export import write_appendix, write_bundle, write_ro_crate
from mechledger.external_labels import (
    import_labels,
    link_label_to_claim,
    read_labels,
    show_label,
)
from mechledger.external_labels import (
    validate_file as validate_labels_file,
)
from mechledger.formatter import format_project
from mechledger.hooks import install_direct_hook, install_precommit_config
from mechledger.indexer import rebuild_index, validate_project
from mechledger.integrity import (
    check_integrity,
    resolve_tamper_record,
    unresolved_tamper_count,
)
from mechledger.language_report import (
    claim_language_report,
    write_claim_language_report,
    write_draft_suggestions,
)
from mechledger.lifecycle import garbage_collect, pin_run, write_run_bundle
from mechledger.open_questions import (
    add_question,
    list_questions,
    resolve_question,
    show_question,
)
from mechledger.prediction import (
    PredictionInputError,
    PredictionStateError,
    find_prediction_by_id,
    load_prediction,
    lock_prediction,
    score_prediction_file,
)
from mechledger.prerequisites import (
    evaluate_experiment_prerequisites,
    load_prerequisite_context,
)
from mechledger.project import Project, command_cwd, find_project, init_project
from mechledger.records import list_records, show_record, validate_record
from mechledger.redaction import redact_artifact, redact_run
from mechledger.run_auditor import capture_run
from mechledger.sessions import (
    add_session_note,
    attach_session_path,
    close_session_record,
    list_sessions,
    load_session,
    review_session,
    session_path,
    start_session,
)
from mechledger.sync_status import (
    blocking_findings,
    evaluate_sync,
    format_sync_diff,
    format_sync_status,
)
from mechledger.tier2 import (
    append_metric,
    evaluate_filtered_report,
    has_registered_artifact_path,
    load_paired_test_result,
    paired_test_markdown,
    write_tier2_check_report,
)
from mechledger.workflows import (
    append_run_ledger,
    crystallize_experiment,
    decision_new_from_declared_surfaces,
    decision_new_from_diff,
    propose_claim,
    reclassify_run,
    review_claim,
    session_close,
    waive_debt,
)

ALLOW_EXTRA_ARGS = {"allow_extra_args": True, "ignore_unknown_options": True}
RUN_EXTRA_ARGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}

app = typer.Typer(no_args_is_help=True, help="MechLedger research-integrity CLI.")
draft_app = typer.Typer(help="Check tagged draft claims.")
session_app = typer.Typer(help="Record, close, and review research sessions.")
artifact_app = typer.Typer(help="Annotate registered artifacts.")
run_ledger_app = typer.Typer(help="Review and append run ledger proposals.")
experiment_app = typer.Typer(help="Validate and crystallize ExperimentSpecs.")
claim_app = typer.Typer(help="Generate and review claim proposals.")
debt_app = typer.Typer(help="Waive visible scientific debt.")
decision_app = typer.Typer(help="Create decision records.")
gate_app = typer.Typer(help="Assess evidence and generate scientific-debt reports.")
calibration_app = typer.Typer(help="Check calibration and positive-control evidence.")
telemetry_app = typer.Typer(help="Check intervention telemetry metrics.")
null_app = typer.Typer(help="Plan or register empirical-null evidence.")
stats_app = typer.Typer(help="Register lightweight statistical evidence.")
prediction_app = typer.Typer(help="Lock and score explainer predictions.")
export_app = typer.Typer(help="Export archival metadata, bundles, and appendices.")
questions_app = typer.Typer(help="Track open research questions.")
labels_app = typer.Typer(help="Import and inspect external label metadata.")
dashboard_app = typer.Typer(help="Write local dashboard data.")
query_app = typer.Typer(help="Inspect canonical MechLedger records locally.")
records_app = typer.Typer(help="Validate optional platform metadata records.")
sync_app = typer.Typer(help="Report local run and ledger sync drift.")
integrity_app = typer.Typer(help="Check and resolve local tamper/staleness records.")
copilot_app = typer.Typer(help="Review local assistant outputs and write provenance.")

app.add_typer(draft_app, name="draft")
app.add_typer(session_app, name="session")
app.add_typer(artifact_app, name="artifact")
app.add_typer(run_ledger_app, name="run-ledger")
app.add_typer(experiment_app, name="experiment")
app.add_typer(claim_app, name="claim")
app.add_typer(debt_app, name="debt")
app.add_typer(decision_app, name="decision")
app.add_typer(gate_app, name="gate")
app.add_typer(calibration_app, name="calibration")
app.add_typer(telemetry_app, name="telemetry")
app.add_typer(null_app, name="null")
app.add_typer(stats_app, name="stats")
app.add_typer(prediction_app, name="prediction")
app.add_typer(export_app, name="export")
app.add_typer(questions_app, name="questions")
app.add_typer(labels_app, name="labels")
app.add_typer(dashboard_app, name="dashboard")
app.add_typer(query_app, name="query")
app.add_typer(records_app, name="records")
app.add_typer(sync_app, name="sync")
app.add_typer(integrity_app, name="integrity")
app.add_typer(copilot_app, name="copilot")


def _project_output_path(project: Project, path: Path) -> Path:
    return path if path.is_absolute() else project.root / path


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


@draft_app.command("suggest")
def draft_suggest(
    files: Annotated[list[Path], typer.Argument(help="Draft files to inspect.")],
    out: Annotated[Path, typer.Option("--out", help="Markdown report path.")],
) -> None:
    try:
        project = find_project()
        path = write_draft_suggestions(project, files, out)
        typer.echo(f"draft_suggestions: {path.relative_to(project.root)}")
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
    check: Annotated[
        bool, typer.Option("--check", help="Check formatting without modifying files.")
    ] = False,
) -> None:
    try:
        if write and check:
            _fail("Use either --write or --check, not both.", code=2)
        project = find_project()
        changed, diff = format_project(project, write=write and not check)
        if diff:
            typer.echo(diff, nl=False)
        if check:
            if changed:
                typer.echo("MechLedger format check failed.")
                raise typer.Exit(1)
            typer.echo("MechLedger format check passed.")
            return
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


@session_app.command("start")
def session_start_command(
    title: Annotated[str, typer.Option("--title", help="Human-readable session title.")],
) -> None:
    try:
        project = find_project()
        record = start_session(project, title)
        typer.echo(f"session_id: {record['session_id']}")
        typer.echo(
            f"session_path: {session_path(project, record['session_id']).relative_to(project.root)}"
        )
        typer.echo(f"status: {record['status']}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@session_app.command("note")
def session_note_command(
    session: Annotated[str, typer.Option("--session")],
    text: Annotated[str, typer.Option("--text")],
) -> None:
    try:
        project = find_project()
        record = add_session_note(project, session, text)
        typer.echo(f"session_id: {record['session_id']}")
        typer.echo(f"notes: {len(record['notes'])}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@session_app.command("attach")
def session_attach_command(
    path: Path,
    session: Annotated[str, typer.Option("--session")],
) -> None:
    try:
        project = find_project()
        record = attach_session_path(project, session, path)
        attachment = record["attached_paths"][-1]
        typer.echo(f"session_id: {record['session_id']}")
        typer.echo(f"attached_path: {attachment['path']}")
        typer.echo(f"sha256: {attachment.get('sha256') or 'missing'}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@session_app.command("close")
def session_close_command(
    session: Annotated[
        str | None,
        typer.Option("--session", help="Close a local auditable session record."),
    ] = None,
    accept: Annotated[
        bool, typer.Option("--accept", help="Append the draft to research_log.md.")
    ] = False,
    since: Annotated[str | None, typer.Option("--since", help="Session start timestamp.")] = None,
) -> None:
    try:
        project = find_project()
        if session:
            record = close_session_record(project, session)
            typer.echo(f"session_id: {record['session_id']}")
            typer.echo(f"status: {record['status']}")
            summary_path = session_path(project, session).parent / "summary.md"
            typer.echo(f"summary: {summary_path.relative_to(project.root)}")
            return
        path = session_close(project, accept=accept, since=since)
        if accept:
            typer.echo("Accepted session close and updated research log.")
        else:
            typer.echo(f"Wrote session draft: {path.relative_to(project.root)}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@session_app.command("review")
def session_review_command(
    session: Annotated[str, typer.Option("--session")],
    accept: Annotated[bool, typer.Option("--accept")] = False,
    reject: Annotated[bool, typer.Option("--reject")] = False,
    decision: Annotated[str | None, typer.Option("--decision")] = None,
) -> None:
    try:
        project = find_project()
        record = review_session(
            project,
            session,
            accept=accept,
            reject=reject,
            decision_id=decision,
        )
        typer.echo(f"session_id: {record['session_id']}")
        typer.echo(f"status: {record['status']}")
        typer.echo(f"review_decision_id: {record.get('review_decision_id') or 'none'}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@session_app.command("list")
def session_list_command() -> None:
    try:
        project = find_project()
        for record in list_sessions(project):
            typer.echo(f"{record['session_id']}\t{record['status']}\t{record['title']}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@session_app.command("show")
def session_show_command(session_id: str) -> None:
    try:
        project = find_project()
        typer.echo(json.dumps(load_session(project, session_id), indent=2, sort_keys=True))
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


@export_app.command("ro-crate")
def export_ro_crate(
    out: Annotated[Path, typer.Option("--out", help="Output RO-Crate directory.")],
) -> None:
    try:
        project = find_project()
        path, warnings = write_ro_crate(project, _project_output_path(project, out))
        typer.echo(f"ro_crate_metadata: {path.relative_to(project.root)}")
        for warning in warnings:
            typer.echo(f"warning: {warning}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@export_app.command("bundle")
def export_bundle(
    out: Annotated[Path, typer.Option("--out", help="Output .tar.gz/.tar.zst or manifest.")],
    run: Annotated[list[str] | None, typer.Option("--run")] = None,
    include_local_runs: Annotated[
        bool, typer.Option("--include-local-runs/--no-include-local-runs")
    ] = False,
    include_artifacts: Annotated[
        bool, typer.Option("--include-artifacts/--no-include-artifacts")
    ] = False,
    redact_env: Annotated[bool, typer.Option("--redact-env/--no-redact-env")] = True,
    manifest_only: Annotated[bool, typer.Option("--manifest-only")] = False,
) -> None:
    try:
        project = find_project()
        path, manifest = write_bundle(
            project,
            _project_output_path(project, out),
            run_aliases=run or [],
            include_local_runs=include_local_runs,
            include_artifacts=include_artifacts,
            redact_env=redact_env,
            manifest_only=manifest_only,
        )
        typer.echo(f"bundle: {path.relative_to(project.root)}")
        typer.echo(f"included_run_ids: {', '.join(manifest['included_run_ids']) or 'none'}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@export_app.command("appendix")
def export_appendix(
    out: Annotated[Path, typer.Option("--out", help="Output Markdown appendix.")],
    claim: Annotated[list[str] | None, typer.Option("--claim")] = None,
    run: Annotated[list[str] | None, typer.Option("--run")] = None,
    include_debt: Annotated[bool, typer.Option("--include-debt")] = False,
    include_decisions: Annotated[bool, typer.Option("--include-decisions")] = False,
    include_artifacts: Annotated[bool, typer.Option("--include-artifacts")] = False,
    output_format: Annotated[str, typer.Option("--format")] = "markdown",
) -> None:
    try:
        if output_format != "markdown":
            _fail("Only --format markdown is supported.", code=2)
        project = find_project()
        path = write_appendix(
            project,
            _project_output_path(project, out),
            claims=claim or [],
            runs=run or [],
            include_debt=include_debt,
            include_decisions=include_decisions,
            include_artifacts=include_artifacts,
        )
        typer.echo(f"appendix: {path.relative_to(project.root)}")
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@sync_app.command("status")
def sync_status_command() -> None:
    try:
        project = find_project()
        findings = evaluate_sync(project)
        typer.echo(format_sync_status(findings), nl=False)
        raise typer.Exit(1 if blocking_findings(findings) else 0)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@sync_app.command("diff")
def sync_diff_command() -> None:
    try:
        project = find_project()
        findings = evaluate_sync(project)
        typer.echo(format_sync_diff(findings), nl=False)
        raise typer.Exit(1 if blocking_findings(findings) else 0)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@app.command("redact")
def redact_command(
    target: str,
    path: Annotated[Path | None, typer.Argument()] = None,
    reason: Annotated[str, typer.Option("--reason")] = "",
    run: Annotated[str | None, typer.Option("--run")] = None,
) -> None:
    try:
        project = find_project()
        if target == "artifact":
            if path is None:
                _fail("Usage: mechledger redact artifact PATH --reason TEXT", code=2)
            record, action = redact_artifact(project, path, reason=reason, run_alias=run)
        else:
            if path is not None:
                _fail("Usage: mechledger redact RUN_ID --reason TEXT", code=2)
            record, action = redact_run(project, target, reason=reason)
        typer.echo(f"redaction_id: {record.redaction_id}")
        typer.echo(f"target_type: {record.target_type}")
        typer.echo(f"target_path: {record.target_path}")
        typer.echo(f"placeholder_path: {record.placeholder_path or 'none'}")
        typer.echo(f"action: {action}")
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@integrity_app.command("check")
def integrity_check_command(
    run: Annotated[str | None, typer.Option("--run")] = None,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        project = find_project()
        unresolved = check_integrity(project, run_alias=run)
        if output_json:
            typer.echo(
                json.dumps(
                    [record.model_dump(mode="json") for record in unresolved],
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            typer.echo(f"unresolved_tamper_records: {len(unresolved)}")
            for record in unresolved:
                typer.echo(
                    f"{record.tamper_id}\t{record.object_type}\t{record.object_id}\t"
                    f"{record.consequence}"
                )
        raise typer.Exit(1 if unresolved else 0)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@integrity_app.command("resolve")
def integrity_resolve_command(
    tamper_id: str,
    decision: Annotated[str, typer.Option("--decision")],
    status: Annotated[str, typer.Option("--status")] = "accepted_as_new_version",
    note: Annotated[str, typer.Option("--note")] = "",
) -> None:
    try:
        project = find_project()
        record = resolve_tamper_record(
            project,
            tamper_id,
            decision_id=decision,
            status=status,
            note=note,
        )
        typer.echo(f"tamper_id: {record.tamper_id}")
        typer.echo(f"resolution_status: {record.resolution_status}")
        typer.echo(f"resolution_decision_id: {record.resolution_decision_id}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@copilot_app.command("list")
def copilot_list_command(
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        project = find_project()
        outputs = list_copilot_outputs(project)
        if output_json:
            typer.echo(
                json.dumps(
                    [item.list_record(project) for item in outputs],
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        if not outputs:
            typer.echo("No copilot outputs found.")
            return
        for item in outputs:
            output = item.output
            review_outcome = output.review_outcome.value if output.review_outcome else "pending"
            typer.echo(
                f"{output.output_id}\t{output.session_id}\t{output.output_type.value}\t"
                f"review={review_outcome}\t"
                f"generated_exists={item.generated_artifact_exists}\t"
                f"prompt_exists={item.prompt_artifact_exists}"
            )
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@copilot_app.command("show")
def copilot_show_command(
    output_id: str,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        project = find_project()
        item = find_copilot_output(project, output_id)
        output = item.output
        if output_json:
            typer.echo(json.dumps(output.model_dump(mode="json"), indent=2, sort_keys=True))
            return
        review_outcome = output.review_outcome.value if output.review_outcome else "pending"
        typer.echo(f"output_id: {output.output_id}")
        typer.echo(f"session_id: {output.session_id}")
        typer.echo(f"output_type: {output.output_type.value}")
        typer.echo(f"model: {output.model}")
        typer.echo(f"human_reviewed: {output.human_reviewed}")
        typer.echo(f"review_outcome: {review_outcome}")
        typer.echo(f"generated_artifact_path: {output.generated_artifact_path}")
        typer.echo(f"generated_artifact_exists: {item.generated_artifact_exists}")
        typer.echo(f"prompt_artifact_path: {output.prompt_artifact_path}")
        typer.echo(f"prompt_artifact_exists: {item.prompt_artifact_exists}")
        typer.echo(f"accepted_artifact_path: {output.accepted_artifact_path or 'none'}")
        typer.echo(f"accepted_provenance_path: {output.accepted_provenance_path or 'none'}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@copilot_app.command("review")
def copilot_review_command(
    output_id: str,
    accept: Annotated[bool, typer.Option("--accept")] = False,
    reject: Annotated[bool, typer.Option("--reject")] = False,
    modified: Annotated[Path | None, typer.Option("--modified")] = None,
    destination: Annotated[Path | None, typer.Option("--to")] = None,
) -> None:
    try:
        project = find_project()
        selected = sum(1 for enabled in [accept, reject, modified is not None] if enabled)
        if selected != 1:
            _fail("Use exactly one of --accept, --reject, or --modified PATH.", code=2)
        if reject:
            output = reject_copilot_output(project, output_id)
            typer.echo(f"output_id: {output.output_id}")
            typer.echo("review_outcome: rejected")
            typer.echo("accepted_artifact_path: none")
            return
        if destination is None:
            _fail("Accepting or modifying copilot output requires --to PATH.", code=2)
        output, provenance = accept_copilot_output(
            project,
            output_id,
            destination=destination,
            modified_path=modified,
        )
        review_outcome = output.review_outcome.value if output.review_outcome else "pending"
        typer.echo(f"output_id: {output.output_id}")
        typer.echo(f"review_outcome: {review_outcome}")
        typer.echo(f"accepted_artifact_path: {output.accepted_artifact_path}")
        typer.echo(f"accepted_provenance_path: {output.accepted_provenance_path}")
        typer.echo(f"accepted_artifact_hash: {provenance.accepted_artifact_hash}")
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@app.command("pin")
def pin_command(run_id: str) -> None:
    try:
        project = find_project()
        canonical, changed = pin_run(project, run_id)
        typer.echo(f"run_id: {canonical}")
        typer.echo("pinned: true")
        typer.echo(f"action: {'pinned' if changed else 'already_pinned'}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@app.command("gc")
def gc_command(
    keep_last: Annotated[int, typer.Option("--keep-last")] = 100,
    keep_pinned: Annotated[bool, typer.Option("--keep-pinned/--no-keep-pinned")] = True,
    archive: Annotated[Path | None, typer.Option("--archive")] = None,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    allow_remove_all_unpinned: Annotated[
        bool, typer.Option("--allow-remove-all-unpinned")
    ] = False,
) -> None:
    try:
        project = find_project()
        manifest = garbage_collect(
            project,
            keep_last=keep_last,
            keep_pinned=keep_pinned,
            archive_dir=archive,
            yes=yes,
            allow_remove_all_unpinned=allow_remove_all_unpinned,
        )
        typer.echo(f"dry_run: {str(manifest['dry_run']).lower()}")
        typer.echo(
            "planned_remove_run_ids: "
            + (", ".join(manifest["planned_remove_run_ids"]) or "none")
        )
        typer.echo(
            "removed_run_ids: " + (", ".join(manifest["removed_run_ids"]) or "none")
        )
        typer.echo(
            "archived_run_ids: " + (", ".join(manifest["archived_run_ids"]) or "none")
        )
        typer.echo("gc_manifest: .mechledger/gc_manifest.json")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@app.command("bundle")
def bundle_command(
    run_id: str,
    out: Annotated[Path, typer.Option("--out")],
) -> None:
    try:
        project = find_project()
        path, manifest = write_run_bundle(project, run_id, out)
        typer.echo(f"run_id: {manifest['run_id']}")
        typer.echo(f"bundle: {path.relative_to(project.root)}")
        typer.echo(f"files: {len(manifest['files'])}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@app.command(context_settings=RUN_EXTRA_ARGS)
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
        argv = list(ctx.args)
        if argv and argv[0] == "--":
            argv = argv[1:]
        if argv == ["--help"]:
            _print_run_help()
            raise typer.Exit(0)
        if argv and argv[0] == "reclassify" and "--help" in argv:
            _handle_run_reclassify_help()
            raise typer.Exit(0)
        project = find_project()
        if _looks_like_run_reclassify(argv):
            canonical = _handle_run_reclassify(project, argv)
            to_class = _option_value(argv, "--to") or ""
            typer.echo(f"Reclassified run {canonical} to {to_class}.")
            typer.echo("Regenerated scientific debt report.")
            return
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
            artifact_type=artifact_type,
            claim_relevance=claim_relevance,
            description=description,
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
        project = find_project()
        context = load_prerequisite_context(project)
        had_blockers = False
        had_debt = False
        for path in paths:
            spec = parse_experiment_spec(path)
            evaluation = evaluate_experiment_prerequisites(spec, context)
            for finding in evaluation.findings:
                typer.echo(finding.format())
            if evaluation.input_errors:
                raise typer.Exit(2)
            had_blockers = had_blockers or bool(evaluation.blockers)
            had_debt = had_debt or bool(evaluation.debt_or_warnings)
            if evaluation.is_clean:
                typer.echo(f"valid {spec.experiment_id}: {path}")
        if had_blockers or had_debt:
            raise typer.Exit(1)
    except typer.Exit:
        raise
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
    force_stale: Annotated[bool, typer.Option("--force-stale")] = False,
) -> None:
    try:
        project = find_project()
        state = review_claim(project, run_id, apply=apply, yes=yes, force_stale=force_stale)
        typer.echo(f"Claim proposal state: {state}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@claim_app.command("language-report")
def claim_language_report_command(
    claim: Annotated[list[str] | None, typer.Option("--claim")] = None,
    all_claims: Annotated[bool, typer.Option("--all")] = False,
    out: Annotated[Path | None, typer.Option("--out")] = None,
) -> None:
    try:
        project = find_project()
        if out:
            path = write_claim_language_report(
                project,
                out,
                claim_ids=claim or [],
                all_claims=all_claims,
            )
            typer.echo(f"claim_language_report: {path.relative_to(project.root)}")
            return
        typer.echo(
            claim_language_report(
                project,
                claim_ids=claim or [],
                all_claims=all_claims,
            ),
            nl=False,
        )
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
    from_declared_surfaces: Annotated[bool, typer.Option("--from-declared-surfaces")] = False,
) -> None:
    try:
        project = find_project()
        if from_declared_surfaces:
            path = decision_new_from_declared_surfaces(project)
        elif from_diff:
            path = decision_new_from_diff(project)
        else:
            _fail(
                "Use `decision new --from-diff` or `decision new --from-declared-surfaces`.",
                code=3,
            )
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
        run_dir = project.runs_dir / canonical
        evidence_path = run_dir / "evidence_assessment.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        open_debts = [
            debt for debt in report.debts if getattr(debt, "status", None) == "open"
        ]
        blocking = [debt for debt in open_debts if getattr(debt, "severity", None) == "blocking"]
        typer.echo(f"run_id: {canonical}")
        typer.echo(f"recommended_claim_status: {evidence['recommended_claim_status']}")
        typer.echo(f"clean_candidate_support: {str(report.clean_candidate_support).lower()}")
        typer.echo(
            "open_debt: "
            + (
                ", ".join(f"{debt.debt_id}/{debt.debt_type}" for debt in open_debts)
                if open_debts
                else "none"
            )
        )
        typer.echo(
            "blocking_findings: "
            + (
                ", ".join(f"{debt.debt_id}/{debt.debt_type}" for debt in blocking)
                if blocking
                else "none"
            )
        )
        evidence_rel = (run_dir / "evidence_assessment.json").relative_to(project.root)
        typer.echo(f"evidence_assessment: {evidence_rel}")
        typer.echo(
            f"scientific_debt_report: "
            f"{(run_dir / 'scientific_debt_report.json').relative_to(project.root)}"
        )
        raise typer.Exit(1 if report.blockers or not report.clean_candidate_support else 0)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@calibration_app.command("check")
def calibration_check(run_id: str) -> None:
    try:
        project = find_project()
        canonical = resolve_run_id(project, run_id)
        run_dir = project.runs_dir / canonical
        report = evaluate_filtered_report(project, canonical, "calibration")
        write_tier2_check_report(run_dir, "calibration_check", report)
        _echo_tier2_summary("calibration_check", report, project, run_dir)
        raise typer.Exit(1 if report.blocking_debts else 0)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@telemetry_app.command("check")
def telemetry_check(run_id: str) -> None:
    try:
        project = find_project()
        canonical = resolve_run_id(project, run_id)
        run_dir = project.runs_dir / canonical
        report = evaluate_filtered_report(project, canonical, "telemetry")
        write_tier2_check_report(run_dir, "telemetry_check", report)
        _echo_tier2_summary("telemetry_check", report, project, run_dir)
        raise typer.Exit(1 if report.blocking_debts else 0)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@null_app.command("run")
def null_run(
    plan: Annotated[bool, typer.Option("--plan", help="Write an empirical-null plan.")] = False,
    register: Annotated[
        str | None, typer.Option("--register", help="Register null evidence for RUN_ID.")
    ] = None,
    experiment: Annotated[str | None, typer.Option("--experiment")] = None,
    feature_set_size: Annotated[int | None, typer.Option("--feature-set-size")] = None,
    seeds: Annotated[int | None, typer.Option("--seeds")] = None,
    sampling: Annotated[str, typer.Option("--sampling")] = "random_feature_sets",
    exclude_feature_ids: Annotated[
        list[str] | None, typer.Option("--exclude-feature-id")
    ] = None,
    output_metric: Annotated[str, typer.Option("--output-metric")] = "specificity_gap",
    planned_output_artifact: Annotated[
        str | None, typer.Option("--planned-output-artifact")
    ] = None,
    null_distribution: Annotated[
        Path | None, typer.Option("--null-distribution")
    ] = None,
    metric: Annotated[str | None, typer.Option("--metric")] = None,
    seed_count: Annotated[int | None, typer.Option("--seed-count")] = None,
    percentile_rank: Annotated[
        float | None, typer.Option("--percentile-rank")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite or re-register.")] = False,
) -> None:
    try:
        if plan == bool(register):
            _fail("Use exactly one null mode: --plan or --register RUN_ID.", code=2)
        project = find_project()
        if plan:
            path = _write_null_plan(
                project.root,
                experiment=experiment,
                feature_set_size=feature_set_size,
                seeds=seeds,
                sampling=sampling,
                exclude_feature_ids=exclude_feature_ids or [],
                output_metric=output_metric,
                planned_output_artifact=planned_output_artifact,
                force=force,
            )
            typer.echo(f"Wrote null plan: {path.relative_to(project.root)}")
            return
        if register is None:
            _fail("Missing --register RUN_ID.", code=2)
        canonical = resolve_run_id(project, register)
        run_dir = project.runs_dir / canonical
        if null_distribution is None:
            _fail("Missing --null-distribution PATH.", code=2)
        resolved_null_path = resolve_artifact_path(project, null_distribution)
        if not resolved_null_path.exists() or not resolved_null_path.is_file():
            _fail(f"Null distribution path does not exist: {null_distribution}", code=2)
        if seed_count is None or seed_count <= 0:
            _fail("--seed-count must be > 0.", code=2)
        if not metric:
            _fail("Missing --metric.", code=2)
        if percentile_rank is not None and (
            percentile_rank < 0.0 or percentile_rank > 1.0
        ):
            _fail("--percentile-rank must be between 0 and 1.", code=2)
        if (
            not force
            and has_registered_artifact_path(project, canonical, resolved_null_path)
        ):
            _fail(f"Null distribution already registered: {null_distribution}", code=2)
        register_artifact(
            project,
            canonical,
            resolved_null_path,
            claim_relevance="required",
            description=f"empirical null distribution for {metric}",
        )
        append_metric(run_dir, "random_null_seed_count", seed_count)
        append_metric(run_dir, "null_distribution_path", str(null_distribution))
        append_metric(run_dir, "null_metric", metric)
        if percentile_rank is not None:
            append_metric(run_dir, "percentile_rank", percentile_rank)
        report = evaluate_filtered_report(project, canonical, "empirical_null")
        write_tier2_check_report(run_dir, "null_check", report)
        generate_scientific_debt_report(project, canonical)
        _echo_tier2_summary("null_check", report, project, run_dir)
        raise typer.Exit(0 if report.clean else 1)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@stats_app.command("paired-test")
def stats_paired_test(
    run_id: str,
    register: Annotated[
        Path | None, typer.Option("--register", help="Registered PairedTestResult JSON.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite existing registration.")
    ] = False,
) -> None:
    try:
        if register is None:
            _fail("Missing --register path/to/paired_test.json.", code=2)
        project = find_project()
        register_path = resolve_artifact_path(project, register)
        if not register_path.exists() or not register_path.is_file():
            _fail(f"Paired-test result does not exist: {register}", code=2)
        canonical = resolve_run_id(project, run_id)
        run_dir = project.runs_dir / canonical
        result = load_paired_test_result(register_path, run_id=canonical)
        output_path = run_dir / "paired_test.json"
        if output_path.exists() and not force:
            _fail(f"Paired-test result already registered: {output_path}", code=2)
        output_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        append_metric(run_dir, "paired_test_name", result.test)
        append_metric(run_dir, "paired_by", result.paired_by)
        append_metric(run_dir, "paired_test_n_pairs", result.n_pairs)
        append_metric(run_dir, "paired_test_p_value", result.p_value)
        append_metric(run_dir, "effect_direction", result.effect_direction)
        append_metric(run_dir, "sign_consistency", result.sign_consistency)
        register_artifact(
            project,
            canonical,
            output_path,
            claim_relevance="required",
            description=f"paired {result.test} test for {result.metric}",
        )
        report = evaluate_filtered_report(project, canonical, "paired_statistic")
        (run_dir / "paired_test.md").write_text(
            paired_test_markdown(result, report), encoding="utf-8"
        )
        generate_scientific_debt_report(project, canonical)
        _echo_tier2_summary("paired_test", report, project, run_dir)
        raise typer.Exit(0 if report.clean else 1)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@prediction_app.command("lock")
def prediction_lock(
    prediction_path: Path,
    force: Annotated[
        bool, typer.Option("--force", help="Human-visible relock after semantic edits.")
    ] = False,
) -> None:
    try:
        project = find_project()
        resolved_path = resolve_artifact_path(project, prediction_path)
        before = load_prediction(resolved_path)
        prediction = lock_prediction(resolved_path, force=force)
        if before.locked_content_hash:
            action = "force_relocked" if force else "already_locked"
        else:
            action = "newly_locked"
        typer.echo(f"prediction_id: {prediction.prediction_id}")
        typer.echo(f"prediction_path: {resolved_path}")
        typer.echo(f"tamper_status: {prediction.tamper_status.value}")
        typer.echo(f"lock_hash: {prediction.locked_content_hash}")
        typer.echo(f"action: {action}")
    except PredictionStateError as exc:
        _fail(str(exc), code=1)
    except (PredictionInputError, ValueError) as exc:
        _fail(str(exc), code=2)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@prediction_app.command("score")
def prediction_score(
    prediction_id: str,
    against_run: Annotated[str, typer.Option("--against-run")],
    prediction_dir: Annotated[
        list[Path] | None,
        typer.Option("--prediction-dir", help="Additional prediction directory to search."),
    ] = None,
) -> None:
    try:
        project = find_project()
        path = find_prediction_by_id(project.root, prediction_id, prediction_dir or [])
        prediction = score_prediction_file(project, path, against_run)
        typer.echo(f"prediction_id: {prediction.prediction_id}")
        typer.echo(f"prediction_path: {path}")
        typer.echo(f"resolved_run_id: {prediction.scored_against_run_id}")
        typer.echo(f"sign_match: {_json_scalar(prediction.sign_match)}")
        typer.echo(
            f"relative_magnitude_match: "
            f"{_json_scalar(prediction.relative_magnitude_match)}"
        )
        typer.echo(f"tamper_status: {prediction.tamper_status.value}")
    except PredictionStateError as exc:
        _fail(str(exc), code=1)
    except (PredictionInputError, ValueError) as exc:
        _fail(str(exc), code=2)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@questions_app.command("list")
def questions_list_command() -> None:
    try:
        project = find_project()
        for question in list_questions(project):
            typer.echo(
                f"{question['question_id']}\t{question['status']}\t"
                f"{question['priority']}\t{question['text']}"
            )
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@questions_app.command("add")
def questions_add_command(
    text: Annotated[str, typer.Option("--text")],
    claim: Annotated[str | None, typer.Option("--claim")] = None,
    experiment: Annotated[str | None, typer.Option("--experiment")] = None,
    run: Annotated[str | None, typer.Option("--run")] = None,
    priority: Annotated[str, typer.Option("--priority")] = "normal",
) -> None:
    try:
        project = find_project()
        record = add_question(
            project,
            text=text,
            claim=claim,
            experiment=experiment,
            run=run,
            priority=priority,
        )
        typer.echo(f"question_id: {record['question_id']}")
        typer.echo(f"status: {record['status']}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@questions_app.command("resolve")
def questions_resolve_command(
    question_id: str,
    decision: Annotated[str, typer.Option("--decision")],
    resolution: Annotated[str, typer.Option("--resolution")],
) -> None:
    try:
        project = find_project()
        record = resolve_question(
            project,
            question_id,
            decision=decision,
            resolution=resolution,
        )
        typer.echo(f"question_id: {record['question_id']}")
        typer.echo(f"status: {record['status']}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@questions_app.command("show")
def questions_show_command(question_id: str) -> None:
    try:
        project = find_project()
        typer.echo(json.dumps(show_question(project, question_id), indent=2, sort_keys=True))
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@labels_app.command("import")
def labels_import_command(path: Path) -> None:
    try:
        project = find_project()
        labels = import_labels(project, path)
        typer.echo(f"imported_labels: {len(labels)}")
        for label in labels:
            typer.echo(f"label_id: {label.label_id}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@labels_app.command("validate")
def labels_validate_command(path: Path) -> None:
    try:
        labels = validate_labels_file(path)
        typer.echo(f"valid_labels: {len(labels)}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@labels_app.command("list")
def labels_list_command() -> None:
    try:
        project = find_project()
        for label in read_labels(project):
            typer.echo(f"{label.label_id}\t{label.source}\t{label.label_text}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@labels_app.command("show")
def labels_show_command(label_id: str) -> None:
    try:
        project = find_project()
        typer.echo(show_label(project, label_id).model_dump_json(indent=2))
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@labels_app.command("link")
def labels_link_command(
    label_id: str,
    claim: Annotated[str, typer.Option("--claim")],
) -> None:
    try:
        project = find_project()
        label = link_label_to_claim(project, label_id, claim)
        typer.echo(f"label_id: {label.label_id}")
        typer.echo(f"linked_claims: {', '.join(label.linked_claims)}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@dashboard_app.command("data")
def dashboard_data_command(
    out: Annotated[Path, typer.Option("--out")],
) -> None:
    try:
        project = find_project()
        output_path = _project_output_path(project, out)
        write_dashboard_data(project, output_path)
        typer.echo(f"dashboard_data: {output_path.relative_to(project.root)}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


def _query_command(
    kind: str,
    *,
    as_json: bool,
    status_filter: str | None,
    claim: str | None,
    experiment: str | None,
    run: str | None,
    severity: str | None,
    limit: int | None,
) -> None:
    project = find_project()
    rows = filter_rows(
        query_rows(project, kind),
        status=status_filter,
        claim=claim,
        experiment=experiment,
        run=run,
        severity=severity,
        limit=limit,
    )
    typer.echo(rows_json(rows) if as_json else rows_text(rows), nl=False)


def _query_options(
    kind: str,
    as_json: bool,
    status_filter: str | None,
    claim: str | None,
    experiment: str | None,
    run: str | None,
    severity: str | None,
    limit: int | None,
) -> None:
    try:
        _query_command(
            kind,
            as_json=as_json,
            status_filter=status_filter,
            claim=claim,
            experiment=experiment,
            run=run,
            severity=severity,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@query_app.command("claims")
def query_claims_command(
    as_json: Annotated[bool, typer.Option("--json")] = False,
    status_filter: Annotated[str | None, typer.Option("--status")] = None,
    claim: Annotated[str | None, typer.Option("--claim")] = None,
    experiment: Annotated[str | None, typer.Option("--experiment")] = None,
    run: Annotated[str | None, typer.Option("--run")] = None,
    severity: Annotated[str | None, typer.Option("--severity")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    _query_options("claims", as_json, status_filter, claim, experiment, run, severity, limit)


@query_app.command("runs")
def query_runs_command(
    as_json: Annotated[bool, typer.Option("--json")] = False,
    status_filter: Annotated[str | None, typer.Option("--status")] = None,
    claim: Annotated[str | None, typer.Option("--claim")] = None,
    experiment: Annotated[str | None, typer.Option("--experiment")] = None,
    run: Annotated[str | None, typer.Option("--run")] = None,
    severity: Annotated[str | None, typer.Option("--severity")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    _query_options("runs", as_json, status_filter, claim, experiment, run, severity, limit)


@query_app.command("debt")
def query_debt_command(
    as_json: Annotated[bool, typer.Option("--json")] = False,
    status_filter: Annotated[str | None, typer.Option("--status")] = None,
    claim: Annotated[str | None, typer.Option("--claim")] = None,
    experiment: Annotated[str | None, typer.Option("--experiment")] = None,
    run: Annotated[str | None, typer.Option("--run")] = None,
    severity: Annotated[str | None, typer.Option("--severity")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    _query_options("debt", as_json, status_filter, claim, experiment, run, severity, limit)


@query_app.command("artifacts")
def query_artifacts_command(
    as_json: Annotated[bool, typer.Option("--json")] = False,
    status_filter: Annotated[str | None, typer.Option("--status")] = None,
    claim: Annotated[str | None, typer.Option("--claim")] = None,
    experiment: Annotated[str | None, typer.Option("--experiment")] = None,
    run: Annotated[str | None, typer.Option("--run")] = None,
    severity: Annotated[str | None, typer.Option("--severity")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    _query_options("artifacts", as_json, status_filter, claim, experiment, run, severity, limit)


@query_app.command("decisions")
def query_decisions_command(
    as_json: Annotated[bool, typer.Option("--json")] = False,
    status_filter: Annotated[str | None, typer.Option("--status")] = None,
    claim: Annotated[str | None, typer.Option("--claim")] = None,
    experiment: Annotated[str | None, typer.Option("--experiment")] = None,
    run: Annotated[str | None, typer.Option("--run")] = None,
    severity: Annotated[str | None, typer.Option("--severity")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    _query_options("decisions", as_json, status_filter, claim, experiment, run, severity, limit)


@query_app.command("experiments")
def query_experiments_command(
    as_json: Annotated[bool, typer.Option("--json")] = False,
    status_filter: Annotated[str | None, typer.Option("--status")] = None,
    claim: Annotated[str | None, typer.Option("--claim")] = None,
    experiment: Annotated[str | None, typer.Option("--experiment")] = None,
    run: Annotated[str | None, typer.Option("--run")] = None,
    severity: Annotated[str | None, typer.Option("--severity")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    _query_options("experiments", as_json, status_filter, claim, experiment, run, severity, limit)


@records_app.command("validate")
def records_validate_command(path: Path) -> None:
    try:
        record = validate_record(path)
        typer.echo(f"record_id: {record.record_id}")
        typer.echo(f"record_type: {record.record_type}")
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@records_app.command("list")
def records_list_command() -> None:
    try:
        project = find_project()
        for record in list_records(project):
            typer.echo(
                "\t".join(
                    [
                        str(record["record_id"]),
                        str(record["canonical_record_type"]),
                        str(record["schema_status"]),
                        str(record["record_specific_id"]),
                        ",".join(record["linked_runs"]),
                        ",".join(record["linked_claims"]),
                        ",".join(record["linked_decisions"]),
                        ",".join(record["artifact_paths"]),
                    ]
                )
            )
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@records_app.command("show")
def records_show_command(record_id: str) -> None:
    try:
        project = find_project()
        typer.echo(json.dumps(show_record(project, record_id), indent=2, sort_keys=True))
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
        tamper_count = unresolved_tamper_count(project)
        if tamper_count:
            typer.echo("\nIntegrity:")
            typer.echo(f"  unresolved tamper records: {tamper_count}")
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


@app.command("next")
def next_command() -> None:
    try:
        project = find_project()
        context = load_prerequisite_context(project)
        ready: list[str] = []
        blocked: list[str] = []
        gated: list[str] = []
        for path in sorted((project.root / "research/experiments").glob("*.md")):
            if path.name.startswith("TEMPLATE_"):
                continue
            spec = parse_experiment_spec(path)
            evaluation = evaluate_experiment_prerequisites(spec, context)
            if evaluation.input_errors or evaluation.blockers:
                findings = evaluation.input_errors or evaluation.blockers
                blocked.append(
                    f"{spec.experiment_id} - {spec.title}: "
                    + "; ".join(finding.message for finding in findings)
                )
            elif evaluation.debt_or_warnings:
                gated.append(
                    f"{spec.experiment_id} - {spec.title}: "
                    + "; ".join(finding.message for finding in evaluation.debt_or_warnings)
                )
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
        if gated:
            typer.echo("DEBT/WARNING GATED")
            for item in gated:
                typer.echo(f"  {item}")
        open_questions = [
            question for question in list_questions(project) if question.get("status") == "open"
        ]
        if open_questions:
            typer.echo("OPEN QUESTIONS")
            for question in open_questions:
                links = []
                if question.get("linked_claims"):
                    links.append("claims=" + ",".join(question["linked_claims"]))
                if question.get("linked_experiments"):
                    links.append("experiments=" + ",".join(question["linked_experiments"]))
                typer.echo(
                    f"  {question['question_id']} - {question['text']}"
                    + (f" ({'; '.join(links)})" if links else "")
                )
        raise typer.Exit(0 if ready or not blocked else 1)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc), code=2)


SUPPORTED_NULL_SAMPLING = {
    "random_feature_sets",
    "density_matched",
    "score_matched",
    "bottom_active",
    "custom",
}


def _write_null_plan(
    root: Path,
    *,
    experiment: str | None,
    feature_set_size: int | None,
    seeds: int | None,
    sampling: str,
    exclude_feature_ids: list[str],
    output_metric: str,
    planned_output_artifact: str | None,
    force: bool,
) -> Path:
    if not experiment:
        raise ValueError("Missing --experiment for null plan.")
    if feature_set_size is None or feature_set_size <= 0:
        raise ValueError("--feature-set-size must be > 0.")
    if seeds is None or seeds <= 0:
        raise ValueError("--seeds must be > 0.")
    if sampling not in SUPPORTED_NULL_SAMPLING:
        raise ValueError(
            "--sampling must be one of: " + ", ".join(sorted(SUPPORTED_NULL_SAMPLING))
        )
    if not output_metric:
        raise ValueError("--output-metric must not be empty.")
    path = root / "research/experiments" / f"{experiment}_null_plan.yaml"
    if path.exists() and not force:
        raise FileExistsError(f"Null plan already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": experiment,
        "feature_set_size": feature_set_size,
        "seed_count": seeds,
        "sampling_method": sampling,
        "exclude_feature_ids": exclude_feature_ids,
        "output_metric": output_metric,
        "planned_output_artifact": planned_output_artifact
        or f"runs/null/{experiment}_null_distribution.jsonl",
    }
    yaml = YAML()
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(payload, handle)
    return path


def _echo_tier2_summary(name: str, report, project, run_dir: Path) -> None:
    open_debts = report.open_debts
    blockers = report.blocking_debts
    typer.echo(f"run_id: {report.run_id}")
    typer.echo(f"assessment_ids: {', '.join(report.assessment_ids)}")
    typer.echo(f"clean: {str(report.clean).lower()}")
    typer.echo(
        "open_debt: "
        + (
            ", ".join(f"{debt.debt_id}/{debt.debt_type}" for debt in open_debts)
            if open_debts
            else "none"
        )
    )
    typer.echo(
        "blocking_findings: "
        + (
            ", ".join(f"{debt.debt_id}/{debt.debt_type}" for debt in blockers)
            if blockers
            else "none"
        )
    )
    json_path = (run_dir / f"{name}.json").relative_to(project.root)
    md_path = (run_dir / f"{name}.md").relative_to(project.root)
    typer.echo(f"{name}: {json_path}")
    typer.echo(f"{name}_markdown: {md_path}")


def _json_scalar(value: object) -> str:
    return json.dumps(value)


def _default_drafts(project) -> list[Path]:
    root = project.root / "research/paper"
    paths = []
    for pattern in ("**/*.md", "**/*.markdown", "**/*.tex"):
        paths.extend(sorted(root.glob(pattern)))
    return paths


def _print_run_help() -> None:
    typer.echo(
        "Usage: mechledger run [OPTIONS] -- COMMAND [ARGS]...\n\n"
        "Capture a local command as a MechLedger run.\n\n"
        "Options:\n"
        "  --experiment TEXT\n"
        "  --class TEXT                 Run class, default scratch.\n"
        "  --purpose TEXT\n"
        "  --hypothesis TEXT\n"
        "  --run-id TEXT\n"
        "  --model TEXT\n"
        "  --hook-point TEXT\n"
        "  --sae-release TEXT\n"
        "  --sae-id TEXT\n"
        "  --seed INTEGER\n\n"
        "Workflow:\n"
        "  mechledger run -- python script.py\n"
        "  mechledger run reclassify RUN_ID --to CLASS --decision D### --reason TEXT"
    )


def _looks_like_run_reclassify(argv: list[str]) -> bool:
    if not argv or argv[0] != "reclassify":
        return False
    return "--help" in argv or any(option in argv for option in ("--to", "--decision", "--reason"))


def _handle_run_reclassify(project, argv: list[str]) -> str:
    if "--help" in argv:
        _handle_run_reclassify_help()
        raise typer.Exit(0)
    if len(argv) < 2:
        raise ValueError(
            "Usage: mechledger run reclassify RUN_ID --to CLASS --decision D### --reason TEXT"
        )
    run_id = argv[1]
    to_class = _required_option(argv, "--to")
    decision = _required_option(argv, "--decision")
    reason = _required_option(argv, "--reason")
    known_tokens = {
        "reclassify",
        run_id,
        "--to",
        to_class,
        "--decision",
        decision,
        "--reason",
        reason,
    }
    extras = [token for token in argv if token not in known_tokens]
    if extras:
        raise ValueError(f"Unexpected run reclassify arguments: {' '.join(extras)}")
    return reclassify_run(
        project,
        run_id,
        to_class=to_class,
        decision_id=decision,
        reason=reason,
    )


def _handle_run_reclassify_help() -> None:
    typer.echo(
        "Usage: mechledger run reclassify RUN_ID --to CLASS "
        "--decision D### --reason TEXT\n\n"
        "Reclassify a local run after accepted human decision review. "
        "This updates the run directory and regenerated scientific-debt report; "
        "it does not edit committed ledgers."
    )


def _required_option(argv: list[str], option: str) -> str:
    value = _option_value(argv, option)
    if value is None or not value.strip():
        raise ValueError(f"Missing required option for run reclassify: {option}")
    return value


def _option_value(argv: list[str], option: str) -> str | None:
    try:
        index = argv.index(option)
    except ValueError:
        return None
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        return ""
    return argv[index + 1]


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


def _fail(message: str, *, code: int) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code)
