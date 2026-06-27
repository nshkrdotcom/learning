from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from IPython import start_ipython
from IPython.core.interactiveshell import InteractiveShell
from rich.console import Console

from mwb.adapters.saelens import SAELensAdapter
from mwb.adapters.transformer_lens import TransformerLensAdapter
from mwb.bundle_audit import BundleAuditService
from mwb.causal_verification import CausalVerificationService
from mwb.context import RunContext
from mwb.doctor import run_doctor
from mwb.evidence_graph import QUERY_KINDS, EvidenceGraphService
from mwb.hypothesis_lifecycle import HypothesisLifecycleService
from mwb.ipython.extension import start_workbench_ipython, unload_ipython_extension
from mwb.ledgers import propose_claim_update, propose_run_ledger_row, validate_ledgers
from mwb.project import ProjectManager
from mwb.refs import stable_ref
from mwb.session import SessionManager, latest_session
from mwb.space_types import SpaceTypeService
from mwb.sqlite_index import rebuild_sqlite_index
from mwb.static_compiler import StaticCompiler
from mwb.workflows.cards import card_from_run, write_card
from mwb.workflows.diagnosis import DiagnosisService
from mwb.workflows.draft_guard import check_draft_text, load_claim_cards
from mwb.workflows.io import load_json_payload
from mwb.workflows.next_probe import build_next_probe, load_next_probe_payload, write_next_probe
from mwb.workflows.preflight import run_preflight
from mwb.workflows.runs import resolve_run_path
from mwb.workflows.self_ground_ingest import ingest_self_ground_run
from mwb.workflows.sweep import parse_axes, write_sweep_run

app = typer.Typer(no_args_is_help=True, help="Mechanistic Workbench local CLI.")
inspect_app = typer.Typer(help="Inspect local workbench state.")
adapter_app = typer.Typer(help="Adapter manifests and conformance checks.")
conformance_app = typer.Typer(help="Run adapter conformance checks.")
demo_app = typer.Typer(help="Run built-in workbench demos.")
ingest_app = typer.Typer(help="Ingest external research artifact sets.")
graph_app = typer.Typer(help="Rebuild and query the local evidence graph.")
ledger_app = typer.Typer(help="Validate Git-native research ledgers and proposals.")
hypothesis_app = typer.Typer(help="Manage hypothesis lifecycle and alternatives.")
space_app = typer.Typer(help="Check mechanistic tensor spaces and unit operations.")
compile_app = typer.Typer(help="Compile static mechanistic plausibility checks.")
bundle_app = typer.Typer(help="Audit and rebalance example/control bundles.")
app.add_typer(inspect_app, name="inspect")
app.add_typer(adapter_app, name="adapter")
app.add_typer(demo_app, name="demo")
app.add_typer(ingest_app, name="ingest")
app.add_typer(graph_app, name="graph")
app.add_typer(ledger_app, name="ledger")
app.add_typer(hypothesis_app, name="hypothesis")
app.add_typer(space_app, name="space")
app.add_typer(compile_app, name="compile")
app.add_typer(bundle_app, name="bundle")
adapter_app.add_typer(conformance_app, name="conformance")
console = Console()
DEFAULT_ROOT = Path(".")
NameOption = Annotated[str, typer.Option("--name", help="Stable project name.")]
RootOption = Annotated[Path, typer.Option("--root", help="Repository or project root.")]
DoctorRootOption = Annotated[Path, typer.Option("--root", help="Project root or child path.")]
ExecuteOption = Annotated[
    list[str] | None,
    typer.Option("--execute", help="Execute one IPython cell, repeatable."),
]
ResumeOption = Annotated[str | None, typer.Option("--resume", help="Resume from a session ref.")]
SessionRefArgument = Annotated[str, typer.Argument(help="Session ref or 'latest'.")]
DeviceOption = Annotated[str, typer.Option("--device", help="Execution device.")]
DryRunOption = Annotated[bool, typer.Option("--dry-run", help="Validate without loading backend.")]
HypothesisFileArgument = Annotated[Path, typer.Argument(help="Hypothesis JSON file.")]
AxisOption = Annotated[
    list[str] | None,
    typer.Option("--axis", help="Sweep axis in name=value[,value...] form."),
]
SelfGroundPathArgument = Annotated[Path, typer.Argument(help="SELF-GROUND run directory.")]
OutputPathOption = Annotated[
    Path | None,
    typer.Option("--output", help="Output SQLite path for rebuilt index."),
]
GraphQueryKindArgument = Annotated[
    str,
    typer.Argument(
        help=(
            "Query kind: claims-depending-on, controls-contradicting, "
            "cells-producing, debt-blocking."
        )
    ),
]
GraphRefArgument = Annotated[str, typer.Argument(help="Source or target ref for the graph query.")]
RunRefArgument = Annotated[str, typer.Argument(help="Run ref or run directory.")]
ProbePathArgument = Annotated[Path, typer.Argument(help="Materialized probe YAML or JSON file.")]
CardRefArgument = Annotated[str, typer.Argument(help="MechanismCard ref or JSON path.")]
HypothesisRefCliArgument = Annotated[str, typer.Argument(help="Hypothesis ref.")]
HypothesisStateOption = Annotated[str, typer.Option("--to-state", help="Target workflow state.")]
EvidenceTierOption = Annotated[
    str | None,
    typer.Option("--evidence-tier", help="Current evidence tier, separate from workflow state."),
]
ClaimStatusOption = Annotated[
    str | None,
    typer.Option("--claim-status", help="Current claim status, separate from workflow state."),
]
ApprovedByOption = Annotated[
    str | None,
    typer.Option("--approved-by", help="Reviewer required for claimable promotion."),
]
DecisionRefOption = Annotated[
    str | None,
    typer.Option("--decision-ref", help="Decision ref required for claimable promotion."),
]
ReasonOption = Annotated[str | None, typer.Option("--reason", help="Transition rationale.")]
SpaceCheckFileArgument = Annotated[Path, typer.Argument(help="Space check JSON file.")]
CompileHypothesisFileArgument = Annotated[Path, typer.Argument(help="Hypothesis JSON file.")]
BundleNameArgument = Annotated[
    str,
    typer.Argument(help="Built-in bundle name, e.g. negation_phase3_calibrated."),
]
MaterializeProbeOption = Annotated[
    bool,
    typer.Option("--materialize", help="Also write probe.yaml/probe.json for runnable probes."),
]


@app.command()
def init(
    name: NameOption = "self-ground",
    root: RootOption = DEFAULT_ROOT,
) -> None:
    """Initialize or reuse a local .mechanism workspace."""
    project = ProjectManager.init(root.resolve(), name=name)
    console.print(f"project: {project.name}")
    console.print(f"workspace: {project.mechanism_dir.relative_to(project.root)}")
    console.print(f"database: {project.sqlite_path.relative_to(project.root)}")


@app.command()
def doctor(root: DoctorRootOption = DEFAULT_ROOT) -> None:
    """Validate local workbench project state without mutating research evidence."""
    report = run_doctor(root.resolve())
    console.print(report.render())
    if report.status != "ok":
        raise typer.Exit(code=1)


@app.command("repair-index")
@app.command("rebuild-index")
def rebuild_index(output: OutputPathOption = None) -> None:
    """Rebuild a separate SQLite index from file-backed .mechanism records."""
    project = ProjectManager.discover()
    report = rebuild_sqlite_index(project, output_path=output)
    console.print_json(json.dumps(report))


@graph_app.command("rebuild")
def graph_rebuild() -> None:
    """Rebuild evidence graph JSONL and SQLite edges from file-backed records."""
    project = ProjectManager.discover_or_create()
    report = EvidenceGraphService(project).rebuild()
    console.print_json(json.dumps(report))


@graph_app.command("query")
def graph_query(kind: GraphQueryKindArgument, ref: GraphRefArgument) -> None:
    """Query graph answers needed for claim, control, cell, and debt review."""
    if kind not in QUERY_KINDS:
        expected = ", ".join(sorted(QUERY_KINDS))
        raise typer.BadParameter(f"unknown graph query kind {kind!r}; expected one of: {expected}")
    project = ProjectManager.discover()
    service = EvidenceGraphService(project)
    if not service.edge_path.exists():
        raise typer.BadParameter("evidence graph is missing; run `mwb graph rebuild` first")
    console.print_json(json.dumps(service.query(kind, ref)))


@ledger_app.command("validate")
def ledger_validate() -> None:
    """Validate and index Git-visible research ledgers."""
    project = ProjectManager.discover_or_create()
    report = validate_ledgers(project)
    console.print_json(json.dumps(report))
    if report["status"] != "ok":
        raise typer.Exit(code=1)


@ledger_app.command("propose-run")
def ledger_propose_run(run_ref: RunRefArgument) -> None:
    """Write a human-reviewable run ledger row proposal for a local run."""
    project = ProjectManager.discover_or_create()
    report = propose_run_ledger_row(project, run_ref)
    console.print_json(json.dumps(report))


@ledger_app.command("propose-claim")
def ledger_propose_claim(card_ref: CardRefArgument) -> None:
    """Write a human-reviewable claim ledger proposal from a MechanismCard."""
    project = ProjectManager.discover_or_create()
    report = propose_claim_update(project, card_ref)
    console.print_json(json.dumps(report))


@hypothesis_app.command("transition")
def hypothesis_transition(
    hypothesis_ref: HypothesisRefCliArgument,
    to_state: HypothesisStateOption,
    evidence_tier: EvidenceTierOption = None,
    claim_status: ClaimStatusOption = None,
    approved_by: ApprovedByOption = None,
    decision_ref: DecisionRefOption = None,
    reason: ReasonOption = None,
) -> None:
    """Record a hypothesis workflow transition receipt."""
    project = ProjectManager.discover_or_create()
    try:
        report = HypothesisLifecycleService(project).transition(
            hypothesis_ref,
            to_state=to_state,
            evidence_tier=evidence_tier,
            claim_status=claim_status,
            approved_by=approved_by,
            decision_ref=decision_ref,
            reason=reason,
        )
    except ValueError as exc:
        console.print(f"error: {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(json.dumps(report))


@hypothesis_app.command("explain")
def hypothesis_explain(run_ref: RunRefArgument) -> None:
    """Write live alternative explanations from a run's blocker report."""
    project = ProjectManager.discover_or_create()
    try:
        report = HypothesisLifecycleService(project).explain(run_ref)
    except FileNotFoundError as exc:
        console.print(f"error: {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(json.dumps(report))


@space_app.command("check")
def space_check(path: SpaceCheckFileArgument) -> None:
    """Validate tensor-space compatibility and mechanistic unit operations."""
    project = ProjectManager.discover_or_create()
    service = SpaceTypeService(project)
    report = service.check_file(path)
    service.write_report(report)
    console.print_json(json.dumps(report.model_dump(mode="json")))
    if report.status == "fail":
        raise typer.Exit(code=1)


@compile_app.command("hypothesis")
def compile_hypothesis(path: CompileHypothesisFileArgument) -> None:
    """Run static algebraic plausibility checks for a hypothesis JSON file."""
    project = ProjectManager.discover_or_create()
    compiler = StaticCompiler(project)
    report = compiler.compile_file(path)
    compiler.write_report(report)
    console.print_json(json.dumps(report.model_dump(mode="json")))
    if report.status == "fail":
        raise typer.Exit(code=1)


@bundle_app.command("audit")
def bundle_audit(name: BundleNameArgument) -> None:
    """Audit token validity, role balance, contamination, and baseline margins."""
    project = ProjectManager.discover_or_create()
    service = BundleAuditService(project)
    try:
        report = service.audit_bundle(name)
    except ValueError as exc:
        console.print(f"error: {exc}")
        raise typer.Exit(code=1) from exc
    service.write_report(report)
    console.print_json(json.dumps(report.model_dump(mode="json")))
    if report.status == "fail":
        raise typer.Exit(code=1)


@bundle_app.command("rebalance")
def bundle_rebalance(
    name: BundleNameArgument = "negation_phase3_calibrated",
    dry_run: DryRunOption = False,
) -> None:
    """Generate heldout and control-family bundle improvement proposals."""
    project = ProjectManager.discover_or_create()
    service = BundleAuditService(project)
    try:
        proposal = service.rebalance_bundle(name, dry_run=dry_run)
    except ValueError as exc:
        console.print(f"error: {exc}")
        raise typer.Exit(code=1) from exc
    service.write_rebalance(proposal)
    console.print_json(json.dumps(proposal.model_dump(mode="json")))


@app.command()
def ipython(execute: ExecuteOption = None, resume: ResumeOption = None) -> None:
    """Launch IPython with the Workbench extension loaded."""
    if execute:
        InteractiveShell.clear_instance()
        shell = InteractiveShell.instance()
        start_workbench_ipython(shell, resume=resume)
        exit_code = 0
        try:
            for cell in execute:
                result = shell.run_cell(cell, store_history=True)
                if result.error_before_exec or result.error_in_exec:
                    exit_code = 1
        finally:
            unload_ipython_extension(shell)
            InteractiveShell.clear_instance()
        if exit_code:
            raise typer.Exit(code=exit_code)
        return

    previous_resume = os.environ.get("MWB_RESUME_SESSION")
    if resume:
        os.environ["MWB_RESUME_SESSION"] = resume
    try:
        start_ipython(argv=["--ext", "mwb.ipython"])
    finally:
        if previous_resume is None:
            os.environ.pop("MWB_RESUME_SESSION", None)
        else:
            os.environ["MWB_RESUME_SESSION"] = previous_resume


@app.command()
def preflight(hypothesis: HypothesisFileArgument) -> None:
    """Run static preflight checks for a hypothesis JSON file."""
    payload = load_json_payload(hypothesis)
    report = run_preflight(payload)
    console.print_json(json.dumps(report.model_dump(mode="json")))
    if report.status == "fail":
        raise typer.Exit(code=1)


@app.command()
def verify(
    hypothesis: HypothesisFileArgument,
    prediction_lock: Annotated[
        Path | None, typer.Option("--prediction-lock", help="Prediction lock JSON file.")
    ] = None,
    diagnostic_only: Annotated[
        bool, typer.Option("--diagnostic-only", help="Produce diagnostic-only verification output.")
    ] = False,
    dry_run: DryRunOption = False,
) -> None:
    """Plan or run causal verification for a hypothesis JSON file."""
    project = ProjectManager.discover_or_create()
    payload = load_json_payload(hypothesis)
    lock_payload = load_json_payload(prediction_lock) if prediction_lock else None
    result = CausalVerificationService(project).verify_payload(
        payload,
        prediction_lock=lock_payload,
        diagnostic_only=diagnostic_only,
        dry_run=dry_run,
    )
    console.print_json(json.dumps(result.model_dump(mode="json")))
    if result.status == "blocked":
        raise typer.Exit(code=1)


@app.command()
def sweep(
    hypothesis: HypothesisFileArgument,
    axis: AxisOption = None,
    dry_run: DryRunOption = False,
) -> None:
    """Plan a verification sweep over a cross-product of axes."""
    project = ProjectManager.discover_or_create()
    payload = load_json_payload(hypothesis) if hypothesis.exists() else {"wb_ref": str(hypothesis)}
    config = parse_axes(axis or [])
    config["source_hypothesis_ref"] = payload["wb_ref"]
    config["dry_run"] = dry_run
    _run_dir, output = write_sweep_run(
        project=project,
        hypothesis_payload=payload,
        config=config,
        dry_run=dry_run,
    )
    console.print_json(json.dumps(output))


@app.command("next-probe")
def next_probe(
    run_path: Annotated[Path, typer.Argument(help="Run directory or JSON file.")],
    materialize: MaterializeProbeOption = False,
) -> None:
    """Generate a deterministic next-probe plan from run artifacts."""
    project = ProjectManager.discover_or_create()
    resolved_run_path = resolve_run_path(run_path, project=project)
    payload = load_next_probe_payload(resolved_run_path)
    plan = build_next_probe(payload)
    if resolved_run_path.is_dir():
        write_next_probe(resolved_run_path, plan)
        if materialize:
            service = DiagnosisService(project)
            tree = service.write_diagnosis(
                resolved_run_path,
                service.diagnose_run_dir(resolved_run_path),
            )
            service.materialize_probe(resolved_run_path, plan=plan, tree=tree)
    elif materialize:
        raise typer.BadParameter("--materialize requires a run directory")
    console.print_json(json.dumps(plan.model_dump(mode="json")))
    if plan.diagnosis["primary"] == "artifact_incomplete" and not materialize:
        raise typer.Exit(code=1)


@app.command()
def diagnose(run_path: Annotated[Path, typer.Argument(help="Run directory or 'latest'.")]) -> None:
    """Write and print a provenance-preserving diagnosis tree for a run."""
    project = ProjectManager.discover_or_create()
    resolved_run_path = resolve_run_path(run_path, project=project)
    if not resolved_run_path.is_dir():
        raise typer.BadParameter("diagnose requires a run directory")
    service = DiagnosisService(project)
    tree = service.write_diagnosis(resolved_run_path, service.diagnose_run_dir(resolved_run_path))
    console.print_json(json.dumps(tree.model_dump(mode="json")))


@app.command("run-probe")
def run_probe(probe_path: ProbePathArgument) -> None:
    """Execute an implemented materialized probe through the local workflow runner."""
    project = ProjectManager.discover_or_create()
    try:
        report = DiagnosisService(project).run_probe(probe_path)
    except ValueError as exc:
        console.print(f"error: {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(json.dumps(report))


@app.command()
def card(run_path: Annotated[Path, typer.Argument(help="Run directory.")]) -> None:
    """Generate a MechanismCard from run artifacts."""
    project = ProjectManager.discover_or_create()
    resolved_run_path = resolve_run_path(run_path, project=project)
    mechanism_card = card_from_run(resolved_run_path)
    write_card(resolved_run_path, mechanism_card, mechanism_dir=project.mechanism_dir)
    console.print_json(json.dumps(mechanism_card.model_dump(mode="json")))


@app.command("draft-check")
def draft_check(draft_path: Annotated[Path, typer.Argument(help="Draft Markdown path.")]) -> None:
    """Check draft claim language against generated MechanismCards."""
    project = ProjectManager.discover()
    claim_cards = load_claim_cards(project.mechanism_dir)
    report = check_draft_text(draft_path.read_text(encoding="utf-8"), claim_cards)
    console.print_json(json.dumps(report))
    if report["status"] in {"blocked", "missing_card"}:
        raise typer.Exit(code=1)


@ingest_app.command("self-ground")
def ingest_self_ground(source: SelfGroundPathArgument) -> None:
    """Ingest a SELF-GROUND E004 artifact set into .mechanism/runs."""
    project = ProjectManager.discover_or_create()
    run_dir = ingest_self_ground_run(source, project=project)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    blocker_report = json.loads((run_dir / "blocker_report.json").read_text(encoding="utf-8"))
    console.print_json(
        json.dumps(
            {
                "run_ref": manifest["run_ref"],
                "status": manifest["status"],
                "run_dir": str(run_dir),
                "primary_blocker": blocker_report.get("primary_blocker"),
            }
        )
    )


@inspect_app.command("session")
def inspect_session(session_ref: SessionRefArgument = "latest") -> None:
    """Inspect a recorded IPython session."""
    project = ProjectManager.discover()
    session_dir = (
        latest_session(project)
        if session_ref == "latest"
        else project.mechanism_dir / "sessions" / session_ref
    )
    session_json = session_dir / "session.json"
    if not session_json.exists():
        raise typer.BadParameter(f"session not found: {session_ref}")
    import json

    payload = json.loads(session_json.read_text())
    console.print(f"session: {payload['session_ref']}")
    console.print(f"surface: {payload['surface']}")
    console.print(f"mode: {payload['mode']}")
    console.print(f"started_at: {payload['started_at']}")
    console.print(f"ended_at: {payload['ended_at']}")


@conformance_app.command("transformer-lens")
def conformance_transformer_lens(
    model: Annotated[str, typer.Option("--model", help="TransformerLens model name.")],
    hook: Annotated[
        str | None, typer.Option("--hook", help="Hook point to capture.")
    ] = "blocks.0.hook_resid_post",
    device: DeviceOption = "cpu",
    dry_run: DryRunOption = False,
) -> None:
    """Run TransformerLens adapter conformance."""
    project = ProjectManager.discover_or_create()
    output_dir = project.mechanism_dir / "adapters" / "transformer_lens"
    result = TransformerLensAdapter().run_conformance(
        model_name=model,
        hook_point=hook,
        device=device,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    console.print_json(json.dumps(result.model_dump(mode="json")))
    if result.status == "fail":
        raise typer.Exit(code=1)


@conformance_app.command("saelens")
def conformance_saelens(
    model: Annotated[str, typer.Option("--model", help="Model name linked to the SAE.")],
    hook: Annotated[str, typer.Option("--hook", help="SAE hook point.")],
    release: Annotated[
        str, typer.Option("--release", help="SAELens release.")
    ] = "pythia-70m-deduped-res-sm",
    sae_id: Annotated[str | None, typer.Option("--sae-id", help="SAE id.")] = None,
    device: DeviceOption = "cpu",
    dry_run: DryRunOption = False,
) -> None:
    """Run SAELens adapter conformance."""
    project = ProjectManager.discover_or_create()
    model_ref = stable_ref("model", "transformer_lens", model)
    tensor_space_ref = stable_ref("space", model_ref, hook)
    output_dir = project.mechanism_dir / "adapters" / "saelens"
    result = SAELensAdapter().run_conformance(
        model_ref=model_ref,
        tensor_space_ref=tensor_space_ref,
        hook_point=hook,
        release=release,
        sae_id=sae_id or hook,
        device=device,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    console.print_json(json.dumps(result.model_dump(mode="json")))
    if result.status == "fail":
        raise typer.Exit(code=1)


@demo_app.command("negation")
def demo_negation(
    model: Annotated[str, typer.Option("--model", help="Model to use for the demo.")],
    device: DeviceOption = "cpu",
    dry_run: DryRunOption = False,
) -> None:
    """Run or validate the built-in SELF-GROUND negation demo."""
    project = ProjectManager.discover_or_create()
    session = SessionManager.start(project, surface="cli", mode="scratch")
    ctx = RunContext(project=project, session=session)
    try:
        bundle = ctx.domains.negation.load("phase3_calibrated")
        if dry_run:
            payload = {
                "status": "dry_run",
                "model": model,
                "device": device,
                "bundle_ref": bundle.targets.wb_ref,
                "control_bundle_ref": bundle.controls.wb_ref,
                "n_examples": len(bundle.targets.examples),
                "control_families": sorted(bundle.controls.control_families),
            }
            console.print_json(json.dumps(payload))
            return
        loaded_model = ctx.models.load_tl(model, device=device)
        acts = ctx.capture(loaded_model, bundle).at("blocks.2.hook_resid_post")
        payload = {
            "status": "captured",
            "model_ref": loaded_model.wb_ref,
            "activation_ref": acts.wb_ref,
            "activation_summary": acts.activation_summary,
        }
        console.print_json(json.dumps(payload))
    finally:
        session.close()


if __name__ == "__main__":
    app()
