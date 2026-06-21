from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from mechanismlab.backends.optional_imports import optional_package_manifest
from mechanismlab.core import (
    ArtifactContract,
    EvidenceThresholds,
    load_claim_spec,
    load_experiment_spec,
    load_run_manifest,
    write_model,
)
from mechanismlab.reports import build_claim_report, write_claim_report_markdown

app = typer.Typer(no_args_is_help=True)
console = Console()


def _load_evidence_payload(run_dir: Path) -> dict:
    path = run_dir / "evidence_payload.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@app.command("validate")
def validate_experiment(experiment: Path) -> None:
    spec = load_experiment_spec(experiment)
    table = Table(title=f"Experiment {spec.experiment_id}")
    table.add_column("field")
    table.add_column("value")
    table.add_row("claim_id", spec.claim_id)
    table.add_row("status", spec.status)
    table.add_row("required_artifacts", ", ".join(spec.required_artifacts) or "none")
    table.add_row("required_controls", ", ".join(spec.required_controls) or "none")
    console.print(table)


@app.command("report")
def report(
    run_dir: Path,
    claim: Annotated[Path | None, typer.Option()] = None,
    experiment: Annotated[Path | None, typer.Option()] = None,
    out_json: Annotated[Path | None, typer.Option()] = None,
    out_md: Annotated[Path | None, typer.Option()] = None,
) -> None:
    claim_spec = load_claim_spec(claim) if claim is not None else None
    experiment_spec = load_experiment_spec(experiment) if experiment is not None else None
    contract = (
        ArtifactContract(required=experiment_spec.required_artifacts)
        if experiment_spec is not None
        else None
    )
    report_model = build_claim_report(
        run_dir=run_dir,
        claim=claim_spec,
        experiment=experiment_spec,
        artifact_contract=contract,
        thresholds=EvidenceThresholds(),
        evidence_payload=_load_evidence_payload(run_dir),
    )
    json_path = out_json or (run_dir / "claim_report.json")
    md_path = out_md or (run_dir / "claim_report.md")
    write_model(report_model, json_path)
    write_claim_report_markdown(report_model, md_path)
    console.print_json(data=report_model.model_dump(mode="json"))


@app.command("inspect-run")
def inspect_run(run_dir: Path) -> None:
    table = Table(title=str(run_dir))
    table.add_column("name")
    table.add_column("kind")
    table.add_column("present")
    for name in [
        "mechanismlab_run_manifest.json",
        "mechanismlab_claim.json",
        "mechanismlab_experiment.json",
        "mechanismlab_claim_report.json",
        "evidence_payload.json",
    ]:
        table.add_row(name, "known", str((run_dir / name).exists()))
    for path in sorted(run_dir.iterdir()) if run_dir.exists() else []:
        if path.is_file():
            table.add_row(path.name, "file", "true")
    console.print(table)
    manifest_path = run_dir / "mechanismlab_run_manifest.json"
    if manifest_path.exists():
        manifest = load_run_manifest(manifest_path)
        console.print_json(data=manifest.model_dump(mode="json"))


@app.command("backends")
def backends() -> None:
    packages = [
        ("transformer_lens", "model"),
        ("sae_lens", "representation"),
        ("nnsight", "remote_execution"),
        ("pyvene", "intervention"),
        ("wandb", "tracker"),
        ("mlflow", "tracker"),
        ("dvc", "artifact_versioning"),
        ("hydra", "config"),
        ("sae_bench", "evaluation"),
        ("saebench", "evaluation"),
        ("ravel", "evaluation"),
    ]
    table = Table(title="MechanismLab Optional Backends")
    table.add_column("package")
    table.add_column("kind")
    table.add_column("available")
    table.add_column("version")
    for package, kind in packages:
        manifest = optional_package_manifest(package, kind=kind)
        table.add_row(
            manifest.name,
            manifest.kind,
            str(manifest.available),
            manifest.version or "",
        )
    console.print(table)


if __name__ == "__main__":
    app()
