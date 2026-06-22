from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from self_ground.io import read_jsonl, read_minimal_pairs, write_jsonl
from self_ground.negation import generate_negation_pairs
from self_ground.real_behavioral_intervention import (
    parse_amplify_factors,
    parse_int_list,
    parse_operations,
    run_real_behavioral_sae_intervention,
)
from self_ground.real_model_check import check_real_model
from self_ground.real_ranking import run_activation_ranking
from self_ground.real_residual_intervention import run_real_residual_intervention
from self_ground.real_sae_intervention import run_real_sae_intervention
from self_ground.sae_compat import verify_sae_compatibility

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("generate-negation")
def generate_negation(
    per_family: Annotated[int, typer.Option(min=1)] = 15,
    out: Annotated[Path, typer.Option()] = Path("data/negation_pairs.jsonl"),
    seed: Annotated[int, typer.Option()] = 7,
) -> None:
    pairs = generate_negation_pairs(per_family=per_family, seed=seed)
    write_jsonl(pairs, out)
    console.print(f"wrote {len(pairs)} pairs to {out}")


@app.command("check-real-model")
def check_real_model_command(
    model: Annotated[str, typer.Option()] = "EleutherAI/pythia-70m",
    hook_point: Annotated[str, typer.Option()] = "blocks.2.hook_resid_post",
    device: Annotated[str, typer.Option()] = "cpu",
    out: Annotated[Path, typer.Option()] = Path("runs/check_real_model.json"),
) -> None:
    artifact = check_real_model(
        model_name=model,
        hook_point=hook_point,
        device=device,
        out=out,
    )
    console.print_json(data=artifact)


@app.command("run-activation-ranking")
def run_activation_ranking_command(
    out: Annotated[Path, typer.Option()],
    pairs: Annotated[Path | None, typer.Option()] = None,
    model: Annotated[str, typer.Option()] = "EleutherAI/pythia-70m",
    hook_point: Annotated[str, typer.Option()] = "blocks.2.hook_resid_post",
    feature_source: Annotated[
        str,
        typer.Option(
            help=(
                "Feature source. When set to sae, SAE metadata must semantically "
                "match the requested model and hook."
            )
        ),
    ] = "residual_dimensions",
    pooling: Annotated[str, typer.Option()] = "final_token",
    per_family: Annotated[int, typer.Option(min=1)] = 15,
    seed: Annotated[int, typer.Option()] = 7,
    top_k_features: Annotated[int, typer.Option(min=1)] = 50,
    device: Annotated[str, typer.Option()] = "cpu",
    sae_release: Annotated[str | None, typer.Option()] = None,
    sae_id: Annotated[str | None, typer.Option()] = None,
    task_source: Annotated[str, typer.Option()] = "generated",
    task_file: Annotated[Path | None, typer.Option()] = None,
    task_source_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    if feature_source == "sae" and (not sae_release or not sae_id):
        raise typer.BadParameter("feature_source=sae requires --sae-release and --sae-id")
    if task_source not in {"generated", "file"}:
        raise typer.BadParameter("--task-source must be generated or file")
    if task_source == "file" and task_file is None:
        raise typer.BadParameter("--task-source=file requires --task-file")
    result = run_activation_ranking(
        out_dir=out,
        pairs_path=pairs,
        per_family=per_family,
        seed=seed,
        model_name=model,
        hook_point=hook_point,
        feature_source=feature_source,
        pooling=pooling,
        top_k_features=top_k_features,
        device=device,
        sae_release=sae_release,
        sae_id=sae_id,
        task_source=task_source,
        task_file=task_file,
        task_source_id=task_source_id,
    )
    console.print(
        f"wrote {result.feature_source} ranking with {result.n_pairs} pairs "
        f"and {result.n_features} features to {result.out_dir}"
    )


@app.command(
    "run-residual-intervention",
    help="Run a TransformerLens residual smoke patch diagnostic; not claim evidence.",
)
def run_residual_intervention_command(
    out: Annotated[Path, typer.Option()],
    ranking_dir: Annotated[Path | None, typer.Option()] = None,
    pairs: Annotated[Path | None, typer.Option()] = None,
    model: Annotated[str, typer.Option()] = "EleutherAI/pythia-70m",
    hook_point: Annotated[str, typer.Option()] = "blocks.2.hook_resid_post",
    per_family: Annotated[int, typer.Option(min=1)] = 15,
    seed: Annotated[int, typer.Option()] = 7,
    top_k_features: Annotated[int, typer.Option(min=1)] = 5,
    operation: Annotated[str, typer.Option()] = "zero",
    factor: Annotated[float, typer.Option()] = 0.0,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    if operation not in {"zero", "amplify"}:
        raise typer.BadParameter("--operation must be zero or amplify")
    if operation == "amplify" and factor == 1.0:
        raise typer.BadParameter("--operation amplify requires --factor not equal to 1.0")
    result = run_real_residual_intervention(
        out_dir=out,
        ranking_dir=ranking_dir,
        pairs_path=pairs,
        per_family=per_family,
        seed=seed,
        model_name=model,
        hook_point=hook_point,
        top_k_features=top_k_features,
        operation=operation,  # type: ignore[arg-type]
        factor=factor,
        device=device,
    )
    console.print(
        f"wrote residual smoke diagnostic with {result.n_pairs} pairs and "
        f"{result.n_features} residual features to {result.out_dir}"
    )


@app.command(
    "check-sae-compatibility",
    help="Check semantic SAE compatibility; shape-only diagnostic is not production.",
)
def check_sae_compatibility_command(
    sae_release: Annotated[str, typer.Option()],
    sae_id: Annotated[str, typer.Option()],
    model: Annotated[str, typer.Option()] = "EleutherAI/pythia-70m-deduped",
    hook_point: Annotated[str, typer.Option()] = "blocks.2.hook_resid_post",
    device: Annotated[str, typer.Option()] = "cpu",
    out: Annotated[Path, typer.Option()] = Path("runs/check_sae_compatibility.json"),
    require_metadata_match: Annotated[
        bool,
        typer.Option(
            "--require-metadata-match/--no-require-metadata-match",
            help="Require SAE-declared model and hook metadata to match.",
        ),
    ] = True,
    allow_shape_only_diagnostic: Annotated[
        bool,
        typer.Option(
            help="Emit shape-only diagnostic fields; not production-compatible.",
        ),
    ] = False,
    allow_metadata_mismatch: Annotated[
        bool,
        typer.Option(
            help=(
                "Record metadata mismatch as diagnostic-only. This cannot support "
                "candidate evidence."
            )
        ),
    ] = False,
    max_reconstruction_l2_relative: Annotated[float | None, typer.Option()] = None,
    max_reconstruction_mse: Annotated[float | None, typer.Option()] = None,
) -> None:
    result = verify_sae_compatibility(
        model_name=model,
        hook_point=hook_point,
        sae_release=sae_release,
        sae_id=sae_id,
        device=device,
        out=out,
        require_metadata_match=require_metadata_match,
        allow_shape_only_diagnostic=allow_shape_only_diagnostic,
        allow_metadata_mismatch=allow_metadata_mismatch,
        max_reconstruction_l2_relative=max_reconstruction_l2_relative,
        max_reconstruction_mse=max_reconstruction_mse,
    )
    console.print_json(data=result.model_dump())
    if not result.compatible:
        raise typer.Exit(code=1)


@app.command(
    "run-sae-intervention",
    help="Run decoded SAE intervention after semantic compatibility checks.",
)
def run_sae_intervention_command(
    sae_release: Annotated[str, typer.Option()],
    sae_id: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
    ranking_dir: Annotated[Path | None, typer.Option()] = None,
    pairs: Annotated[Path | None, typer.Option()] = None,
    model: Annotated[str, typer.Option()] = "EleutherAI/pythia-70m-deduped",
    hook_point: Annotated[str, typer.Option()] = "blocks.2.hook_resid_post",
    per_family: Annotated[int, typer.Option(min=1)] = 15,
    seed: Annotated[int, typer.Option()] = 7,
    top_k_features: Annotated[int, typer.Option(min=1)] = 5,
    operation: Annotated[str, typer.Option()] = "ablate",
    factor: Annotated[float, typer.Option()] = 1.0,
    patch_mode: Annotated[str, typer.Option()] = "delta",
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    if operation not in {"ablate", "amplify"}:
        raise typer.BadParameter("--operation must be ablate or amplify")
    if patch_mode not in {"replace", "delta"}:
        raise typer.BadParameter("--patch-mode must be replace or delta")
    if operation == "amplify" and factor == 1.0:
        raise typer.BadParameter("--operation amplify requires --factor not equal to 1.0")
    result = run_real_sae_intervention(
        out_dir=out,
        ranking_dir=ranking_dir,
        pairs_path=pairs,
        per_family=per_family,
        seed=seed,
        model_name=model,
        hook_point=hook_point,
        sae_release=sae_release,
        sae_id=sae_id,
        top_k_features=top_k_features,
        operation=operation,  # type: ignore[arg-type]
        factor=factor,
        patch_mode=patch_mode,  # type: ignore[arg-type]
        device=device,
    )
    console.print_json(
        data={
            "out_dir": str(result.out_dir),
            "n_pairs": result.n_pairs,
            "n_features": result.n_features,
            "operation": result.operation,
            "patch_mode": result.patch_mode,
            "top_features": result.top_features,
            "compatible": result.compatible,
        }
    )
    if not result.compatible:
        raise typer.Exit(code=1)


@app.command(
    "run-phase3-behavioral-evaluation",
    help="Run Phase 3 token-contrast evaluation for decoded SAE interventions.",
)
def run_phase3_behavioral_evaluation_command(
    ranking_dir: Annotated[Path, typer.Option()],
    sae_release: Annotated[str, typer.Option()],
    sae_id: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()] = Path("runs/test_phase3_behavioral_evaluation"),
    tasks: Annotated[Path | None, typer.Option()] = None,
    task_source: Annotated[str, typer.Option()] = "generated",
    task_file: Annotated[Path | None, typer.Option()] = None,
    task_bank_calibration_dir: Annotated[Path | None, typer.Option()] = None,
    task_source_id: Annotated[str | None, typer.Option()] = None,
    per_family: Annotated[int, typer.Option(min=1)] = 10,
    seed: Annotated[int, typer.Option()] = 7,
    model: Annotated[str, typer.Option()] = "EleutherAI/pythia-70m-deduped",
    hook_point: Annotated[str, typer.Option()] = "blocks.2.hook_resid_post",
    top_k_features: Annotated[int, typer.Option(min=1)] = 5,
    baseline_mode: Annotated[str, typer.Option()] = "top-vs-random-multiseed",
    random_seeds: Annotated[str, typer.Option()] = "7,11,13",
    operations: Annotated[str, typer.Option()] = "ablate",
    amplify_factors: Annotated[str, typer.Option("--amplify-factors")] = "2.0",
    patch_mode: Annotated[str, typer.Option()] = "delta",
    token_position: Annotated[int, typer.Option()] = -1,
    device: Annotated[str, typer.Option()] = "cpu",
    reduction: Annotated[str, typer.Option()] = "mean",
    min_valid_tasks_per_family: Annotated[int, typer.Option(min=1)] = 2,
    allow_metadata_mismatch: Annotated[
        bool,
        typer.Option(
            help=(
                "Run diagnostic-only if SAE metadata mismatches. This cannot support "
                "candidate evidence."
            )
        ),
    ] = False,
    write_report: Annotated[bool, typer.Option("--write-report/--no-write-report")] = True,
    max_relative_norm_drift_warning: Annotated[float, typer.Option()] = 0.5,
    max_decoded_delta_norm_ratio_warning: Annotated[float, typer.Option()] = 0.5,
    density_tolerance: Annotated[float, typer.Option()] = 0.10,
    abs_mean_tolerance: Annotated[float, typer.Option()] = 0.10,
    allow_relaxed_density_matching: Annotated[
        bool,
        typer.Option(
            "--allow-relaxed-density-matching/--no-allow-relaxed-density-matching"
        ),
    ] = True,
    task_calibration_mode: Annotated[
        str,
        typer.Option(
            help="Baseline-only calibration: none, baseline-intended-direction, or baseline-margin."
        ),
    ] = "none",
    min_baseline_margin: Annotated[float | None, typer.Option()] = None,
    min_calibrated_tasks_per_family: Annotated[int, typer.Option(min=1)] = 3,
    allow_family_drop: Annotated[
        bool,
        typer.Option("--allow-family-drop/--no-allow-family-drop"),
    ] = False,
    feature_selection_mode: Annotated[
        str,
        typer.Option(
            help="Top feature mode: top, top-positive, top-absolute, top-family-consistent."
        ),
    ] = "top",
    min_family_consistency: Annotated[int, typer.Option(min=1)] = 3,
) -> None:
    if task_source not in {"generated", "file"}:
        raise typer.BadParameter("--task-source must be generated or file")
    if task_source == "file" and task_file is None:
        raise typer.BadParameter("--task-source=file requires --task-file")
    result = run_real_behavioral_sae_intervention(
        out_dir=out,
        ranking_dir=ranking_dir,
        tasks_path=tasks,
        task_source=task_source,  # type: ignore[arg-type]
        task_file=task_file,
        task_bank_calibration_dir=task_bank_calibration_dir,
        task_source_id=task_source_id,
        per_family=per_family,
        seed=seed,
        model_name=model,
        hook_point=hook_point,
        sae_release=sae_release,
        sae_id=sae_id,
        top_k_features=top_k_features,
        baseline_mode=baseline_mode,  # type: ignore[arg-type]
        random_seeds=parse_int_list(random_seeds),
        operations=parse_operations(operations),
        amplify_factors=parse_amplify_factors(amplify_factors),
        patch_mode=patch_mode,  # type: ignore[arg-type]
        token_position=token_position,
        device=device,
        reduction=reduction,  # type: ignore[arg-type]
        min_valid_tasks_per_family=min_valid_tasks_per_family,
        allow_metadata_mismatch=allow_metadata_mismatch,
        write_report=write_report,
        max_relative_norm_drift_warning=max_relative_norm_drift_warning,
        max_decoded_delta_norm_ratio_warning=max_decoded_delta_norm_ratio_warning,
        density_tolerance=density_tolerance,
        abs_mean_tolerance=abs_mean_tolerance,
        allow_relaxed_density_matching=allow_relaxed_density_matching,
        task_calibration_mode=task_calibration_mode,  # type: ignore[arg-type]
        min_baseline_margin=min_baseline_margin,
        min_calibrated_tasks_per_family=min_calibrated_tasks_per_family,
        allow_family_drop=allow_family_drop,
        feature_selection_mode=feature_selection_mode,  # type: ignore[arg-type]
        min_family_consistency=min_family_consistency,
    )
    console.print_json(data=result.__dict__ | {"out_dir": str(result.out_dir)})
    if not result.compatible or not result.task_validation_passed or result.n_rows == 0:
        raise typer.Exit(code=1)


@app.command("summarize-run")
def summarize_run(run_dir: Path) -> None:
    pairs_path = run_dir / "pairs.jsonl"
    pair_count = len(read_minimal_pairs(pairs_path)) if pairs_path.exists() else 0
    proxy_path = run_dir / "feature_space_proxy_results.jsonl"
    top_examples_path = run_dir / "top_examples.jsonl"
    intervention_path = run_dir / "intervention_results.jsonl"
    row_count = 0
    row_label = "rows"
    if intervention_path.exists():
        row_count = len(read_jsonl(intervention_path))
        row_label = "intervention_rows"
    elif proxy_path.exists():
        row_count = len(read_jsonl(proxy_path))
        row_label = "feature_space_proxy_rows"
    elif top_examples_path.exists():
        row_count = len(read_jsonl(top_examples_path))
        row_label = "top_example_rows"
    table = Table(title=str(run_dir))
    table.add_column("artifact")
    table.add_column("count")
    table.add_row("pairs", str(pair_count))
    table.add_row(row_label, str(row_count))
    console.print(table)


if __name__ == "__main__":
    app()
