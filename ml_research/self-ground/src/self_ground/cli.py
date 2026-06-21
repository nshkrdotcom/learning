from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from self_ground.io import read_jsonl, read_minimal_pairs, write_jsonl
from self_ground.negation import generate_negation_pairs
from self_ground.real_model_check import check_real_model
from self_ground.real_ranking import run_activation_ranking
from self_ground.real_residual_intervention import run_real_residual_intervention

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
    feature_source: Annotated[str, typer.Option()] = "residual_dimensions",
    pooling: Annotated[str, typer.Option()] = "final_token",
    per_family: Annotated[int, typer.Option(min=1)] = 15,
    seed: Annotated[int, typer.Option()] = 7,
    top_k_features: Annotated[int, typer.Option(min=1)] = 50,
    device: Annotated[str, typer.Option()] = "cpu",
    sae_release: Annotated[str | None, typer.Option()] = None,
    sae_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    if feature_source == "sae" and (not sae_release or not sae_id):
        raise typer.BadParameter("feature_source=sae requires --sae-release and --sae-id")
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
    )
    console.print(
        f"wrote {result.feature_source} ranking with {result.n_pairs} pairs "
        f"and {result.n_features} features to {result.out_dir}"
    )


@app.command("run-residual-intervention")
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
        f"wrote residual intervention with {result.n_pairs} pairs and "
        f"{result.n_features} residual features to {result.out_dir}"
    )


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
