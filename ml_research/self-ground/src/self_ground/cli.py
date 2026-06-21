from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from self_ground.experiment import run_negation_experiment
from self_ground.io import read_jsonl, read_minimal_pairs, write_jsonl
from self_ground.negation import generate_negation_pairs

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


@app.command("run-negation")
def run_negation(
    pairs: Annotated[Path, typer.Option()],
    out: Annotated[Path, typer.Option()],
    model: Annotated[str, typer.Option()] = "gpt2-small",
    layer: Annotated[str, typer.Option()] = "blocks.8.hook_resid_post",
    sae_release: Annotated[str | None, typer.Option()] = None,
    sae_id: Annotated[str | None, typer.Option()] = None,
    top_k_features: Annotated[int, typer.Option(min=1)] = 20,
    device: Annotated[str | None, typer.Option()] = None,
) -> None:
    if not sae_release or not sae_id:
        raise typer.BadParameter("run-negation requires --sae-release and --sae-id")

    result = run_negation_experiment(
        pairs_path=pairs,
        out_dir=out,
        model_name=model,
        layer=layer,
        sae_release=sae_release,
        sae_id=sae_id,
        top_k_features=top_k_features,
        device=device,
    )
    console.print(f"wrote run with {result.n_pairs} pairs to {result.out_dir}")


@app.command("summarize-run")
def summarize_run(run_dir: Path) -> None:
    pairs = read_minimal_pairs(run_dir / "pairs.jsonl")
    interventions = read_jsonl(run_dir / "intervention_results.jsonl")
    table = Table(title=str(run_dir))
    table.add_column("artifact")
    table.add_column("count")
    table.add_row("pairs", str(len(pairs)))
    table.add_row("intervention_rows", str(len(interventions)))
    console.print(table)


if __name__ == "__main__":
    app()
