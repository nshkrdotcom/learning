from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from local_mi_lab.config import config_to_yaml, evenly_spaced_layers, experiment_name
from local_mi_lab.head_hooks import resolve_head_patch_site
from local_mi_lab.head_patching import run_head_specific_patching
from local_mi_lab.paths import make_run_dir, resolve_repo_path
from local_mi_lab.plots import plot_head_specific_gaps, plot_status_counts
from local_mi_lab.resources import collect_resource_snapshot


def layers_from_spec(spec: Any, n_layers: int) -> list[int]:
    if spec == "auto_even_6":
        return evenly_spaced_layers(n_layers, 6)
    if spec == "all":
        return list(range(n_layers))
    if isinstance(spec, str) and "," in spec:
        return [int(layer) for layer in spec.split(",") if layer.strip()]
    if isinstance(spec, list):
        return [int(layer) for layer in spec]
    raise ValueError(f"Unsupported head-specific layer spec {spec!r}")


def expand_heads(layers: list[int], n_heads: int) -> list[tuple[int, int]]:
    return [(layer, head) for layer in layers for head in range(n_heads)]


def make_head_sweep_run_dir(config: dict[str, Any]) -> Path:
    return make_run_dir(config["outputs"]["run_root"], experiment_name(config))


def fail_if_existing_outputs(run_dir: Path, *, resume: bool, overwrite: bool) -> None:
    output = run_dir / "head_specific_patching_results.csv"
    if output.exists() and not resume and not overwrite:
        raise FileExistsError(
            f"{output} already exists; pass --resume to skip completed rows or --overwrite to replace"
        )


def completed_row_keys(existing_rows: pd.DataFrame) -> set[tuple[int, int, str, str, str, str]]:
    if existing_rows.empty:
        return set()
    return {
        (
            int(row.layer),
            int(row.head),
            str(row.example_id),
            str(row.family),
            str(row.metric),
            str(row.intervention),
        )
        for row in existing_rows.itertuples()
    }


def summary_status_counts(by_head: pd.DataFrame) -> dict[str, int]:
    if by_head.empty or "specificity_status" not in by_head:
        return {}
    return {str(key): int(value) for key, value in by_head["specificity_status"].value_counts().items()}


def run_head_specific_induction_sweep(
    model: Any,
    config: dict[str, Any],
    source_run: str | Path,
    *,
    output_run: str | Path | None = None,
    layers_override: str | None = None,
    examples_per_family_override: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> Path:
    source = resolve_repo_path(source_run)
    if not (source / "prompts.csv").exists():
        raise FileNotFoundError(f"Source run is missing prompts.csv: {source}")
    run_dir = resolve_repo_path(output_run) if output_run else make_head_sweep_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    fail_if_existing_outputs(run_dir, resume=resume, overwrite=overwrite)

    head_cfg = config.get("head_specific", {})
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    layer_spec = layers_override or head_cfg.get("layers", "auto_even_6")
    layers = layers_from_spec(layer_spec, n_layers)
    heads = expand_heads(layers, n_heads)
    families = list(head_cfg.get("families", []))
    examples_per_family = int(examples_per_family_override or head_cfg.get("examples_per_family", 8))
    metric = str(head_cfg.get("metric", "true_vs_control_logit_diff"))
    intervention = str(head_cfg.get("intervention", "head_clean_to_corrupt_patch"))
    position = str(head_cfg.get("position", "final"))
    seed = int(config["experiment"]["seed"])

    shutil.copyfile(source / "prompts.csv", run_dir / "prompts.csv")
    pd.read_csv(source / "prompts.csv").head(24).to_csv(run_dir / "prompt_sample.csv", index=False)
    (run_dir / "config.yaml").write_text(config_to_yaml(config), encoding="utf-8")
    (run_dir / "source_run.txt").write_text(str(source) + "\n", encoding="utf-8")

    hook_resolution = {
        "source_run": str(source),
        "seed": seed,
        "layers": layers,
        "heads_tested": [f"L{layer}H{head}" for layer, head in heads],
        "metric": metric,
        "intervention": intervention,
        "position": position,
        "families": families,
        "examples_per_family": examples_per_family,
        "resources": collect_resource_snapshot("."),
        "patch_sites": [resolve_head_patch_site(model, layer) for layer in layers],
    }
    (run_dir / "head_hook_resolution.json").write_text(
        json.dumps(hook_resolution, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = run_head_specific_patching(
        model,
        run_dir,
        heads=heads,
        seed=seed,
        families=families,
        examples_per_family=examples_per_family,
        metric=metric,
        intervention=intervention,
        position_label=position,
    )
    summary.update(
        {
            "source_run": str(source),
            "seed": seed,
            "layers_tested": layers,
            "heads_tested": len(heads),
            "metric": metric,
            "intervention": intervention,
            "position": position,
            "examples_per_family": examples_per_family,
        }
    )
    (run_dir / "head_specific_induction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    by_head = pd.read_csv(run_dir / "head_specific_patching_by_head.csv")
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    plot_head_specific_gaps(by_head, figures / "head_specific_head_gaps.png")
    plot_status_counts(by_head, figures / "head_specific_status_counts.png", "Head-specific status counts")
    write_sweep_summary_md(run_dir, summary)
    return run_dir


def write_sweep_summary_md(run_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Head-Specific Induction Sweep Summary",
        "",
        f"Source run: `{summary['source_run']}`",
        f"Seed: `{summary['seed']}`",
        f"Metric: `{summary['metric']}`",
        f"Intervention: `{summary['intervention']}`",
        f"Heads tested: `{summary['heads_tested']}`",
        f"Head-specific patching: `{summary['head_specific_patch']}`",
        f"Patch scopes: `{summary['actual_patch_scopes']}`",
        f"Positive mean effect size: `{summary['positive_mean_effect_size']}`",
        f"Max control mean effect size: `{summary['max_control_mean_effect_size']}`",
        f"Best positive-minus-control effect gap: `{summary['best_positive_minus_control_effect_gap']}`",
        f"Specificity statuses: `{summary['specificity_status_counts']}`",
        "",
        "This is a bounded head-specific practice sweep. It is not a mechanism or circuit claim.",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
