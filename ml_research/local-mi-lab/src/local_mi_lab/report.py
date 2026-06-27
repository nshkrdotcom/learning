from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from local_mi_lab.paths import relative_files, resolve_repo_path

MANDATORY_LANGUAGE = [
    "This is not a mechanism claim unless causal intervention evidence supports it.",
    "This is not a broad model claim.",
    "This is a local MI practice run.",
]


def read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def generate_run_summary(run_dir: str | Path) -> str:
    root = resolve_repo_path(run_dir)
    config = read_json_if_present(root / "capability_report.json")
    baseline = read_json_if_present(root / "baseline_metrics.json")
    activation_manifest = read_json_if_present(root / "activations" / "manifest.json")
    activation_summary = read_json_if_present(root / "activation_summary.json")
    logit_lens = read_json_if_present(root / "logit_lens_summary.json")
    patching_metadata = read_json_if_present(root / "patching_metadata.json")
    files = relative_files(root) if root.exists() else []

    lines = [
        "# Run Summary",
        "",
        "## Question",
        "",
        _question_text(root),
        "",
        "## Model and resources",
        "",
        _model_resources_text(config, baseline, activation_manifest),
        "",
        "## Files present",
        "",
        *_files_text(files),
        "",
        "## Baseline behavior",
        "",
        _baseline_text(baseline),
        "",
        "## Activation cache",
        "",
        _activation_text(activation_manifest or activation_summary),
        "",
        "## Logit lens",
        "",
        _logit_lens_text(logit_lens),
        "",
        "## Activation patching",
        "",
        _patching_text(root, patching_metadata),
        "",
        "## What this shows",
        "",
        "This is a local MI practice run. It shows only the artifacts present in this run directory.",
        "",
        "## What this does not show",
        "",
        *MANDATORY_LANGUAGE[:2],
        "",
        "## Next step",
        "",
        _next_step_text(baseline, activation_manifest, logit_lens),
        "",
        MANDATORY_LANGUAGE[2],
    ]
    return "\n".join(lines) + "\n"


def write_run_summary(run_dir: str | Path) -> Path:
    root = resolve_repo_path(run_dir)
    summary = generate_run_summary(root)
    output = root / "summary.md"
    output.write_text(summary, encoding="utf-8")
    return output


def _question_text(root: Path) -> str:
    if (root / "prompts.csv").exists():
        return "Does the model show a simple expected-token behavior on the prompt set?"
    if (root / "capability_report.json").exists():
        return "Can the configured model and local environment run the first-pass workflow?"
    return "Missing: no question artifact was found."


def _model_resources_text(
    capability: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
    activation: dict[str, Any] | None,
) -> str:
    if capability:
        status = capability.get("status", "unknown")
        model = capability.get("model", "unknown")
        return f"Capability report present for `{model}` with status `{status}`."
    if baseline:
        return f"Baseline metrics present for `{baseline.get('model', 'unknown')}`."
    if activation:
        return f"Activation manifest present for `{activation.get('model', 'unknown')}`."
    return "Missing: no model/resource artifact was found."


def _files_text(files: list[str]) -> list[str]:
    if not files:
        return ["Missing: run directory has no files."]
    return [f"- `{file}`" for file in files]


def _baseline_text(baseline: dict[str, Any] | None) -> str:
    if baseline is None:
        return "Missing: `baseline_metrics.json` was not found."
    return (
        f"Examples: {baseline.get('n_examples')}. "
        f"Mean expected probability: {baseline.get('mean_expected_probability')}. "
        f"Median expected rank: {baseline.get('median_expected_rank')}. "
        f"Mean probability diff versus control: {baseline.get('mean_probability_diff_vs_control')}."
    )


def _activation_text(manifest: dict[str, Any] | None) -> str:
    if manifest is None:
        return "Missing: activation manifest was not found."
    layers = manifest.get("layers", [])
    files = manifest.get("files", [])
    return f"Selected activation cache present for layers {layers}; tensor files: {len(files)}."


def _logit_lens_text(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "Missing: `logit_lens_summary.json` was not found."
    return (
        "Logit lens summary present. "
        f"Best layer by mean probability: {summary.get('best_layer_by_mean_probability')}. "
        "This is descriptive, not causal evidence."
    )


def _patching_text(root: Path, metadata: dict[str, Any] | None) -> str:
    if not (root / "patching_results.csv").exists():
        return "Missing: `patching_results.csv` was not found."
    if metadata:
        return (
            f"Patching results present for metric `{metadata.get('metric')}`. "
            f"Exploratory full sweep: {metadata.get('exploratory')}."
        )
    return "Patching results present, but metadata is missing."


def _next_step_text(
    baseline: dict[str, Any] | None,
    activation: dict[str, Any] | None,
    logit_lens: dict[str, Any] | None,
) -> str:
    if baseline is None:
        return "Run baseline behavior measurement before interpretability analysis."
    if activation is None:
        return "Cache selected activations for the successful baseline run."
    if logit_lens is None:
        return "Run logit lens on the selected activation workflow."
    return "Inspect the tables and plots, then run a small explicit activation patching task."
