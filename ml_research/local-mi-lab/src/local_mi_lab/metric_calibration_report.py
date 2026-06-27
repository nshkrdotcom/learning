from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from local_mi_lab.metric_calibration import (
    aggregate_metric_calibration_by_family,
    build_calibration_prompt_bank,
    score_metric_calibration_examples,
    summarize_metric_calibration,
)
from local_mi_lab.models import load_hooked_transformer
from local_mi_lab.paths import repo_root, resolve_repo_path


def run_metric_calibration(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
    tracked_summary_path: str | Path = "docs/results/induction_metric_calibration_v1.md",
    learning_note_path: str | Path = "docs/learning_notes/2026-06-26_induction_metric_calibration.md",
) -> dict[str, Path]:
    output_root = resolve_repo_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    prompts = build_calibration_prompt_bank(seed=int(config.get("experiment", {}).get("seed", 0)))
    model = load_hooked_transformer(config)
    model.eval()
    by_example = score_metric_calibration_examples(model, model.tokenizer, prompts)
    by_family = aggregate_metric_calibration_by_family(by_example)
    summary = summarize_metric_calibration(by_example, by_family)
    summary["source_spec"] = "docs/experiments/induction_metric_calibration_v1.md"
    summary["command"] = (
        "uv run python scripts/run_metric_calibration.py "
        f"--output {_display_path(output_root)}"
    )

    by_example_path = output_root / "metric_calibration_by_example.csv"
    by_family_path = output_root / "metric_calibration_by_family.csv"
    summary_path = output_root / "metric_calibration_summary.json"
    markdown_path = output_root / "induction_metric_calibration_v1.md"
    tracked_path = resolve_repo_path(tracked_summary_path)
    learning_path = resolve_repo_path(learning_note_path)
    tracked_path.parent.mkdir(parents=True, exist_ok=True)
    learning_path.parent.mkdir(parents=True, exist_ok=True)

    by_example.to_csv(by_example_path, index=False)
    by_family.to_csv(by_family_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown = render_metric_calibration_markdown(summary, by_family)
    markdown_path.write_text(markdown, encoding="utf-8")
    tracked_path.write_text(markdown, encoding="utf-8")
    learning_path.write_text(render_metric_calibration_learning_note(summary, by_family), encoding="utf-8")
    return {
        "by_example": by_example_path,
        "by_family": by_family_path,
        "summary": summary_path,
        "markdown": markdown_path,
        "tracked_summary": tracked_path,
        "learning_note": learning_path,
    }


def render_metric_calibration_markdown(summary: dict[str, Any], by_family: pd.DataFrame) -> str:
    status = summary["status"]
    lines = [
        "# Induction Metric Calibration v1",
        "",
        "## Source",
        "",
        "- Spec: `docs/experiments/induction_metric_calibration_v1.md`",
        f"- Command: `{summary.get('command', '')}`",
        "- Model: `gpt2-small`",
        "",
        "## Executive Summary",
        "",
        _status_sentence(status),
        "",
        "## Primary Metric",
        "",
        "`true_vs_control_logit_diff`.",
        "",
        "## Thresholds",
        "",
        _dict_table(summary["thresholds"], "threshold", "value"),
        "",
        "## Overall Separation",
        "",
        f"- Positive mean: `{summary['positive_mean_true_vs_control_logit_diff']}`",
        f"- Max control mean: `{summary['max_control_mean_true_vs_control_logit_diff']}`",
        f"- Positive-minus-max-control gap: `{summary['positive_minus_max_control_gap']}`",
        f"- Weakest positive family mean: `{summary['weakest_positive_family_mean']}`",
        f"- Hardest control family: `{summary['hardest_control_family']}`",
        "",
        "## Family Summary",
        "",
        _family_table(by_family),
        "",
        "## Domain and Length Checks",
        "",
        f"- Positive domain means: `{summary['positive_domain_means']}`",
        f"- Positive length means: `{summary['positive_length_means']}`",
        "",
        "## Decision",
        "",
        f"Final status: `{status}`.",
        f"Search allowed: `{summary['search_allowed']}`.",
        "",
        "## Interpretation",
        "",
        _interpretation(status),
        "",
        "## What This Does Not Show",
        "",
        "This does not show an induction head, a circuit, or a broad GPT-2 property. "
        "It only tests whether one local behavior metric is calibrated enough for future specifications.",
        "",
        "## Exact Next Command",
        "",
        "```bash",
        "less docs/results/induction_metric_calibration_v1.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def render_metric_calibration_learning_note(summary: dict[str, Any], by_family: pd.DataFrame) -> str:
    status = summary["status"]
    hardest = summary.get("hardest_control_family") or {}
    lines = [
        "# Induction Metric Calibration: Practice Note",
        "",
        "## Question",
        "",
        "Can `true_vs_control_logit_diff` separate repeated-token positives from controls before another head search?",
        "",
        "## Result",
        "",
        f"Final status: `{status}`.",
        "",
        "## What Worked",
        "",
        _worked_text(summary),
        "",
        "## What Failed",
        "",
        _failed_text(summary, by_family),
        "",
        "## Hardest Control",
        "",
        f"`{hardest}`",
        "",
        "## What I Learned",
        "",
        "Metric calibration is a prerequisite for candidate search. A good-looking intervention row is not useful if controls score under the same rule.",
        "",
        "## What I Will Not Claim",
        "",
        "I will not claim an induction head, a circuit, or broad GPT-2 behavior from this calibration pass.",
        "",
    ]
    return "\n".join(lines)


def _status_sentence(status: str) -> str:
    if status == "metric_calibrated_for_next_spec":
        return "The metric passed the pre-registered calibration thresholds for this small prompt set."
    if status == "blocked_tokenization":
        return "The calibration is blocked by tokenization failures in required rows."
    if status == "prompt_bank_needs_revision":
        return "The prompt bank did not behave consistently across domain or length slices."
    return "The metric did not pass the pre-registered separation thresholds."


def _interpretation(status: str) -> str:
    if status == "metric_calibrated_for_next_spec":
        return "Calibration success would only permit a tighter next spec, with held-out controls and refusal of mechanism claims."
    if status == "prompt_bank_needs_revision":
        return "Do not search for heads. Revise prompt construction and rerun calibration."
    if status == "blocked_tokenization":
        return "Do not score the metric until tokenization and prompt metadata are fixed."
    return "Do not search for heads. The metric is still false-positive-prone or insufficiently separated from controls."


def _worked_text(summary: dict[str, Any]) -> str:
    gap = summary.get("positive_minus_max_control_gap")
    if gap is not None and float(gap) > 0:
        return f"Positives exceeded the hardest control by `{gap}` on average."
    return "No robust positive-over-control separation was established."


def _failed_text(summary: dict[str, Any], by_family: pd.DataFrame) -> str:
    if summary["status"] == "metric_calibrated_for_next_spec":
        return "No pre-registered failure threshold was crossed in this small calibration."
    controls = by_family[by_family["should_show_induction_behavior"] == False]  # noqa: E712
    if controls.empty:
        return "No control families were available, which itself invalidates calibration."
    hardest = controls.sort_values("mean_true_vs_control_logit_diff", ascending=False).iloc[0]
    return (
        f"The hardest control `{hardest['family']}` had mean diff "
        f"`{hardest['mean_true_vs_control_logit_diff']}`, so controls remain central to interpretation."
    )


def _dict_table(values: dict[str, Any], key_name: str, value_name: str) -> str:
    lines = [f"| {key_name} | {value_name} |", "| --- | --- |"]
    for key, value in values.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _family_table(by_family: pd.DataFrame) -> str:
    if by_family.empty:
        return "No family rows."
    columns = [
        "family",
        "should_show_induction_behavior",
        "mean_true_vs_control_logit_diff",
        "fraction_diff_positive",
        "median_target_rank",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in by_family[columns].to_dict("records"):
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    root = repo_root()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()
