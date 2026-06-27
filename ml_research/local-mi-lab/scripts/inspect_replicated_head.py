from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from local_mi_lab.head_patching import parse_head_spec
from local_mi_lab.paths import resolve_repo_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", required=True, help="Head spec such as L7H7.")
    parser.add_argument("--runs", nargs="+", required=True, help="Head-specific sweep run dirs.")
    parser.add_argument(
        "--output",
        default="reports/head_specific_induction_causality_v1",
        help="Output directory for inspection artifacts.",
    )
    args = parser.parse_args()
    layer, head = parse_head_spec(args.head)
    output = resolve_repo_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, family_rows, attention_rows = inspect_head(layer, head, args.runs)
    label = f"L{layer}H{head}"
    examples_path = output / f"replicated_head_{label}_examples.csv"
    markdown_path = output / f"replicated_head_{label}_inspection.md"
    pd.DataFrame(rows).to_csv(examples_path, index=False)
    markdown_path.write_text(
        render_inspection_markdown(label, args.runs, rows, family_rows, attention_rows),
        encoding="utf-8",
    )
    print(markdown_path)


def inspect_head(
    layer: int,
    head: int,
    run_dirs: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        root = resolve_repo_path(run_dir)
        by_head = pd.read_csv(root / "head_specific_patching_by_head.csv")
        selected = by_head[(by_head["layer"] == layer) & (by_head["head"] == head)]
        if selected.empty:
            continue
        result_rows = pd.read_csv(root / "head_specific_patching_results.csv")
        result_rows = result_rows[
            (result_rows["layer"] == layer) & (result_rows["head"] == head)
        ].copy()
        result_rows["effect_size_numeric"] = pd.to_numeric(
            result_rows["effect_size"],
            errors="coerce",
        )
        for row in selected.itertuples(index=False):
            family_rows.append(
                {
                    "run": root.name,
                    "seed": int(row.seed),
                    "layer": int(row.layer),
                    "head": int(row.head),
                    "positive_mean_effect_size": float(row.positive_mean_effect_size),
                    "max_control_mean_effect_size": float(row.max_control_mean_effect_size),
                    "positive_minus_control_effect_gap": float(
                        row.positive_minus_control_effect_gap
                    ),
                    "hardest_control_family": row.hardest_control_family,
                    "specificity_status": row.specificity_status,
                }
            )
        rows.extend(_example_rows(root.name, result_rows))
        attention_rows.extend(_attention_summary_rows(root, layer, head))
    return rows, family_rows, attention_rows


def render_inspection_markdown(
    label: str,
    run_dirs: list[str],
    rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    attention_rows: list[dict[str, Any]],
) -> str:
    examples = pd.DataFrame(rows)
    family = pd.DataFrame(family_rows)
    attention = pd.DataFrame(attention_rows)
    positive = examples[examples["family"] == "positive_repeat_sequence"].copy()
    controls = examples[examples["family"] != "positive_repeat_sequence"].copy()
    positive_top = positive.sort_values("effect_size", ascending=False).head(5)
    positive_failures = positive.sort_values("effect_size", ascending=True).head(5)
    control_moved = controls.sort_values("effect_size", ascending=False).head(5)
    return "\n".join(
        [
            f"# Replicated Head Inspection: {label}",
            "",
            "## Scope",
            "",
            "This is a manual-inspection front end for existing head-specific patching artifacts. It does not run a new intervention and does not establish a mechanism claim.",
            "",
            "## Runs",
            "",
            *[f"- `{run_dir}`" for run_dir in run_dirs],
            "",
            "## Seed Summary",
            "",
            *_markdown_table(
                family,
                [
                    "seed",
                    "positive_mean_effect_size",
                    "max_control_mean_effect_size",
                    "positive_minus_control_effect_gap",
                    "specificity_status",
                ],
            ),
            "",
            "## Attention Pattern Context",
            "",
            *_markdown_table(
                attention,
                ["run", "family", "mean_attention_to_previous_occurrence", "n_examples"],
            ),
            "",
            "## Strong Positive Examples",
            "",
            *_markdown_table(
                positive_top,
                ["run", "seed", "example_id", "effect_size", "clean_prompt", "corrupt_prompt"],
            ),
            "",
            "## Positive Failures",
            "",
            *_markdown_table(
                positive_failures,
                ["run", "seed", "example_id", "effect_size", "clean_prompt", "corrupt_prompt"],
            ),
            "",
            "## Controls That Moved",
            "",
            *_markdown_table(
                control_moved,
                ["run", "seed", "family", "example_id", "effect_size", "corrupt_prompt"],
            ),
            "",
            "## What To Check Manually",
            "",
            "- Are the positive examples using the intended repeated-prefix structure?",
            "- Do the controls that moved reveal a prompt artifact?",
            "- Does the head attention pattern line up with the causal effect, or are they separate signals?",
            "- Does the result remain plausible after ignoring any prior random-comparison label?",
            "",
            "## What Not To Claim",
            "",
            "Do not claim an induction head or circuit from this inspection. This is a narrow replicated candidate under selected prompts, controls, hook, position, and metric.",
            "",
        ]
    )


def _example_rows(run_name: str, rows: pd.DataFrame) -> list[dict[str, Any]]:
    keep = [
        "seed",
        "example_id",
        "family",
        "control_family",
        "effect_size_numeric",
        "effect_size_status",
        "clean_prompt",
        "corrupt_prompt",
        "true_expected_next_token",
        "wrong_or_control_token",
    ]
    subset = rows[keep].rename(columns={"effect_size_numeric": "effect_size"})
    subset.insert(0, "run", run_name)
    return subset.to_dict("records")


def _attention_summary_rows(root: Path, layer: int, head: int) -> list[dict[str, Any]]:
    source = _source_run(root)
    path = source / "attention_patterns_by_head.csv"
    if not path.exists():
        return []
    rows = pd.read_csv(path)
    rows = rows[(rows["layer"] == layer) & (rows["head"] == head)].copy()
    if rows.empty or "family" not in rows:
        return []
    rows["attention_numeric"] = pd.to_numeric(
        rows["attention_to_previous_occurrence"],
        errors="coerce",
    )
    grouped = rows.groupby("family", as_index=False).agg(
        mean_attention_to_previous_occurrence=("attention_numeric", "mean"),
        n_examples=("example_id", "nunique"),
    )
    grouped.insert(0, "run", root.name)
    return grouped.to_dict("records")


def _source_run(root: Path) -> Path:
    source_path = root / "source_run.txt"
    if not source_path.exists():
        return root
    text = source_path.read_text(encoding="utf-8").strip()
    return Path(text) if text else root


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["No rows."]
    rows = df.copy()
    for column in columns:
        if column not in rows:
            rows[column] = ""
    rows = rows[columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return lines


if __name__ == "__main__":
    main()
