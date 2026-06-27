from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from local_mi_lab.paths import resolve_repo_path


def load_results(run_dirs: list[str | Path]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        root = resolve_repo_path(run_dir)
        path = root / "heldout_robustness_results.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        frame = pd.read_csv(path)
        frame["source_run"] = str(root)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    rows = pd.concat(frames, ignore_index=True)
    rows["effect_size"] = pd.to_numeric(rows["effect_size"], errors="coerce")
    return rows


def parse_candidate(candidate: str) -> tuple[int, int]:
    text = candidate.strip().upper()
    if not text.startswith("L") or "H" not in text:
        raise ValueError("Candidate must look like L7H7")
    layer_text, head_text = text[1:].split("H", 1)
    return int(layer_text), int(head_text)


def inspect_candidate(rows: pd.DataFrame, candidate: str) -> tuple[pd.DataFrame, str]:
    layer, head = parse_candidate(candidate)
    subset = rows[(rows["layer"] == layer) & (rows["head"] == head)].copy()
    if subset.empty:
        raise ValueError(f"No held-out rows found for {candidate}")
    subset["inspection_bucket"] = subset.apply(_bucket_row, axis=1)
    selected = pd.concat(
        [
            _top_rows(subset, "strongest_positive_success", ascending=False),
            _top_rows(subset, "strongest_positive_failure", ascending=True),
            _top_rows(subset, "controls_that_moved", ascending=False),
            _top_rows(subset, "wrong_target_failure", ascending=False),
        ],
        ignore_index=True,
    ).drop_duplicates(
        subset=[
            "seed",
            "example_id",
            "family",
            "intervention",
            "position_label",
            "effect_size",
        ]
    )
    markdown = render_counterexample_markdown(candidate, subset, selected)
    return selected, markdown


def write_counterexample_artifacts(
    candidate: str,
    rows: pd.DataFrame,
    markdown: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = resolve_repo_path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe = candidate.upper()
    csv_path = root / f"counterexamples_{safe}.csv"
    md_path = root / f"counterexamples_{safe}.md"
    rows.to_csv(csv_path, index=False)
    md_path.write_text(markdown, encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path}


def render_counterexample_markdown(
    candidate: str,
    subset: pd.DataFrame,
    selected: pd.DataFrame,
) -> str:
    status_counts = subset["inspection_bucket"].value_counts().to_dict()
    failures = _family_failure_lines(subset)
    intervention = _intervention_lines(subset)
    controls = selected[selected["inspection_bucket"] == "controls_that_moved"]
    wrong = selected[selected["inspection_bucket"] == "wrong_target_failure"]
    return "\n".join(
        [
            f"# Held-Out Counterexamples: {candidate.upper()}",
            "",
            "## Why this head was inspected",
            "",
            "This head was part of the fixed held-out candidate set after the earlier "
            "head-specific sweep. The goal is to inspect where the held-out causal result "
            "succeeds, fails, or moves controls.",
            "",
            "## Strongest successes",
            "",
            _rows_table(selected[selected["inspection_bucket"] == "strongest_positive_success"]),
            "",
            "## Strongest failures",
            "",
            _rows_table(selected[selected["inspection_bucket"] == "strongest_positive_failure"]),
            "",
            "## Controls that moved",
            "",
            _rows_table(controls),
            "",
            "## Prompt constructions that broke it",
            "",
            "\n".join(failures) if failures else "No family-level failures were identified.",
            "",
            "## Intervention/position sensitivity",
            "",
            "\n".join(intervention) if intervention else "No intervention summary was available.",
            "",
            "## Wrong-target failures",
            "",
            _rows_table(wrong),
            "",
            "## Interpretation",
            "",
            f"Inspection buckets: `{status_counts}`.",
            "",
            "A useful candidate should move positive examples more than controls across "
            "held-out families and intervention variants. Rows where controls move or "
            "positive effects vanish are counterexamples for this lab stage.",
            "",
            "## What not to claim",
            "",
            "Do not claim induction-head discovery, a circuit, or broad GPT-2 behavior from "
            "these examples. This is counterexample-oriented practice evidence.",
            "",
        ]
    )


def _bucket_row(row: pd.Series) -> str:
    effect = row["effect_size"]
    if pd.isna(effect):
        return "invalid_or_unavailable"
    if row["family"] == "heldout_wrong_target_same_prompt" and effect > 0:
        return "wrong_target_failure"
    if row["heldout_family_type"] == "control" and effect > 0:
        return "controls_that_moved"
    if row["heldout_family_type"] == "positive" and effect > 0:
        return "strongest_positive_success"
    if row["heldout_family_type"] == "positive" and effect <= 0:
        return "strongest_positive_failure"
    return "other"


def _top_rows(subset: pd.DataFrame, bucket: str, *, ascending: bool) -> pd.DataFrame:
    rows = subset[subset["inspection_bucket"] == bucket].copy()
    if rows.empty:
        return rows
    return rows.sort_values("effect_size", ascending=ascending).head(12)


def _rows_table(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "No rows in this bucket."
    cols = [
        "seed",
        "family",
        "example_id",
        "intervention",
        "position_label",
        "effect_size",
        "true_expected_next_token",
        "wrong_or_control_token",
        "clean_prompt",
        "corrupt_prompt",
    ]
    view = rows[cols].copy()
    view["effect_size"] = view["effect_size"].map(lambda value: f"{float(value):.4f}")
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, separator]
    for row in view.itertuples(index=False):
        values = [str(value).replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _family_failure_lines(subset: pd.DataFrame) -> list[str]:
    rows = subset.copy()
    rows["effect_size"] = pd.to_numeric(rows["effect_size"], errors="coerce")
    grouped = rows.groupby(["family", "heldout_family_type"], as_index=False).agg(
        mean_effect=("effect_size", "mean"),
        n_valid=("effect_size", lambda values: int(values.notna().sum())),
        n_positive=("effect_size", lambda values: int((values > 0).sum())),
    )
    lines = []
    for row in grouped.sort_values("mean_effect").itertuples(index=False):
        if row.heldout_family_type == "positive" and row.mean_effect <= 0:
            lines.append(
                f"- {row.family}: positive-family mean effect `{row.mean_effect:.4f}` "
                f"over {row.n_valid} valid rows."
            )
        if row.heldout_family_type == "control" and row.mean_effect > 0:
            lines.append(
                f"- {row.family}: control-family mean effect `{row.mean_effect:.4f}` "
                f"over {row.n_valid} valid rows."
            )
    return lines


def _intervention_lines(subset: pd.DataFrame) -> list[str]:
    rows = subset.copy()
    rows["effect_size"] = pd.to_numeric(rows["effect_size"], errors="coerce")
    grouped = rows.groupby(["intervention", "position_label"], as_index=False).agg(
        mean_effect=("effect_size", "mean"),
        n_valid=("effect_size", lambda values: int(values.notna().sum())),
        n_controls_moved=(
            "inspection_bucket",
            lambda values: int((values == "controls_that_moved").sum()),
        ),
    )
    return [
        (
            f"- {row.intervention} at {row.position_label}: mean effect "
            f"`{row.mean_effect:.4f}`, valid rows `{row.n_valid}`, controls moved "
            f"`{row.n_controls_moved}`."
        )
        for row in grouped.itertuples(index=False)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument(
        "--output",
        default="reports/head_specific_induction_heldout_robustness_v1",
    )
    args = parser.parse_args()
    rows = load_results(args.runs)
    selected, markdown = inspect_candidate(rows, args.candidate)
    paths = write_counterexample_artifacts(args.candidate, selected, markdown, args.output)
    print(paths["markdown"])


if __name__ == "__main__":
    main()
