from __future__ import annotations

from pathlib import Path

import pandas as pd

from local_mi_lab.paths import resolve_repo_path


def inspect_candidate_characterization_counterexamples(
    *,
    candidate: str,
    report_dir: str | Path,
    output_dir: str | Path,
    top_k: int = 8,
) -> dict[str, Path]:
    report_root = resolve_repo_path(report_dir)
    output_root = resolve_repo_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    by_candidate = pd.read_csv(report_root / "candidate_characterization_multiseed_by_candidate.csv")
    candidate_row = _candidate_row(by_candidate, candidate)
    candidate_id = str(candidate_row["candidate_id"])
    rows = _load_seed_rows(report_root, candidate_id)
    rows = _attach_prompt_metadata(report_root, rows)
    counterexamples = build_counterexample_table(rows, top_k=top_k)
    csv_path = output_root / f"counterexamples_{candidate}.csv"
    md_path = output_root / f"counterexamples_{candidate}.md"
    counterexamples.to_csv(csv_path, index=False)
    md_path.write_text(
        render_counterexample_markdown(candidate, candidate_row, counterexamples, by_candidate),
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": md_path}


def build_counterexample_table(rows: pd.DataFrame, *, top_k: int = 8) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=_counterexample_columns())
    table = rows.copy()
    table["effect_size_numeric"] = pd.to_numeric(table["effect_size"], errors="coerce")
    positive = table[table["heldout_family_type"] == "positive"].dropna(
        subset=["effect_size_numeric"],
    )
    control = table[table["heldout_family_type"] == "control"].dropna(
        subset=["effect_size_numeric"],
    )
    pieces = [
        _tag(positive.sort_values("effect_size_numeric", ascending=False).head(top_k), "strongest_positive_success"),
        _tag(positive.sort_values("effect_size_numeric", ascending=True).head(top_k), "strongest_positive_failure"),
        _tag(control.sort_values("effect_size_numeric", ascending=False).head(top_k), "control_moved"),
        _tag(
            control[control["family"].astype(str).str.contains("target_swap|wrong_target", regex=True)]
            .sort_values("effect_size_numeric", ascending=False)
            .head(top_k),
            "wrong_target_control_moved",
        ),
        _tag(_domain_failures(positive, top_k), "token_domain_or_length_failure"),
        _tag(_position_intervention_mismatches(positive, top_k), "position_intervention_mismatch"),
    ]
    combined = pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=_counterexample_columns())
    return combined[_counterexample_columns()].drop_duplicates()


def render_counterexample_markdown(
    candidate: str,
    candidate_row: pd.Series,
    rows: pd.DataFrame,
    by_candidate: pd.DataFrame,
) -> str:
    negative = by_candidate[by_candidate["candidate_group"].astype(str).str.startswith("negative_control")]
    negative_top = (
        negative.sort_values("mean_positive_minus_control_gap", ascending=False)
        .head(5)[["layer", "head", "mean_positive_minus_control_gap", "final_characterization_status"]]
        .to_dict("records")
    )
    lines = [
        f"# Candidate Characterization Counterexamples: {candidate}",
        "",
        "## Why this candidate was inspected",
        "",
        (
            f"{candidate} was in the fixed primary candidate set from the prior held-out pass. "
            f"Its final characterization status was `{candidate_row['final_characterization_status']}`."
        ),
        "",
        "## Strongest successes",
        "",
        _rows_table(rows, "strongest_positive_success"),
        "",
        "## Strongest failures",
        "",
        _rows_table(rows, "strongest_positive_failure"),
        "",
        "## Controls that moved",
        "",
        _rows_table(rows, "control_moved"),
        "",
        "## Negative-control comparison",
        "",
        f"Top negative-control rows by mean gap: `{negative_top}`.",
        "",
        "## Token domains where it failed",
        "",
        _rows_table(rows, "token_domain_or_length_failure"),
        "",
        "## Sequence lengths where it failed",
        "",
        _rows_table(rows, "token_domain_or_length_failure", include_length=True),
        "",
        "## Position/intervention mismatch",
        "",
        _rows_table(rows, "position_intervention_mismatch"),
        "",
        "## Attention/effect mismatch",
        "",
        "Read this alongside the attention/effect alignment artifact. A positive or negative "
        "single-example effect is not enough without candidate-level alignment.",
        "",
        "## OV/QK mismatch",
        "",
        "Read this alongside the OV/QK diagnostic artifact. OV and QK margins are local "
        "signatures only, and they did not establish a complete circuit.",
        "",
        "## Interpretation",
        "",
        (
            "The counterexamples are part of the result. This candidate should not be upgraded "
            "unless its failures, moved controls, and position/intervention sensitivity are explained."
        ),
        "",
        "## What not to claim",
        "",
        "Do not claim induction-head discovery, a circuit, or broad GPT-2 behavior from this artifact.",
        "",
    ]
    return "\n".join(lines)


def _candidate_row(by_candidate: pd.DataFrame, candidate: str) -> pd.Series:
    labels = "L" + by_candidate["layer"].astype(int).astype(str) + "H" + by_candidate["head"].astype(int).astype(str)
    matches = by_candidate[labels == candidate]
    if matches.empty:
        raise ValueError(f"Candidate {candidate} not found in consolidated report")
    return matches.iloc[0]


def _load_seed_rows(report_root: Path, candidate_id: str) -> pd.DataFrame:
    frames = []
    for path in sorted(report_root.glob("seed*/candidate_characterization_results.csv")):
        table = pd.read_csv(path)
        frames.append(table[table["candidate_id"] == candidate_id].copy())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _attach_prompt_metadata(report_root: Path, rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    prompt_frames = []
    for path in sorted(report_root.glob("seed*/prompts.csv")):
        prompt_frames.append(pd.read_csv(path))
    if not prompt_frames:
        rows["token_domain"] = ""
        rows["sequence_length_bucket"] = ""
        return rows
    prompts = pd.concat(prompt_frames, ignore_index=True)
    keep = [
        "example_id",
        "token_domain",
        "sequence_length_bucket",
        "characterization_axis",
        "distractor_position_hint",
    ]
    base = rows.drop(
        columns=[column for column in keep if column != "example_id" and column in rows.columns],
        errors="ignore",
    )
    return base.merge(prompts[keep].drop_duplicates("example_id"), on="example_id", how="left")


def _domain_failures(positive: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if positive.empty:
        return positive
    grouped = (
        positive.groupby(["token_domain", "sequence_length_bucket"], dropna=False, as_index=False)
        .agg(mean_effect=("effect_size_numeric", "mean"))
        .sort_values("mean_effect", ascending=True)
    )
    bad_keys = set(
        tuple(row)
        for row in grouped.head(max(1, top_k // 2))[["token_domain", "sequence_length_bucket"]].itertuples(
            index=False,
            name=None,
        )
    )
    mask = [
        (row.token_domain, row.sequence_length_bucket) in bad_keys
        for row in positive.itertuples(index=False)
    ]
    return positive[mask].sort_values("effect_size_numeric", ascending=True).head(top_k)


def _position_intervention_mismatches(positive: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if positive.empty:
        return positive
    grouped = (
        positive.groupby(["intervention", "position_label"], as_index=False)
        .agg(mean_effect=("effect_size_numeric", "mean"))
        .sort_values("mean_effect", ascending=True)
    )
    weak_pairs = set(
        tuple(row)
        for row in grouped.head(max(1, top_k // 2))[["intervention", "position_label"]].itertuples(
            index=False,
            name=None,
        )
    )
    mask = [
        (row.intervention, row.position_label) in weak_pairs
        for row in positive.itertuples(index=False)
    ]
    return positive[mask].sort_values("effect_size_numeric", ascending=True).head(top_k)


def _tag(rows: pd.DataFrame, tag: str) -> pd.DataFrame:
    if rows.empty:
        return rows
    tagged = rows.copy()
    tagged["counterexample_type"] = tag
    return tagged


def _rows_table(rows: pd.DataFrame, tag: str, *, include_length: bool = False) -> str:
    subset = rows[rows["counterexample_type"] == tag].copy()
    if subset.empty:
        return "No rows."
    columns = ["seed", "family", "intervention", "position_label", "effect_size", "token_domain"]
    if include_length:
        columns.append("sequence_length_bucket")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in subset.head(8).to_dict("records"):
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _counterexample_columns() -> list[str]:
    return [
        "counterexample_type",
        "seed",
        "candidate_id",
        "candidate_group",
        "layer",
        "head",
        "family",
        "heldout_family_type",
        "example_id",
        "token_domain",
        "sequence_length_bucket",
        "intervention",
        "position_label",
        "effect_size",
        "effect_size_status",
        "true_expected_next_token",
        "wrong_or_control_token",
        "clean_prompt",
        "corrupt_prompt",
    ]
