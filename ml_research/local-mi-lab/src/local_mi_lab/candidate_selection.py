from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from local_mi_lab.paths import resolve_repo_path

CANDIDATE_COLUMNS = [
    "candidate_id",
    "source",
    "layer",
    "head",
    "component",
    "position_label",
    "raw_positive_attention",
    "max_control_attention",
    "positive_minus_control_attention_gap",
    "max_control_family",
    "gap_positive",
    "selection_reason",
    "expected_specificity",
]

ATTN_OUT = "attn_out"
RESID_POST = "resid_post"


def select_controlled_patching_candidates(
    run_dir: str | Path,
    top_k_raw: int = 5,
    top_k_control: int = 5,
    top_k_gap: int = 5,
    n_random: int = 5,
    seed: int = 0,
) -> list[dict[str, Any]]:
    root = resolve_repo_path(run_dir)
    summary = json.loads((root / "attention_summary.json").read_text(encoding="utf-8"))
    attention_by_family = pd.read_csv(root / "attention_by_family.csv")
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int | str, str]] = set()

    for row in summary.get("top_heads_on_positive_examples", [])[:top_k_raw]:
        _add_candidate(
            candidates,
            seen,
            row=row,
            source="top_raw_positive_attention",
            component=ATTN_OUT,
            attention_by_family=attention_by_family,
            selection_reason="High raw previous-occurrence attention on positive examples.",
        )

    for row in summary.get("top_heads_on_controls", [])[:top_k_control]:
        _add_candidate(
            candidates,
            seen,
            row=row,
            source="top_control_firing_attention",
            component=ATTN_OUT,
            attention_by_family=attention_by_family,
            selection_reason="High previous-occurrence attention on controls.",
        )

    positive_gap_rows = [
        row
        for row in summary.get("top_heads_by_positive_minus_control_gap", [])
        if float(row.get("positive_minus_control_attention_gap") or 0.0) > 0.0
    ]
    for row in positive_gap_rows[:top_k_gap]:
        _add_candidate(
            candidates,
            seen,
            row=row,
            source="top_positive_minus_control_gap",
            component=ATTN_OUT,
            attention_by_family=attention_by_family,
            selection_reason="Positive-minus-control attention gap was positive.",
        )

    _add_random_candidates(
        candidates,
        seen,
        attention_by_family=attention_by_family,
        n_random=n_random,
        seed=seed,
    )
    _assign_candidate_ids(candidates)
    return candidates


def write_candidate_artifacts(
    run_dir: str | Path,
    candidates: list[dict[str, Any]],
    *,
    top_k_raw: int,
    top_k_control: int,
    top_k_gap: int,
    n_random: int,
    seed: int,
) -> dict[str, Path]:
    root = resolve_repo_path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "controlled_patching_candidates.csv"
    json_path = root / "controlled_patching_candidates.json"
    md_path = root / "controlled_patching_selection.md"
    pd.DataFrame(candidates, columns=CANDIDATE_COLUMNS).to_csv(csv_path, index=False)
    payload = {
        "run_dir": str(root),
        "selection": {
            "top_k_raw": top_k_raw,
            "top_k_control": top_k_control,
            "top_k_gap": top_k_gap,
            "n_random": n_random,
            "seed": seed,
        },
        "gap_positive_candidates": [
            candidate for candidate in candidates if candidate["gap_positive"]
        ],
        "candidates": candidates,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_selection_markdown(root, candidates, payload["selection"]), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def _add_candidate(
    candidates: list[dict[str, Any]],
    seen: set[tuple[int, int | str, str]],
    *,
    row: dict[str, Any],
    source: str,
    component: str,
    attention_by_family: pd.DataFrame,
    selection_reason: str,
) -> None:
    layer = int(row["layer"])
    head = int(row["head"])
    key = (layer, head, component)
    if key in seen:
        return
    seen.add(key)
    stats = _attention_stats(attention_by_family, layer, head)
    gap = _row_gap(row, stats)
    candidates.append(
        {
            "candidate_id": "",
            "source": source,
            "layer": layer,
            "head": head,
            "component": component,
            "position_label": "final",
            "raw_positive_attention": stats["raw_positive_attention"],
            "max_control_attention": stats["max_control_attention"],
            "positive_minus_control_attention_gap": gap,
            "max_control_family": stats["max_control_family"],
            "gap_positive": gap > 0.0,
            "selection_reason": selection_reason,
            "expected_specificity": (
                "attention_gap_candidate" if gap > 0.0 else "nonspecific_attention_candidate"
            ),
        }
    )


def _add_random_candidates(
    candidates: list[dict[str, Any]],
    seen: set[tuple[int, int | str, str]],
    *,
    attention_by_family: pd.DataFrame,
    n_random: int,
    seed: int,
) -> None:
    all_heads = sorted(
        {
            (int(row.layer), int(row.head))
            for row in attention_by_family[["layer", "head"]].drop_duplicates().itertuples()
        }
    )
    rng = random.Random(seed)
    rng.shuffle(all_heads)
    for layer, head in all_heads:
        if len([c for c in candidates if c["source"] == "random_comparison"]) >= n_random:
            return
        key = (layer, head, ATTN_OUT)
        if key in seen:
            continue
        seen.add(key)
        stats = _attention_stats(attention_by_family, layer, head)
        gap = stats["raw_positive_attention"] - stats["max_control_attention"]
        candidates.append(
            {
                "candidate_id": "",
                "source": "random_comparison",
                "layer": layer,
                "head": head,
                "component": ATTN_OUT,
                "position_label": "final",
                "raw_positive_attention": stats["raw_positive_attention"],
                "max_control_attention": stats["max_control_attention"],
                "positive_minus_control_attention_gap": gap,
                "max_control_family": stats["max_control_family"],
                "gap_positive": gap > 0.0,
                "selection_reason": "Deterministic random comparison head.",
                "expected_specificity": "comparison_candidate",
            }
        )


def _attention_stats(attention_by_family: pd.DataFrame, layer: int, head: int) -> dict[str, Any]:
    rows = attention_by_family[
        (attention_by_family["layer"] == layer) & (attention_by_family["head"] == head)
    ].copy()
    positive = rows[rows["family"] == "positive_repeat_sequence"]
    controls = rows[rows["family"] != "positive_repeat_sequence"].dropna(
        subset=["mean_attention_to_previous_occurrence"]
    )
    raw_positive = (
        float(positive["mean_attention_to_previous_occurrence"].iloc[0])
        if not positive.empty and pd.notna(positive["mean_attention_to_previous_occurrence"].iloc[0])
        else 0.0
    )
    if controls.empty:
        return {
            "raw_positive_attention": raw_positive,
            "max_control_attention": 0.0,
            "max_control_family": "",
        }
    max_row = controls.sort_values(
        "mean_attention_to_previous_occurrence",
        ascending=False,
    ).iloc[0]
    return {
        "raw_positive_attention": raw_positive,
        "max_control_attention": float(max_row["mean_attention_to_previous_occurrence"]),
        "max_control_family": str(max_row["family"]),
    }


def _row_gap(row: dict[str, Any], stats: dict[str, Any]) -> float:
    if "positive_minus_control_attention_gap" in row:
        return float(row.get("positive_minus_control_attention_gap") or 0.0)
    return float(stats["raw_positive_attention"] - stats["max_control_attention"])


def _assign_candidate_ids(candidates: list[dict[str, Any]]) -> None:
    for i, candidate in enumerate(candidates):
        candidate["candidate_id"] = f"cand_{i:03d}"


def _selection_markdown(
    run_dir: Path,
    candidates: list[dict[str, Any]],
    selection: dict[str, int],
) -> str:
    any_positive_gap = any(candidate["gap_positive"] for candidate in candidates)
    nonspecific = [
        candidate
        for candidate in candidates
        if candidate["expected_specificity"] == "nonspecific_attention_candidate"
    ]
    lines = [
        "# Controlled Patching Candidate Selection",
        "",
        "## Artifacts Read",
        "",
        f"- `{run_dir / 'attention_summary.json'}`",
        f"- `{run_dir / 'attention_by_family.csv'}`",
        "",
        "## Selection",
        "",
        f"- Top raw positive heads: {selection['top_k_raw']}",
        f"- Top control-firing heads: {selection['top_k_control']}",
        f"- Top positive-minus-control gap heads: {selection['top_k_gap']}",
        f"- Random comparison heads: {selection['n_random']}",
        f"- Seed: {selection['seed']}",
        "",
        "## Positive Gap Status",
        "",
        f"Any positive-minus-control attention gaps selected: {any_positive_gap}.",
        "",
        "## Nonspecific Candidates",
        "",
    ]
    if nonspecific:
        for candidate in nonspecific:
            lines.append(
                f"- `{candidate['candidate_id']}` L{candidate['layer']}H{candidate['head']} "
                f"gap={candidate['positive_minus_control_attention_gap']:.6f}; "
                "expected to be nonspecific until causal controls say otherwise."
            )
    else:
        lines.append("No nonspecific raw/control attention candidates were selected.")
    lines.extend(
        [
            "",
            "## Comparison Candidates",
            "",
        ]
    )
    for candidate in candidates:
        if candidate["source"] == "random_comparison":
            lines.append(
                f"- `{candidate['candidate_id']}` L{candidate['layer']}H{candidate['head']}"
            )
    lines.extend(
        [
            "",
            "Raw attention candidates are not causal candidates. Candidate selection only defines what to test next.",
        ]
    )
    return "\n".join(lines) + "\n"
