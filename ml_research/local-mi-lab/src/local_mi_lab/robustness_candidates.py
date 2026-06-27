from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from local_mi_lab.paths import resolve_repo_path

REQUIRED_REPLICATED_HEADS = [(7, 7), (9, 11), (7, 11), (7, 0), (0, 8)]
PRIOR_RAW_ATTENTION_HEADS = [(0, 1), (0, 5), (0, 10), (11, 8), (0, 4)]

CANDIDATE_COLUMNS = [
    "candidate_id",
    "layer",
    "head",
    "candidate_group",
    "source_reason",
    "prior_mean_gap",
    "prior_replication_status",
    "prior_raw_attention_candidate",
    "prior_random_comparison_candidate",
    "include_in_main",
]


def select_heldout_candidate_set(
    multiseed_csv: str | Path,
    *,
    n_negative_no_effect: int = 3,
    n_negative_nonspecific: int = 3,
    allow_missing_required: bool = False,
) -> list[dict[str, Any]]:
    table = pd.read_csv(resolve_repo_path(multiseed_csv))
    _validate_required_heads(table, allow_missing_required=allow_missing_required)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    for layer, head in REQUIRED_REPLICATED_HEADS:
        if not _has_head(table, layer, head) and allow_missing_required:
            continue
        prior = _prior_row(table, layer, head)
        group = (
            "random_comparison_replicated"
            if bool(prior["random_comparison_candidate_in_any_seed"])
            else "replicated_candidate"
        )
        _append_candidate(
            rows,
            seen,
            prior,
            candidate_group=group,
            source_reason="Pre-registered replicated narrow candidate from prior multi-seed report.",
            include_in_main=True,
        )

    for layer, head in PRIOR_RAW_ATTENTION_HEADS:
        prior = _prior_row(table, layer, head)
        _append_candidate(
            rows,
            seen,
            prior,
            candidate_group="prior_raw_attention_failed",
            source_reason="Prior raw-attention candidate retained as a comparison head.",
            include_in_main=True,
        )

    for prior in _negative_rows(table, "no_effect", n_negative_no_effect, seen):
        _append_candidate(
            rows,
            seen,
            prior,
            candidate_group="negative_control_no_effect",
            source_reason="Deterministic no-effect negative control from prior multi-seed report.",
            include_in_main=True,
        )

    for prior in _negative_rows(table, "nonspecific", n_negative_nonspecific, seen):
        _append_candidate(
            rows,
            seen,
            prior,
            candidate_group="negative_control_nonspecific",
            source_reason="Deterministic nonspecific negative control from prior multi-seed report.",
            include_in_main=True,
        )

    if len([row for row in rows if row["candidate_group"].startswith("negative_control")]) < 5:
        raise ValueError("Could not select at least five deterministic negative controls")
    for index, row in enumerate(rows):
        row["candidate_id"] = f"heldout_cand_{index:03d}"
    return rows


def write_heldout_candidate_artifacts(
    candidates: list[dict[str, Any]],
    output_csv: str | Path,
    *,
    source_multiseed: str | Path,
) -> dict[str, Path]:
    csv_path = resolve_repo_path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = csv_path.with_suffix(".json")
    md_path = csv_path.with_suffix(".md")
    pd.DataFrame(candidates, columns=CANDIDATE_COLUMNS).to_csv(csv_path, index=False)
    payload = {
        "source_multiseed": str(resolve_repo_path(source_multiseed)),
        "n_candidates": len(candidates),
        "candidate_groups": _group_counts(candidates),
        "candidates": candidates,
        "selection_note": "Candidate set is fixed from prior results before held-out scoring.",
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_candidate_markdown(payload), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def _validate_required_heads(table: pd.DataFrame, *, allow_missing_required: bool) -> None:
    present = {(int(row.layer), int(row.head)) for row in table.itertuples()}
    missing = [f"L{layer}H{head}" for layer, head in REQUIRED_REPLICATED_HEADS if (layer, head) not in present]
    if missing and not allow_missing_required:
        raise ValueError(f"Required replicated heads missing from prior report: {missing}")


def _prior_row(table: pd.DataFrame, layer: int, head: int) -> pd.Series:
    match = table[(table["layer"] == layer) & (table["head"] == head)]
    if match.empty:
        raise ValueError(f"Missing required head L{layer}H{head}")
    return match.iloc[0]


def _has_head(table: pd.DataFrame, layer: int, head: int) -> bool:
    return not table[(table["layer"] == layer) & (table["head"] == head)].empty


def _append_candidate(
    rows: list[dict[str, Any]],
    seen: set[tuple[int, int]],
    prior: pd.Series,
    *,
    candidate_group: str,
    source_reason: str,
    include_in_main: bool,
) -> None:
    layer = int(prior["layer"])
    head = int(prior["head"])
    key = (layer, head)
    if key in seen:
        return
    seen.add(key)
    rows.append(
        {
            "candidate_id": "",
            "layer": layer,
            "head": head,
            "candidate_group": candidate_group,
            "source_reason": source_reason,
            "prior_mean_gap": float(prior["mean_positive_minus_control_gap"]),
            "prior_replication_status": str(prior["replication_status"]),
            "prior_raw_attention_candidate": bool(prior["raw_attention_candidate_in_any_seed"]),
            "prior_random_comparison_candidate": bool(
                prior["random_comparison_candidate_in_any_seed"]
            ),
            "include_in_main": include_in_main,
        }
    )


def _negative_rows(
    table: pd.DataFrame,
    status: str,
    n_rows: int,
    seen: set[tuple[int, int]],
) -> list[pd.Series]:
    pool = table[table["replication_status"] == status].copy()
    pool["abs_gap"] = pool["mean_positive_minus_control_gap"].abs()
    pool = pool.sort_values(["abs_gap", "layer", "head"], ascending=[True, True, True])
    selected = []
    for row in pool.itertuples(index=False):
        key = (int(row.layer), int(row.head))
        if key in seen:
            continue
        selected.append(pd.Series(row._asdict()))
        if len(selected) >= n_rows:
            return selected
    if len(selected) < n_rows:
        raise ValueError(f"Could not select {n_rows} negative controls with status {status!r}")
    return selected


def _group_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        group = str(candidate["candidate_group"])
        counts[group] = counts.get(group, 0) + 1
    return dict(sorted(counts.items()))


def _candidate_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Held-Out Robustness Candidate Set",
        "",
        "This candidate set was selected from prior multi-seed artifacts before held-out scoring.",
        "",
        f"- Source: `{payload['source_multiseed']}`",
        f"- Candidates: `{payload['n_candidates']}`",
        f"- Groups: `{payload['candidate_groups']}`",
        "",
        "| candidate_id | head | group | prior status | prior gap | flags |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in payload["candidates"]:
        flags = []
        if candidate["prior_raw_attention_candidate"]:
            flags.append("raw-attention")
        if candidate["prior_random_comparison_candidate"]:
            flags.append("random-comparison")
        lines.append(
            "| {candidate_id} | L{layer}H{head} | {candidate_group} | "
            "{prior_replication_status} | {prior_mean_gap:.4f} | {flags} |".format(
                **candidate,
                flags=", ".join(flags) if flags else "",
            )
        )
    lines.extend(
        [
            "",
            "L7H7 remains flagged if it was a prior random-comparison candidate. The held-out run must not treat that label as evidence.",
            "",
        ]
    )
    return "\n".join(lines)
