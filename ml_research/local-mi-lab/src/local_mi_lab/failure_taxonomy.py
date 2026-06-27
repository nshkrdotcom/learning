from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from local_mi_lab.paths import repo_root, resolve_repo_path

PRIMARY_HEADS = {"L7H7", "L9H11", "L7H11", "L7H0", "L0H8"}

FAILURE_MODES = [
    "control_moved",
    "target_swap_leak",
    "reversed_control_leak",
    "domain_flip",
    "length_flip",
    "intervention_disagreement",
    "position_mismatch",
    "attention_effect_decoupled",
    "ov_qk_local_only",
    "negative_control_support",
]

ROW_COLUMNS = [
    "head_label",
    "candidate_id",
    "candidate_group",
    "source_file",
    "counterexample_type",
    "seed",
    "family",
    "heldout_family_type",
    "token_domain",
    "sequence_length_bucket",
    "intervention",
    "position_label",
    "effect_size",
    "failure_mode",
    "reason",
]

HEAD_COLUMNS = [
    "head_label",
    "candidate_id",
    "candidate_group",
    "final_status",
    "mean_gap",
    "mean_attention_effect_corr",
    *FAILURE_MODES,
    "dominant_failure_modes",
    "n_failure_modes",
    "interpretation",
]


@dataclass(frozen=True)
class SummaryHeadRow:
    head_label: str
    candidate_id: str
    candidate_group: str
    final_status: str
    mean_gap: float | None
    mean_attention_effect_corr: float | None
    ov_statuses: str
    qk_statuses: str


def build_failure_taxonomy(
    *,
    counterexamples_dir: str | Path,
    summary_path: str | Path,
    output_dir: str | Path,
    tracked_summary_path: str | Path = "docs/results/head_specific_candidate_characterization_failure_taxonomy_v1.md",
) -> dict[str, Path]:
    counterexamples_root = resolve_repo_path(counterexamples_dir)
    summary_file = resolve_repo_path(summary_path)
    output_root = resolve_repo_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    counterexamples = load_counterexamples(counterexamples_root)
    summary_text = summary_file.read_text(encoding="utf-8")
    summary_rows = parse_summary_head_rows(summary_text)
    row_taxonomy = classify_counterexample_rows(counterexamples)
    by_head = summarize_failure_taxonomy_by_head(
        row_taxonomy,
        summary_rows=summary_rows,
        negative_control_support=summary_has_negative_control_support(summary_text),
    )
    summary = render_failure_summary(counterexamples_root, summary_file, row_taxonomy, by_head)

    row_path = output_root / "failure_taxonomy_by_row.csv"
    head_path = output_root / "failure_taxonomy_by_head.csv"
    summary_json_path = output_root / "failure_taxonomy_summary.json"
    markdown_path = output_root / "failure_taxonomy.md"
    tracked_path = resolve_repo_path(tracked_summary_path)
    tracked_path.parent.mkdir(parents=True, exist_ok=True)

    row_taxonomy.to_csv(row_path, index=False)
    by_head.to_csv(head_path, index=False)
    summary_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown = render_failure_taxonomy_markdown(summary, by_head, row_taxonomy)
    markdown_path.write_text(markdown, encoding="utf-8")
    tracked_path.write_text(markdown, encoding="utf-8")

    return {
        "by_row": row_path,
        "by_head": head_path,
        "summary": summary_json_path,
        "markdown": markdown_path,
        "tracked_summary": tracked_path,
    }


def load_counterexamples(counterexamples_dir: str | Path) -> pd.DataFrame:
    root = resolve_repo_path(counterexamples_dir)
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("counterexamples_*.csv")):
        table = pd.read_csv(path)
        table["source_file"] = path.as_posix()
        table["head_label"] = "L" + table["layer"].astype(int).astype(str) + "H" + table["head"].astype(int).astype(str)
        frames.append(table)
    if not frames:
        raise FileNotFoundError(f"No counterexamples_*.csv files found in {root}")
    return pd.concat(frames, ignore_index=True)


def classify_counterexample_rows(counterexamples: pd.DataFrame) -> pd.DataFrame:
    if counterexamples.empty:
        return pd.DataFrame(columns=ROW_COLUMNS)
    rows = counterexamples.copy()
    rows["effect_size_numeric"] = pd.to_numeric(rows.get("effect_size"), errors="coerce")
    contexts = _sign_contexts(rows)
    classified: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        for mode, reason in classify_single_counterexample(row, contexts).items():
            classified.append(_row_output(row, mode, reason))
    if not classified:
        return pd.DataFrame(columns=ROW_COLUMNS)
    return pd.DataFrame(classified, columns=ROW_COLUMNS).drop_duplicates()


def classify_single_counterexample(
    row: dict[str, Any],
    contexts: dict[str, set[str]] | None = None,
) -> dict[str, str]:
    contexts = contexts or {}
    modes: dict[str, str] = {}
    counterexample_type = str(row.get("counterexample_type", ""))
    family = str(row.get("family", ""))
    head_label = _head_label_from_row(row)
    heldout_family_type = str(row.get("heldout_family_type", ""))
    effect = _float_or_none(row.get("effect_size"))

    if counterexample_type == "control_moved" or (
        heldout_family_type == "control" and effect is not None and effect > 0
    ):
        modes["control_moved"] = "A control row had positive intervention effect."
    if counterexample_type == "wrong_target_control_moved" or "target_swap" in family or "wrong_target" in family:
        modes["target_swap_leak"] = "A wrong-target or target-swap control moved."
    if "reversed_control" in family and effect is not None and effect > 0:
        modes["reversed_control_leak"] = "A reversed-order control moved."
    if counterexample_type == "token_domain_or_length_failure":
        if head_label in contexts.get("domain_flip_heads", set()):
            modes["domain_flip"] = "Effect direction disagreed across token domains."
        if head_label in contexts.get("length_flip_heads", set()):
            modes["length_flip"] = "Effect direction disagreed across sequence lengths."
        if "domain_flip" not in modes and "length_flip" not in modes:
            modes["domain_flip"] = "Token-domain or length failure row was present."
            modes["length_flip"] = "Token-domain or length failure row was present."
    if counterexample_type == "position_intervention_mismatch":
        if head_label in contexts.get("intervention_flip_heads", set()):
            modes["intervention_disagreement"] = "Effect direction disagreed across interventions."
        if head_label in contexts.get("position_flip_heads", set()):
            modes["position_mismatch"] = "Effect direction disagreed across positions."
        if "intervention_disagreement" not in modes and "position_mismatch" not in modes:
            modes["intervention_disagreement"] = "Position/intervention mismatch row was present."
            modes["position_mismatch"] = "Position/intervention mismatch row was present."
    return modes


def parse_summary_head_rows(summary_text: str) -> dict[str, SummaryHeadRow]:
    rows: dict[str, SummaryHeadRow] = {}
    for line in summary_text.splitlines():
        if not line.startswith("| L"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        head_label, candidate_id, group, status, gap, corr, ov, qk = cells[:8]
        if not candidate_id.startswith("heldout_cand_"):
            continue
        rows.setdefault(
            head_label,
            SummaryHeadRow(
                head_label=head_label,
                candidate_id=candidate_id,
                candidate_group=group,
                final_status=status,
                mean_gap=_float_or_none(gap),
                mean_attention_effect_corr=_float_or_none(corr),
                ov_statuses=ov,
                qk_statuses=qk,
            ),
        )
    return rows


def summary_has_negative_control_support(summary_text: str) -> bool:
    for line in summary_text.splitlines():
        if "negative_control" in line and "characterization_supports" in line:
            return True
    return False


def summarize_failure_taxonomy_by_head(
    row_taxonomy: pd.DataFrame,
    *,
    summary_rows: dict[str, SummaryHeadRow],
    negative_control_support: bool,
) -> pd.DataFrame:
    head_labels = sorted(set(row_taxonomy["head_label"]) | set(summary_rows))
    rows: list[dict[str, Any]] = []
    for head_label in head_labels:
        subset = row_taxonomy[row_taxonomy["head_label"] == head_label]
        mode_counts = Counter(subset["failure_mode"]) if not subset.empty else Counter()
        summary = summary_rows.get(head_label)
        if summary is not None:
            if _is_attention_effect_decoupled(summary):
                mode_counts["attention_effect_decoupled"] += 1
            if _has_local_ov_qk_without_specificity(summary):
                mode_counts["ov_qk_local_only"] += 1
        if negative_control_support and head_label in PRIMARY_HEADS:
            mode_counts["negative_control_support"] += 1

        row = {
            "head_label": head_label,
            "candidate_id": _first_non_empty(subset.get("candidate_id")) if summary is None else summary.candidate_id,
            "candidate_group": _first_non_empty(subset.get("candidate_group")) if summary is None else summary.candidate_group,
            "final_status": "" if summary is None else summary.final_status,
            "mean_gap": None if summary is None else summary.mean_gap,
            "mean_attention_effect_corr": None
            if summary is None
            else summary.mean_attention_effect_corr,
        }
        row.update({mode: int(mode_counts.get(mode, 0)) for mode in FAILURE_MODES})
        active_modes = [mode for mode in FAILURE_MODES if mode_counts.get(mode, 0) > 0]
        row["dominant_failure_modes"] = ",".join(active_modes[:5])
        row["n_failure_modes"] = len(active_modes)
        row["interpretation"] = interpret_head_failure(head_label, active_modes, summary)
        rows.append(row)
    return pd.DataFrame(rows, columns=HEAD_COLUMNS)


def render_failure_summary(
    counterexamples_dir: Path,
    summary_path: Path,
    row_taxonomy: pd.DataFrame,
    by_head: pd.DataFrame,
) -> dict[str, Any]:
    mode_counts = (
        row_taxonomy["failure_mode"].value_counts().sort_index().to_dict()
        if not row_taxonomy.empty
        else {}
    )
    head_modes = {
        row["head_label"]: row["dominant_failure_modes"]
        for row in by_head.to_dict("records")
        if row.get("dominant_failure_modes")
    }
    return {
        "source_artifacts": {
            "counterexamples_dir": _display_path(counterexamples_dir),
            "summary_path": _display_path(summary_path),
        },
        "command": (
            "uv run python scripts/build_characterization_failure_taxonomy.py "
            f"--counterexamples {_display_path(counterexamples_dir)} "
            f"--summary {_display_path(summary_path)} "
            "--output reports/head_specific_candidate_characterization_v1/failure_taxonomy"
        ),
        "n_row_failure_labels": int(len(row_taxonomy)),
        "n_heads": int(len(by_head)),
        "failure_mode_counts": mode_counts,
        "head_modes": head_modes,
        "final_status": "failure_taxonomy_complete_no_candidate_search_justified",
        "refused_claims": [
            "No induction-head discovery.",
            "No circuit claim.",
            "No broad GPT-2 claim.",
            "No new candidate selection from this taxonomy.",
        ],
    }


def render_failure_taxonomy_markdown(
    summary: dict[str, Any],
    by_head: pd.DataFrame,
    row_taxonomy: pd.DataFrame,
) -> str:
    lines = [
        "# Head-Specific Candidate Characterization Failure Taxonomy v1",
        "",
        "## Source Artifacts",
        "",
        f"- Counterexamples: `{summary['source_artifacts']['counterexamples_dir']}`",
        f"- Characterization summary: `{summary['source_artifacts']['summary_path']}`",
        "",
        "## Command",
        "",
        "```bash",
        summary["command"],
        "```",
        "",
        "## Result",
        "",
        "All primary heads remain falsified. This taxonomy diagnoses why they failed; it does not select new heads.",
        "",
        "## Failure Mode Counts",
        "",
        _dict_table(summary["failure_mode_counts"], "failure_mode", "n_rows"),
        "",
        "## Primary Heads",
        "",
        _head_table(by_head[by_head["head_label"].isin(PRIMARY_HEADS)]),
        "",
        "## By-Head Interpretation",
        "",
    ]
    for row in by_head[by_head["head_label"].isin(PRIMARY_HEADS)].sort_values("head_label").to_dict("records"):
        lines.extend(
            [
                f"### {row['head_label']}",
                "",
                f"- Final status: `{row.get('final_status', '')}`",
                f"- Dominant failure modes: `{row.get('dominant_failure_modes', '')}`",
                f"- Interpretation: {row.get('interpretation', '')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Concrete Counterexample Rows",
            "",
            _counterexample_sample_table(row_taxonomy),
            "",
            "## What This Means",
            "",
            "The failure pattern points to calibration work before any future candidate search. "
            "Controls moved, target-swap and reversed-order controls leaked, domain and length variants disagreed, "
            "and intervention or position slices were not stable enough to support a mechanism claim.",
            "",
            "## What This Does Not Show",
            "",
            "This does not show an induction head, a circuit, or a broad GPT-2 property. "
            "It also does not justify adding new heads. The next step is metric and prompt calibration.",
            "",
            "## Exact Next Command",
            "",
            "```bash",
            "less docs/results/head_specific_candidate_characterization_failure_taxonomy_v1.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def interpret_head_failure(
    head_label: str,
    active_modes: list[str],
    summary: SummaryHeadRow | None,
) -> str:
    if not active_modes:
        return "No taxonomy rows were available for this head."
    parts: list[str] = []
    if "control_moved" in active_modes:
        parts.append("controls moved under the same intervention logic")
    if "target_swap_leak" in active_modes:
        parts.append("wrong-target or target-swap controls leaked")
    if "reversed_control_leak" in active_modes:
        parts.append("reversed-order controls moved")
    if "domain_flip" in active_modes or "length_flip" in active_modes:
        parts.append("token domain or sequence length changed the effect direction")
    if "intervention_disagreement" in active_modes or "position_mismatch" in active_modes:
        parts.append("intervention or position variants disagreed")
    if "attention_effect_decoupled" in active_modes:
        parts.append("attention/effect alignment was weak")
    if "ov_qk_local_only" in active_modes:
        parts.append("OV/QK local diagnostics did not combine with causal specificity")
    if "negative_control_support" in active_modes:
        parts.append("negative-control support made the slice-level rule too permissive")
    final_status = "" if summary is None else f" Its consolidated status was `{summary.final_status}`."
    return f"{head_label} failed because " + "; ".join(parts) + "." + final_status


def _sign_contexts(rows: pd.DataFrame) -> dict[str, set[str]]:
    positive = rows[rows["heldout_family_type"] == "positive"].copy()
    positive = positive.dropna(subset=["effect_size_numeric"])
    return {
        "domain_flip_heads": _heads_with_sign_flip(positive, "token_domain"),
        "length_flip_heads": _heads_with_sign_flip(positive, "sequence_length_bucket"),
        "intervention_flip_heads": _heads_with_sign_flip(positive, "intervention"),
        "position_flip_heads": _heads_with_sign_flip(positive, "position_label"),
    }


def _heads_with_sign_flip(rows: pd.DataFrame, column: str) -> set[str]:
    heads: set[str] = set()
    if rows.empty or column not in rows:
        return heads
    grouped = rows.groupby(["head_label", column], dropna=False).agg(mean_effect=("effect_size_numeric", "mean"))
    for head_label, group in grouped.reset_index().groupby("head_label"):
        values = group["mean_effect"].dropna()
        if (values > 0).any() and (values < 0).any():
            heads.add(str(head_label))
    return heads


def _row_output(row: dict[str, Any], mode: str, reason: str) -> dict[str, Any]:
    return {
        "head_label": _head_label_from_row(row),
        "candidate_id": row.get("candidate_id", ""),
        "candidate_group": row.get("candidate_group", ""),
        "source_file": row.get("source_file", ""),
        "counterexample_type": row.get("counterexample_type", ""),
        "seed": row.get("seed", ""),
        "family": row.get("family", ""),
        "heldout_family_type": row.get("heldout_family_type", ""),
        "token_domain": row.get("token_domain", ""),
        "sequence_length_bucket": row.get("sequence_length_bucket", ""),
        "intervention": row.get("intervention", ""),
        "position_label": row.get("position_label", ""),
        "effect_size": row.get("effect_size", ""),
        "failure_mode": mode,
        "reason": reason,
    }


def _head_label_from_row(row: dict[str, Any]) -> str:
    if row.get("head_label"):
        return str(row["head_label"])
    return f"L{int(row['layer'])}H{int(row['head'])}"


def _is_attention_effect_decoupled(summary: SummaryHeadRow) -> bool:
    corr = summary.mean_attention_effect_corr
    return corr is None or corr <= 0.05


def _has_local_ov_qk_without_specificity(summary: SummaryHeadRow) -> bool:
    if summary.final_status != "falsified_candidate":
        return False
    return "supports" in summary.ov_statuses or "supports" in summary.qk_statuses


def _first_non_empty(values: pd.Series | None) -> str:
    if values is None:
        return ""
    for value in values.dropna().astype(str):
        if value:
            return value
    return ""


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _dict_table(values: dict[str, Any], key_name: str, value_name: str) -> str:
    if not values:
        return "No rows."
    lines = [f"| {key_name} | {value_name} |", "| --- | --- |"]
    for key, value in sorted(values.items()):
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    root = repo_root()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _head_table(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "No primary heads."
    columns = ["head_label", "final_status", "mean_gap", "dominant_failure_modes"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows.sort_values("head_label").to_dict("records"):
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _counterexample_sample_table(row_taxonomy: pd.DataFrame) -> str:
    if row_taxonomy.empty:
        return "No row-level labels."
    columns = ["head_label", "failure_mode", "family", "intervention", "position_label", "effect_size"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    sample = row_taxonomy.sort_values(["head_label", "failure_mode"]).head(20)
    for row in sample.to_dict("records"):
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)
