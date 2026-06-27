from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from local_mi_lab.paths import resolve_repo_path

PRIMARY_HEADS = {"L7H7", "L9H11", "L7H11", "L7H0", "L0H8"}

MULTISEED_CANDIDATE_COLUMNS = [
    "candidate_id",
    "candidate_group",
    "layer",
    "head",
    "n_seeds",
    "seeds_present",
    "n_support_seeds",
    "n_downgrade_seeds",
    "n_falsify_seeds",
    "mean_positive_minus_control_gap",
    "mean_attention_effect_corr",
    "mean_source_attention_margin",
    "mean_ov_copy_margin",
    "mean_qk_source_margin",
    "position_statuses",
    "ov_statuses",
    "qk_statuses",
    "final_characterization_status",
]

STATUS_COLORS = {
    "strengthened_local_candidate": "#0b6e4f",
    "downgraded_candidate": "#b7791f",
    "falsified_candidate": "#b91c1c",
    "insufficient_characterization_data": "#6b7280",
}


def load_characterization_run(run_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = resolve_repo_path(run_dir)
    by_candidate_path = root / "candidate_characterization_by_candidate.csv"
    summary_path = root / "candidate_characterization_summary.json"
    if not by_candidate_path.exists():
        raise FileNotFoundError(f"Missing {by_candidate_path}")
    by_candidate = pd.read_csv(by_candidate_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    seed = int(by_candidate["seed"].iloc[0]) if not by_candidate.empty else summary.get("seed")
    manifest = {
        "run_dir": str(root),
        "seed": seed,
        "n_candidate_rows": int(len(by_candidate)),
        "summary": summary,
    }
    return by_candidate, manifest


def compare_candidate_characterization_runs(
    run_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    tracked_summary_path: str | Path | None = "docs/results/head_specific_candidate_characterization_v1.md",
) -> dict[str, Path]:
    root = resolve_repo_path(output_dir)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        by_candidate, manifest = load_characterization_run(run_dir)
        frames.append(by_candidate)
        manifests.append(manifest)
    candidates = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    by_candidate = summarize_multiseed_candidates(candidates)
    by_axis = summarize_multiseed_axes(candidates)
    summary = characterization_multiseed_summary(by_candidate, by_axis, manifests)

    manifest_path = root / "run_manifest.json"
    by_candidate_path = root / "candidate_characterization_multiseed_by_candidate.csv"
    by_axis_path = root / "candidate_characterization_multiseed_by_axis.csv"
    summary_path = root / "candidate_characterization_summary.json"
    markdown_path = root / "head_specific_candidate_characterization_v1.md"
    manifest_path.write_text(json.dumps({"runs": manifests}, indent=2) + "\n", encoding="utf-8")
    by_candidate.to_csv(by_candidate_path, index=False)
    by_axis.to_csv(by_axis_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown = render_characterization_markdown(summary, by_candidate, by_axis)
    markdown_path.write_text(markdown, encoding="utf-8")

    generated = []
    generated.extend(_save_figure_pair(plot_status_counts(by_candidate), figures / "characterization_candidate_statuses"))
    generated.extend(
        _save_figure_pair(
            plot_attention_effect_summary(by_candidate),
            figures / "attention_effect_alignment_summary",
        )
    )
    generated.extend(
        _save_figure_pair(plot_ov_qk_summary(by_candidate), figures / "ov_qk_diagnostic_summary")
    )
    generated.extend(
        _save_figure_pair(
            plot_position_specificity_summary(candidates),
            figures / "position_specificity_summary",
        )
    )

    paths = {
        "manifest": manifest_path,
        "by_candidate": by_candidate_path,
        "by_axis": by_axis_path,
        "summary": summary_path,
        "markdown": markdown_path,
    }
    for index, path in enumerate(generated):
        paths[f"figure_{index}"] = path
    if tracked_summary_path is not None:
        tracked = resolve_repo_path(tracked_summary_path)
        tracked.parent.mkdir(parents=True, exist_ok=True)
        tracked.write_text(markdown, encoding="utf-8")
        paths["tracked_summary"] = tracked
    return paths


def summarize_multiseed_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=MULTISEED_CANDIDATE_COLUMNS)
    rows = candidates.copy()
    for column in [
        "positive_minus_control_gap",
        "spearman_attention_effect_corr",
        "mean_source_attention_margin",
        "ov_copy_margin",
        "qk_source_margin",
    ]:
        rows[column] = pd.to_numeric(rows.get(column), errors="coerce")
    negative_reference = _negative_control_reference(rows)
    summaries: list[dict[str, Any]] = []
    for (candidate_id, layer, head), group in rows.groupby(["candidate_id", "layer", "head"], sort=True):
        seeds = sorted(int(seed) for seed in group["seed"].dropna().unique())
        support = int((group["characterization_seed_status"] == "characterization_supports").sum())
        downgrade = int((group["characterization_seed_status"] == "characterization_downgrades").sum())
        falsify = int((group["characterization_seed_status"] == "characterization_falsifies").sum())
        mean_gap = _mean_or_none(group["positive_minus_control_gap"])
        negative_dominates = (
            not str(group["candidate_group"].iloc[0]).startswith("negative_control")
            and support >= 2
            and negative_reference["support_seeds"] >= support
            and negative_reference["mean_gap"] is not None
            and mean_gap is not None
            and negative_reference["mean_gap"] >= mean_gap
        )
        status = classify_final_characterization_status(
            group,
            negative_control_dominates=negative_dominates,
        )
        summaries.append(
            {
                "candidate_id": str(candidate_id),
                "candidate_group": str(group["candidate_group"].iloc[0]),
                "layer": int(layer),
                "head": int(head),
                "n_seeds": len(seeds),
                "seeds_present": ",".join(str(seed) for seed in seeds),
                "n_support_seeds": support,
                "n_downgrade_seeds": downgrade,
                "n_falsify_seeds": falsify,
                "mean_positive_minus_control_gap": mean_gap,
                "mean_attention_effect_corr": _mean_or_none(group["spearman_attention_effect_corr"]),
                "mean_source_attention_margin": _mean_or_none(group["mean_source_attention_margin"]),
                "mean_ov_copy_margin": _mean_or_none(group["ov_copy_margin"]),
                "mean_qk_source_margin": _mean_or_none(group["qk_source_margin"]),
                "position_statuses": _join_unique(group["position_specificity_status"]),
                "ov_statuses": _join_unique(group["ov_status"]),
                "qk_statuses": _join_unique(group["qk_status"]),
                "final_characterization_status": status,
            }
        )
    return pd.DataFrame(summaries, columns=MULTISEED_CANDIDATE_COLUMNS).sort_values(
        ["final_characterization_status", "mean_positive_minus_control_gap"],
        ascending=[True, False],
        na_position="last",
    )


def classify_final_characterization_status(
    group: pd.DataFrame,
    *,
    negative_control_dominates: bool = False,
) -> str:
    seeds = set(int(seed) for seed in group["seed"].dropna().unique())
    if len(seeds) < 2:
        return "insufficient_characterization_data"
    statuses = group["characterization_seed_status"].dropna()
    support = int((statuses == "characterization_supports").sum())
    downgrade = int((statuses == "characterization_downgrades").sum())
    falsify = int((statuses == "characterization_falsifies").sum())
    mean_corr = _mean_or_none(pd.to_numeric(group["spearman_attention_effect_corr"], errors="coerce"))
    ov_support = int((group["ov_status"] == "ov_supports_copy").sum())
    qk_support = int((group["qk_status"] == "qk_supports_source_selection").sum())
    if negative_control_dominates:
        return "falsified_candidate"
    if falsify >= 2:
        return "falsified_candidate"
    if support >= 2 and falsify == 0 and mean_corr is not None and mean_corr > 0:
        if ov_support >= 2 or qk_support >= 2:
            return "strengthened_local_candidate"
        return "downgraded_candidate"
    if support >= 1 or downgrade >= 1:
        return "downgraded_candidate"
    return "falsified_candidate"


def summarize_multiseed_axes(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "candidate_group",
                "layer",
                "head",
                "axis",
                "mean_value",
                "statuses",
            ]
        )
    axis_specs = [
        ("effect_gap", "positive_minus_control_gap", "characterization_seed_status"),
        ("attention_effect_alignment", "spearman_attention_effect_corr", "characterization_seed_status"),
        ("source_attention_margin", "mean_source_attention_margin", "characterization_seed_status"),
        ("ov_copy", "ov_copy_margin", "ov_status"),
        ("qk_source_selection", "qk_source_margin", "qk_status"),
    ]
    rows: list[dict[str, Any]] = []
    for key, group in candidates.groupby(["candidate_id", "candidate_group", "layer", "head"], sort=True):
        candidate_id, candidate_group, layer, head = key
        for axis, value_column, status_column in axis_specs:
            values = pd.to_numeric(group.get(value_column), errors="coerce")
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_group": candidate_group,
                    "layer": int(layer),
                    "head": int(head),
                    "axis": axis,
                    "mean_value": _mean_or_none(values),
                    "statuses": _join_unique(group.get(status_column, pd.Series(dtype=str))),
                }
            )
    return pd.DataFrame(rows)


def characterization_multiseed_summary(
    by_candidate: pd.DataFrame,
    by_axis: pd.DataFrame,
    manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts = (
        by_candidate["final_characterization_status"].value_counts().to_dict()
        if not by_candidate.empty
        else {}
    )
    strengthened = by_candidate[
        by_candidate["final_characterization_status"] == "strengthened_local_candidate"
    ].copy()
    primary = by_candidate[_head_labels(by_candidate).isin(PRIMARY_HEADS)].copy()
    raw = by_candidate[by_candidate["candidate_group"] == "prior_raw_attention_failed"].copy()
    negative = by_candidate[by_candidate["candidate_group"].str.startswith("negative_control")].copy()
    executive = (
        "One or more candidates strengthened under local characterization, but this is still "
        "not an induction-head discovery or circuit claim."
        if not strengthened.empty
        else "No prior candidate survived characterization as a strengthened local candidate. "
        "The previous head-specific and held-out effects should be treated as fragile local artifacts."
    )
    return {
        "n_runs": len(manifests),
        "seeds": [manifest["seed"] for manifest in manifests],
        "n_candidates": int(len(by_candidate)),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "strengthened_candidates": _candidate_rows(strengthened),
        "primary_candidate_outcomes": _candidate_rows(primary),
        "prior_raw_attention_outcomes": _candidate_rows(raw),
        "negative_control_outcomes": _candidate_rows(negative),
        "axis_summary": by_axis.to_dict("records") if not by_axis.empty else [],
        "executive_summary": executive,
        "limitation": (
            "This characterization uses fixed GPT-2 small heads, synthetic local prompts, "
            "hook_z interventions, and local OV/QK diagnostics. It is not a mechanism claim, "
            "a full circuit claim, or a broad GPT-2 claim."
        ),
    }


def render_characterization_markdown(
    summary: dict[str, Any],
    by_candidate: pd.DataFrame,
    by_axis: pd.DataFrame,
) -> str:
    lines = [
        "# Head-Specific Candidate Characterization v1",
        "",
        "## Executive summary",
        "",
        summary["executive_summary"],
        "",
        "This report refuses induction-head, circuit, and broad GPT-2 claims. It only "
        "summarizes a fixed-candidate local characterization pass.",
        "",
        "## Prior held-out result",
        "",
        "The prior held-out robustness pass produced mixed and nonspecific results: no "
        "candidate was safe to treat as an induction-head discovery, and negative controls "
        "made the permissive survival rule look too weak.",
        "",
        "## Fixed candidates",
        "",
        _candidate_table(by_candidate),
        "",
        "## Characterization design",
        "",
        "- Primary metric: `true_vs_control_logit_diff`.",
        "- Fixed primary heads: `L7H7`, `L9H11`, `L7H11`, `L7H0`, `L0H8`.",
        "- Prior raw-attention heads and deterministic negative controls were retained.",
        "- No candidates were selected from characterization results.",
        "",
        "## Attention/effect alignment",
        "",
        _axis_table(by_axis, "attention_effect_alignment"),
        "",
        "## Position sensitivity",
        "",
        _position_table(by_candidate),
        "",
        "## Token-domain and sequence-length sensitivity",
        "",
        "Token-domain and sequence-length variation were built into the characterization "
        "prompt families. The consolidated status treats candidates with fragile or "
        "inconsistent evidence as downgraded or falsified.",
        "",
        "## OV diagnostics",
        "",
        _axis_table(by_axis, "ov_copy"),
        "",
        "## QK diagnostics",
        "",
        _axis_table(by_axis, "qk_source_selection"),
        "",
        "## Primary candidates",
        "",
        _group_outcomes(by_candidate, {"replicated_candidate", "random_comparison_replicated"}),
        "",
        "## Prior raw-attention comparison heads",
        "",
        _group_outcomes(by_candidate, {"prior_raw_attention_failed"}),
        "",
        "## Negative controls",
        "",
        _group_outcomes(by_candidate, {"negative_control_no_effect", "negative_control_nonspecific"}),
        "",
        "## Counterexamples",
        "",
        "Counterexample inspection artifacts should be read before treating any candidate "
        "as meaningful. Negative controls and failed prompt families are part of the result.",
        "",
        "## Final statuses",
        "",
        f"`{summary['status_counts']}`",
        "",
        "## What strengthened",
        "",
        _strengthened_text(summary),
        "",
        "## What downgraded",
        "",
        _status_text(by_candidate, "downgraded_candidate"),
        "",
        "## What falsified",
        "",
        _status_text(by_candidate, "falsified_candidate"),
        "",
        "## What this teaches",
        "",
        "A candidate can replicate under one generator and still fail local characterization. "
        "Attention/effect alignment, position sensitivity, OV/QK diagnostics, and negative "
        "controls make the pipeline harder to fool.",
        "",
        "## What this does not show",
        "",
        "This does not show an induction head, a circuit, or a broad GPT-2 property.",
        "",
        "## Next recommendation",
        "",
        "Write up the negative or downgraded result before adding new heads or new models.",
        "",
    ]
    return "\n".join(lines)


def plot_status_counts(by_candidate: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if by_candidate.empty:
        ax.text(0.5, 0.5, "No candidate rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    counts = by_candidate["final_characterization_status"].value_counts()
    colors = [STATUS_COLORS.get(status, "#4b5563") for status in counts.index]
    ax.bar(counts.index, counts.to_numpy(), color=colors)
    ax.set_ylabel("Candidates")
    ax.set_title("Candidate characterization final statuses")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_attention_effect_summary(by_candidate: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if by_candidate.empty:
        ax.text(0.5, 0.5, "No candidate rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    rows = by_candidate.copy()
    rows["head_label"] = _head_labels(rows)
    rows = rows.sort_values("mean_attention_effect_corr", ascending=False, na_position="last")
    ax.barh(
        rows["head_label"].iloc[::-1],
        rows["mean_attention_effect_corr"].fillna(0).iloc[::-1],
        color="#2563eb",
    )
    ax.axvline(0, color="#111827", linewidth=0.9)
    ax.set_xlabel("Mean Spearman attention/effect correlation")
    ax.set_ylabel("Head")
    ax.set_title("Attention/effect alignment by fixed candidate")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_ov_qk_summary(by_candidate: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if by_candidate.empty:
        ax.text(0.5, 0.5, "No candidate rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    rows = by_candidate.copy()
    rows["head_label"] = _head_labels(rows)
    x = range(len(rows))
    width = 0.38
    ax.bar(
        [value - width / 2 for value in x],
        rows["mean_ov_copy_margin"].fillna(0),
        width=width,
        label="OV copy margin",
        color="#0b6e4f",
    )
    ax.bar(
        [value + width / 2 for value in x],
        rows["mean_qk_source_margin"].fillna(0),
        width=width,
        label="QK source margin",
        color="#8f2d56",
    )
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(rows["head_label"], rotation=45, ha="right")
    ax.set_ylabel("Mean diagnostic margin")
    ax.set_title("OV/QK local diagnostics")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_position_specificity_summary(candidates: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    if candidates.empty:
        ax.text(0.5, 0.5, "No candidate rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    counts = candidates["position_specificity_status"].value_counts()
    ax.bar(counts.index, counts.to_numpy(), color="#4b5563")
    ax.set_ylabel("Seed-candidate rows")
    ax.set_title("Position specificity statuses")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _save_figure_pair(fig: plt.Figure, base_path: Path) -> list[Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    paths = [base_path.with_suffix(".png"), base_path.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def _negative_control_reference(rows: pd.DataFrame) -> dict[str, Any]:
    negative = rows[rows["candidate_group"].astype(str).str.startswith("negative_control")]
    if negative.empty:
        return {"support_seeds": 0, "mean_gap": None}
    support_counts = (
        negative.groupby("candidate_id")["characterization_seed_status"]
        .apply(lambda col: int((col == "characterization_supports").sum()))
        .max()
    )
    gaps = negative.groupby("candidate_id")["positive_minus_control_gap"].mean()
    return {
        "support_seeds": int(support_counts),
        "mean_gap": float(gaps.max()) if not gaps.empty else None,
    }


def _candidate_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        clean = {}
        for key, value in row.items():
            if pd.isna(value):
                clean[key] = None
            elif hasattr(value, "item"):
                clean[key] = value.item()
            else:
                clean[key] = value
        rows.append(clean)
    return rows


def _candidate_table(by_candidate: pd.DataFrame) -> str:
    lines = [
        "| head | candidate_id | group | final status | mean gap | mean corr | OV | QK |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if by_candidate.empty:
        lines.append("| none |  |  |  |  |  |  |  |")
        return "\n".join(lines)
    rows = by_candidate.copy()
    rows["head_label"] = _head_labels(rows)
    for row in rows.sort_values(["candidate_group", "layer", "head"]).itertuples(index=False):
        lines.append(
            f"| {row.head_label} | {row.candidate_id} | {row.candidate_group} | "
            f"{row.final_characterization_status} | {_fmt(row.mean_positive_minus_control_gap)} | "
            f"{_fmt(row.mean_attention_effect_corr)} | {row.ov_statuses} | {row.qk_statuses} |"
        )
    return "\n".join(lines)


def _axis_table(by_axis: pd.DataFrame, axis: str) -> str:
    rows = by_axis[by_axis["axis"] == axis].copy() if not by_axis.empty else pd.DataFrame()
    lines = ["| head | group | mean value | statuses |", "| --- | --- | --- | --- |"]
    if rows.empty:
        lines.append("| none |  |  |  |")
        return "\n".join(lines)
    rows["head_label"] = _head_labels(rows)
    for row in rows.sort_values("mean_value", ascending=False, na_position="last").itertuples(index=False):
        lines.append(
            f"| {row.head_label} | {row.candidate_group} | {_fmt(row.mean_value)} | {row.statuses} |"
        )
    return "\n".join(lines)


def _position_table(by_candidate: pd.DataFrame) -> str:
    lines = ["| head | group | position statuses |", "| --- | --- | --- |"]
    if by_candidate.empty:
        lines.append("| none |  |  |")
        return "\n".join(lines)
    rows = by_candidate.copy()
    rows["head_label"] = _head_labels(rows)
    for row in rows.sort_values(["candidate_group", "layer", "head"]).itertuples(index=False):
        lines.append(f"| {row.head_label} | {row.candidate_group} | {row.position_statuses} |")
    return "\n".join(lines)


def _group_outcomes(by_candidate: pd.DataFrame, groups: set[str]) -> str:
    rows = by_candidate[by_candidate["candidate_group"].isin(groups)].copy()
    if rows.empty:
        return "No rows."
    counts = rows["final_characterization_status"].value_counts().to_dict()
    return f"Outcome counts: `{counts}`.\n\n" + _candidate_table(rows)


def _strengthened_text(summary: dict[str, Any]) -> str:
    strengthened = summary.get("strengthened_candidates", [])
    if not strengthened:
        return "No candidate strengthened under the local characterization rule."
    labels = [f"L{row['layer']}H{row['head']}" for row in strengthened]
    return "Strengthened local candidates: " + ", ".join(labels) + "."


def _status_text(by_candidate: pd.DataFrame, status: str) -> str:
    rows = by_candidate[by_candidate["final_characterization_status"] == status].copy()
    if rows.empty:
        return "None."
    labels = [f"L{int(row.layer)}H{int(row.head)}" for row in rows.itertuples(index=False)]
    return ", ".join(labels) + "."


def _head_labels(frame: pd.DataFrame) -> pd.Series:
    return "L" + frame["layer"].astype(int).astype(str) + "H" + frame["head"].astype(int).astype(str)


def _join_unique(values: pd.Series) -> str:
    clean = sorted({str(value) for value in values.dropna() if str(value)})
    return ",".join(clean)


def _mean_or_none(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.mean()) if not finite.empty else None


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}"
