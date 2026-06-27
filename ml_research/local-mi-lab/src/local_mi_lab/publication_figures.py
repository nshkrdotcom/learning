from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from local_mi_lab.paths import resolve_repo_path

STATUS_COLORS = {
    "replicated_head_specific_candidate": "#0b6e4f",
    "head_specific_positive_candidate": "#0b6e4f",
    "nonspecific": "#b7791f",
    "nonspecific_moves_controls": "#b7791f",
    "no_effect": "#6b7280",
    "no_positive_effect": "#6b7280",
    "not_replicated": "#8f2d56",
    "not_head_specific": "#b91c1c",
}


def generate_head_specific_publication_figures(
    report_dir: str | Path = "reports/head_specific_induction_causality_v1",
    output_dir: str | Path = "figures/head_specific_induction_causality_v1",
) -> dict[str, Any]:
    report_root = resolve_repo_path(report_dir)
    output_root = resolve_repo_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    by_head = pd.read_csv(report_root / "head_specific_multiseed_by_head.csv")
    manifest = json.loads((report_root / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (report_root / "head_specific_multiseed_summary.json").read_text(encoding="utf-8")
    )
    l7_examples_path = report_root / "replicated_head_L7H7_examples.csv"
    l7_examples = pd.read_csv(l7_examples_path) if l7_examples_path.exists() else pd.DataFrame()

    generated = []
    generated.extend(
        _save_figure_pair(
            plot_multiseed_candidate_gaps(by_head),
            output_root / "figure_1_multiseed_candidate_gaps",
        )
    )
    generated.extend(
        _save_figure_pair(
            plot_seed_status_counts(manifest),
            output_root / "figure_2_seed_status_counts",
        )
    )
    generated.extend(
        _save_figure_pair(
            plot_candidate_group_outcomes(by_head),
            output_root / "figure_3_candidate_group_outcomes",
        )
    )
    generated.extend(
        _save_figure_pair(
            plot_l7h7_example_effects(l7_examples),
            output_root / "figure_4_l7h7_example_effects",
        )
    )

    figure_manifest = {
        "source_report_dir": str(report_root),
        "output_dir": str(output_root),
        "generated_files": [str(path) for path in generated],
        "source_artifacts": [
            str(report_root / "head_specific_multiseed_by_head.csv"),
            str(report_root / "head_specific_multiseed_summary.json"),
            str(report_root / "run_manifest.json"),
            str(l7_examples_path),
        ],
        "primary_metric": "true_vs_control_logit_diff",
        "interpretation_limit": (
            "Figures summarize local practice artifacts. They do not establish an induction-head "
            "mechanism or a broad GPT-2 claim."
        ),
        "replicated_candidates": [
            f"L{row['layer']}H{row['head']}" for row in summary.get("replicated_candidates", [])
        ],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(figure_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return figure_manifest


def plot_multiseed_candidate_gaps(by_head: pd.DataFrame, top_k: int = 20) -> plt.Figure:
    top = (
        by_head.sort_values("mean_positive_minus_control_gap", ascending=False)
        .head(top_k)
        .copy()
    )
    labels = [_head_label(row) for row in top.itertuples()]
    colors = [STATUS_COLORS.get(row.replication_status, "#4b5563") for row in top.itertuples()]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.barh(labels[::-1], top["mean_positive_minus_control_gap"].iloc[::-1], color=colors[::-1])
    ax.axvline(0, color="#111827", linewidth=0.9)
    ax.set_xlabel("Mean positive-minus-control effect gap")
    ax.set_ylabel("Head")
    ax.set_title("Head-specific causal gaps across seeds")
    ax.grid(axis="x", alpha=0.25)
    _add_status_legend(ax)
    fig.tight_layout()
    return fig


def plot_seed_status_counts(manifest: dict[str, Any]) -> plt.Figure:
    rows = []
    for run in manifest.get("runs", []):
        run_dir = Path(run["run_dir"])
        path = run_dir / "head_specific_patching_by_head.csv"
        if path.exists():
            table = pd.read_csv(path)
            rows.append(table[["seed", "specificity_status"]])
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    if not rows:
        ax.text(0.5, 0.5, "No seed rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    counts = pd.concat(rows, ignore_index=True)
    pivot = counts.groupby(["seed", "specificity_status"]).size().unstack(fill_value=0)
    columns = [col for col in STATUS_COLORS if col in pivot.columns]
    pivot = pivot[columns]
    bottom = None
    x = range(len(pivot.index))
    for column in pivot.columns:
        values = pivot[column].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=STATUS_COLORS.get(column, "#4b5563"),
            label=column,
        )
        bottom = values if bottom is None else bottom + values
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(seed) for seed in pivot.index])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Heads")
    ax.set_title("Seed-level head-specific outcomes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize="small", loc="upper right")
    fig.tight_layout()
    return fig


def plot_candidate_group_outcomes(by_head: pd.DataFrame) -> plt.Figure:
    groups = []
    for row in by_head.itertuples():
        if row.replication_status == "replicated_head_specific_candidate":
            group = "replicated"
        elif bool(row.raw_attention_candidate_in_any_seed):
            group = "prior raw attention"
        elif bool(row.random_comparison_candidate_in_any_seed):
            group = "random comparison"
        else:
            group = "other selected heads"
        groups.append(group)
    table = by_head.copy()
    table["candidate_group"] = groups
    counts = (
        table.groupby(["candidate_group", "replication_status"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    bottom = None
    x = range(len(counts.index))
    for status in [col for col in STATUS_COLORS if col in counts.columns]:
        values = counts[status].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=STATUS_COLORS.get(status, "#4b5563"),
            label=status,
        )
        bottom = values if bottom is None else bottom + values
    ax.set_xticks(list(x))
    ax.set_xticklabels(counts.index, rotation=20, ha="right")
    ax.set_ylabel("Heads")
    ax.set_title("Outcome by candidate provenance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    return fig


def plot_l7h7_example_effects(examples: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    if examples.empty:
        ax.text(0.5, 0.5, "No L7H7 example rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    rows = examples.copy()
    rows["effect_size"] = pd.to_numeric(rows["effect_size"], errors="coerce")
    rows = rows.dropna(subset=["effect_size"])
    order = [
        "positive_repeat_sequence",
        "distractor_repeat_control",
        "random_expected_token_control",
        "same_token_frequency_control",
    ]
    data = [rows[rows["family"] == family]["effect_size"].to_numpy() for family in order]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(list(range(1, len(order) + 1)))
    ax.set_xticklabels([_short_family(family) for family in order])
    for index, values in enumerate(data, start=1):
        if len(values) == 0:
            continue
        jitter = _deterministic_jitter(len(values), index)
        ax.scatter(jitter, values, s=12, alpha=0.55, color="#0b6e4f")
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_ylabel("Effect size")
    ax.set_title("L7H7 example-level effects by family")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _save_figure_pair(fig: plt.Figure, stem: Path) -> list[Path]:
    svg = stem.with_suffix(".svg")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return [svg, pdf]


def _head_label(row: Any) -> str:
    label = f"L{int(row.layer)}H{int(row.head)}"
    flags = []
    if bool(row.raw_attention_candidate_in_any_seed):
        flags.append("raw")
    if bool(row.random_comparison_candidate_in_any_seed):
        flags.append("rand")
    return f"{label} ({','.join(flags)})" if flags else label


def _add_status_legend(ax: plt.Axes) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=status)
        for status, color in STATUS_COLORS.items()
    ]
    ax.legend(handles=handles, frameon=False, fontsize="small", loc="lower right")


def _short_family(family: str) -> str:
    return {
        "positive_repeat_sequence": "positive",
        "distractor_repeat_control": "distractor",
        "random_expected_token_control": "wrong target",
        "same_token_frequency_control": "same freq.",
    }.get(family, family)


def _deterministic_jitter(n_values: int, center: float) -> list[float]:
    if n_values <= 1:
        return [center]
    spread = 0.18
    step = (2 * spread) / max(n_values - 1, 1)
    return [center - spread + index * step for index in range(n_values)]
