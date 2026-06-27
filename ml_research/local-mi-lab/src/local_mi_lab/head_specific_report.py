from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from local_mi_lab.paths import resolve_repo_path

MULTISEED_COLUMNS = [
    "layer",
    "head",
    "n_seeds",
    "seeds_present",
    "mean_positive_effect",
    "mean_max_control_effect",
    "mean_positive_minus_control_gap",
    "min_positive_minus_control_gap",
    "n_positive_specific_seeds",
    "n_nonspecific_seeds",
    "n_no_effect_seeds",
    "replication_status",
    "raw_attention_candidate_in_any_seed",
    "random_comparison_candidate_in_any_seed",
]


def load_head_specific_run(run_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = resolve_repo_path(run_dir)
    by_head_path = root / "head_specific_patching_by_head.csv"
    summary_path = root / "head_specific_induction_summary.json"
    if not by_head_path.exists():
        raise FileNotFoundError(f"Missing {by_head_path}")
    by_head = pd.read_csv(by_head_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    flags = candidate_flags_for_run(root)
    by_head["raw_attention_candidate_in_seed"] = by_head.apply(
        lambda row: (int(row["layer"]), int(row["head"])) in flags["raw_attention"],
        axis=1,
    )
    by_head["random_comparison_candidate_in_seed"] = by_head.apply(
        lambda row: (int(row["layer"]), int(row["head"])) in flags["random_comparison"],
        axis=1,
    )
    manifest = {
        "run_dir": str(root),
        "seed": int(by_head["seed"].iloc[0]) if "seed" in by_head and not by_head.empty else None,
        "n_heads": int(len(by_head)),
        "summary": summary,
        "raw_attention_heads": sorted([f"L{layer}H{head}" for layer, head in flags["raw_attention"]]),
        "random_comparison_heads": sorted(
            [f"L{layer}H{head}" for layer, head in flags["random_comparison"]]
        ),
    }
    return by_head, manifest


def candidate_flags_for_run(run_dir: str | Path) -> dict[str, set[tuple[int, int]]]:
    root = Path(run_dir)
    source_run = _source_run_path(root)
    if source_run is None:
        source_run = root
    raw: set[tuple[int, int]] = set()
    random_comparison: set[tuple[int, int]] = set()
    candidate_csv = source_run / "controlled_patching_candidates.csv"
    if candidate_csv.exists():
        candidates = pd.read_csv(candidate_csv)
        for row in candidates.itertuples():
            if pd.isna(row.head):
                continue
            key = (int(row.layer), int(row.head))
            if row.source == "top_raw_positive_attention":
                raw.add(key)
            elif row.source == "random_comparison":
                random_comparison.add(key)
    attention_summary = source_run / "attention_summary.json"
    if attention_summary.exists() and not raw:
        payload = json.loads(attention_summary.read_text(encoding="utf-8"))
        for row in payload.get("top_heads_on_positive_examples", [])[:5]:
            raw.add((int(row["layer"]), int(row["head"])))
    return {"raw_attention": raw, "random_comparison": random_comparison}


def summarize_multiseed_heads(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=MULTISEED_COLUMNS)
    rows = pd.concat(frames, ignore_index=True)
    summaries = []
    for (layer, head), group in rows.groupby(["layer", "head"], sort=True):
        seeds = sorted(int(seed) for seed in group["seed"].dropna().unique())
        status_counts = group["specificity_status"].value_counts().to_dict()
        relevant_min_gap = _minimum_relevant_gap(group)
        summaries.append(
            {
                "layer": int(layer),
                "head": int(head),
                "n_seeds": len(seeds),
                "seeds_present": ",".join(str(seed) for seed in seeds),
                "mean_positive_effect": float(group["positive_mean_effect_size"].mean()),
                "mean_max_control_effect": float(group["max_control_mean_effect_size"].mean()),
                "mean_positive_minus_control_gap": float(
                    group["positive_minus_control_effect_gap"].mean()
                ),
                "min_positive_minus_control_gap": relevant_min_gap,
                "n_positive_specific_seeds": int(
                    status_counts.get("head_specific_positive_candidate", 0)
                ),
                "n_nonspecific_seeds": int(status_counts.get("nonspecific_moves_controls", 0)),
                "n_no_effect_seeds": int(status_counts.get("no_positive_effect", 0)),
                "replication_status": classify_replication_status(group),
                "raw_attention_candidate_in_any_seed": bool(
                    group["raw_attention_candidate_in_seed"].any()
                    if "raw_attention_candidate_in_seed" in group
                    else False
                ),
                "random_comparison_candidate_in_any_seed": bool(
                    group["random_comparison_candidate_in_seed"].any()
                    if "random_comparison_candidate_in_seed" in group
                    else False
                ),
            }
        )
    return pd.DataFrame(summaries, columns=MULTISEED_COLUMNS).sort_values(
        "mean_positive_minus_control_gap",
        ascending=False,
    )


def classify_replication_status(group: pd.DataFrame) -> str:
    if not bool(group["head_specific_patch"].all()):
        return "not_head_specific"
    n_seeds = int(group["seed"].nunique())
    if n_seeds < 2:
        return "insufficient_seeds"
    statuses = group["specificity_status"].tolist()
    positive = group[group["specificity_status"] == "head_specific_positive_candidate"]
    positive_count = len(positive)
    nonspecific_count = statuses.count("nonspecific_moves_controls")
    no_effect_count = statuses.count("no_positive_effect")
    if (
        positive_count >= 2
        and float(positive["positive_minus_control_effect_gap"].min()) > 0.0
        and nonspecific_count == 0
    ):
        return "replicated_head_specific_candidate"
    if nonspecific_count >= 1:
        return "nonspecific"
    if no_effect_count >= max(1, (n_seeds + 1) // 2):
        return "no_effect"
    if positive_count == 1:
        return "not_replicated"
    return "not_replicated"


def _minimum_relevant_gap(group: pd.DataFrame) -> float:
    positive = group[group["specificity_status"] == "head_specific_positive_candidate"]
    if not positive.empty:
        return float(positive["positive_minus_control_effect_gap"].min())
    return float(group["positive_minus_control_effect_gap"].min())


def compare_head_specific_runs(run_dirs: list[str | Path], output_dir: str | Path) -> dict[str, Path]:
    root = resolve_repo_path(output_dir)
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        frame, manifest = load_head_specific_run(run_dir)
        frames.append(frame)
        manifests.append(manifest)
    by_head = summarize_multiseed_heads(frames)
    summary = multiseed_summary(by_head, manifests)

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "run_manifest.json"
    by_head_path = root / "head_specific_multiseed_by_head.csv"
    summary_path = root / "head_specific_multiseed_summary.json"
    markdown_path = root / "head_specific_induction_causality_v1.md"
    gap_fig = figures_dir / "multiseed_head_gaps.png"
    status_fig = figures_dir / "status_by_seed.png"

    manifest_path.write_text(json.dumps({"runs": manifests}, indent=2) + "\n", encoding="utf-8")
    by_head.to_csv(by_head_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_multiseed_markdown(summary, by_head, manifests), encoding="utf-8")
    plot_multiseed_head_gaps(by_head, gap_fig)
    plot_status_by_seed(pd.concat(frames, ignore_index=True), status_fig)
    return {
        "manifest": manifest_path,
        "by_head": by_head_path,
        "summary": summary_path,
        "markdown": markdown_path,
        "gap_figure": gap_fig,
        "status_figure": status_fig,
    }


def multiseed_summary(by_head: pd.DataFrame, manifests: list[dict[str, Any]]) -> dict[str, Any]:
    replicated = by_head[
        by_head["replication_status"] == "replicated_head_specific_candidate"
    ].copy()
    top = by_head.sort_values("mean_positive_minus_control_gap", ascending=False).head(10)
    status_counts = by_head["replication_status"].value_counts().to_dict() if not by_head.empty else {}
    return {
        "n_runs": len(manifests),
        "seeds": [manifest["seed"] for manifest in manifests],
        "n_heads": int(len(by_head)),
        "replicated_candidates": _head_rows(replicated),
        "n_replicated_candidates": int(len(replicated)),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "top_heads_by_mean_gap": _head_rows(top),
        "raw_attention_candidate_outcomes": _head_rows(
            by_head[by_head["raw_attention_candidate_in_any_seed"]]
        ),
        "random_comparison_head_outcomes": _head_rows(
            by_head[by_head["random_comparison_candidate_in_any_seed"]]
        ),
        "executive_summary": (
            "This experiment identified at least one narrow replicated head-specific causal "
            "candidate under this prompt set and metric. It is still not a full induction-head "
            "discovery; it requires manual inspection, broader prompts, and benchmark comparison."
            if not replicated.empty
            else "This experiment did not identify a replicated head-specific induction "
            "candidate. The prior raw-attention and layer-level patching results should be "
            "treated as false-positive-prone practice artifacts."
        ),
        "limitation": (
            "The result is limited to GPT-2 small, these synthetic prompt families, selected "
            "layers, final-position hook_z patching, and the true-vs-control logit-diff metric."
        ),
    }


def render_multiseed_markdown(
    summary: dict[str, Any],
    by_head: pd.DataFrame,
    manifests: list[dict[str, Any]],
) -> str:
    replicated = summary["replicated_candidates"]
    raw = summary["raw_attention_candidate_outcomes"][:10]
    random_heads = summary["random_comparison_head_outcomes"][:10]
    top = summary["top_heads_by_mean_gap"][:10]
    return "\n".join(
        [
            "# Head-Specific Induction Causality v1",
            "",
            "## Executive summary",
            "",
            summary["executive_summary"],
            "",
            "This is practice evidence, not a mechanism or circuit claim.",
            "",
            "## Experimental setup",
            "",
            "- Model: `gpt2-small`",
            "- Primary metric: `true_vs_control_logit_diff`",
            "- Intervention: `head_clean_to_corrupt_patch`",
            "- Patch site: `blocks.<layer>.attn.hook_z` when available",
            "- Position: final token",
            "- Prompt families: positive, distractor, random-expected-token, and same-token-frequency controls",
            "",
            "## Prior false-positive result",
            "",
            "Earlier raw previous-occurrence attention and layer-level `attn_out` patching were false-positive-prone. Layer-level `attn_out` patching is not head-specific.",
            "",
            "## Hook verification",
            "",
            "The head-specific runs recorded `head_specific_patch=true` and `actual_patch_scope=single_head_z` when supported. Any non-head-specific run would be classified separately.",
            "",
            "## Metrics",
            "",
            "`true_vs_control_logit_diff` is the primary metric. Target-logit movement is weaker and is not used as the main replication rule.",
            "",
            "## Seeds and runs",
            "",
            *_run_lines(manifests),
            "",
            "## Head-specific results",
            "",
            *_table_lines(top, ["layer", "head", "mean_positive_minus_control_gap", "replication_status"]),
            "",
            "## Raw attention candidates",
            "",
            *_table_lines(raw, ["layer", "head", "mean_positive_minus_control_gap", "replication_status"]),
            "",
            "## Random comparison heads",
            "",
            *_table_lines(
                random_heads,
                ["layer", "head", "mean_positive_minus_control_gap", "replication_status"],
            ),
            "",
            "## Controls",
            "",
            "A candidate that moves controls as much as positives is nonspecific. Controls are part of the causal comparison, not only the descriptive phase.",
            "",
            "## Replication analysis",
            "",
            f"- Heads compared: `{summary['n_heads']}`",
            f"- Status counts: `{summary['status_counts']}`",
            f"- Replicated candidates: `{summary['n_replicated_candidates']}`",
            "",
            "## What survived",
            "",
            _survival_text(replicated),
            "",
            "## What failed",
            "",
            "Raw attention alone failed as a sufficient filter. Random comparison heads are reported separately and cannot be treated as induction evidence without manual follow-up.",
            "",
            "## What this teaches",
            "",
            "Head-specific causal checks are stricter than descriptive attention. Multi-seed replication changes how seriously a small causal gap should be taken.",
            "",
            "## What this does not show",
            "",
            "This does not establish a broad induction-head mechanism. It does not identify a full circuit, does not test path patching, and does not generalize beyond this local practice prompt set.",
            "",
            "## Next experiment recommendation",
            "",
            _next_recommendation(replicated),
            "",
        ]
    )


def plot_multiseed_head_gaps(by_head: pd.DataFrame, path: Path, top_k: int = 24) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    if by_head.empty:
        ax.text(0.5, 0.5, "No head rows", ha="center", va="center")
        ax.axis("off")
    else:
        top = by_head.sort_values("mean_positive_minus_control_gap", ascending=False).head(top_k)
        labels = [f"L{int(row.layer)}H{int(row.head)}" for row in top.itertuples()]
        ax.bar(labels, top["mean_positive_minus_control_gap"], color="#0b6e4f")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Mean positive-minus-control gap")
        ax.set_title("Multi-seed head-specific gaps")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_status_by_seed(rows: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if rows.empty:
        ax.text(0.5, 0.5, "No rows", ha="center", va="center")
        ax.axis("off")
    else:
        counts = rows.groupby(["seed", "specificity_status"]).size().unstack(fill_value=0)
        counts.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Head count")
        ax.set_title("Head-specific status by seed")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _source_run_path(root: Path) -> Path | None:
    path = root / "source_run.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return Path(text) if text else None


def _head_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        payload = row._asdict()
        rows.append(
            {
                key: _jsonable(value)
                for key, value in payload.items()
                if key in MULTISEED_COLUMNS
            }
        )
    return rows


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _run_lines(manifests: list[dict[str, Any]]) -> list[str]:
    return [
        f"- Seed `{manifest['seed']}`: `{manifest['run_dir']}`; heads `{manifest['n_heads']}`"
        for manifest in manifests
    ]


def _table_lines(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _survival_text(replicated: list[dict[str, Any]]) -> str:
    if not replicated:
        return "No head met the pre-registered replicated-head-specific rule."
    labels = [f"L{int(row['layer'])}H{int(row['head'])}" for row in replicated]
    return "Replicated candidates under this narrow rule: " + ", ".join(labels) + "."


def _next_recommendation(replicated: list[dict[str, Any]]) -> str:
    if not replicated:
        return "Stop and write the negative-result lesson before changing tasks."
    return (
        "Manually inspect the replicated candidate examples and run a smaller held-out prompt "
        "check before any stronger claim."
    )
