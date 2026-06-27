from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from local_mi_lab.attention import attention_pattern_hook_name, prompt_word_token_spans
from local_mi_lab.paths import resolve_repo_path
from local_mi_lab.prompts import read_prompts_csv
from local_mi_lab.types import PromptRecord

ALIGNMENT_COLUMNS = [
    "seed",
    "candidate_id",
    "candidate_group",
    "layer",
    "head",
    "family",
    "heldout_family_type",
    "example_id",
    "intervention",
    "position_label",
    "effect_size",
    "effect_size_status",
    "attention_to_expected_source",
    "attention_to_best_distractor",
    "source_attention_margin",
    "target_token",
    "source_token",
    "clean_prompt",
    "corrupt_prompt",
]


def source_attention_margin(
    attention_to_expected_source: float | None,
    attention_to_best_distractor: float | None,
) -> float | None:
    if attention_to_expected_source is None or attention_to_best_distractor is None:
        return None
    if pd.isna(attention_to_expected_source) or pd.isna(attention_to_best_distractor):
        return None
    return float(attention_to_expected_source) - float(attention_to_best_distractor)


def pearson_corr(x_values: pd.Series, y_values: pd.Series) -> float | None:
    frame = pd.DataFrame({"x": x_values, "y": y_values}).apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna()
    if len(frame) < 2 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return None
    return float(frame["x"].corr(frame["y"], method="pearson"))


def spearman_corr(x_values: pd.Series, y_values: pd.Series) -> float | None:
    frame = pd.DataFrame({"x": x_values, "y": y_values}).apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna()
    if len(frame) < 2 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return None
    return float(frame["x"].rank(method="average").corr(frame["y"].rank(method="average")))


def classify_alignment(
    *,
    positive_corr: float | None,
    control_corr: float | None,
    positive_source_margin: float | None,
    control_source_margin: float | None,
    n_valid_examples: int,
    min_valid_examples: int = 4,
) -> str:
    if n_valid_examples < min_valid_examples:
        return "insufficient_examples"
    if positive_corr is None or positive_source_margin is None:
        return "insufficient_examples"
    if positive_corr <= 0 or positive_source_margin <= 0:
        return "not_aligned"
    control_matches = False
    if control_corr is not None and control_corr >= positive_corr:
        control_matches = True
    if (
        control_source_margin is not None
        and positive_source_margin is not None
        and control_source_margin >= positive_source_margin
    ):
        control_matches = True
    return "aligned_but_control_like" if control_matches else "aligned_positive_specific"


def summarize_attention_effect_by_candidate(examples: pd.DataFrame) -> pd.DataFrame:
    if examples.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "candidate_group",
                "layer",
                "head",
                "n_examples",
                "n_valid_examples",
                "mean_effect_size",
                "mean_attention_to_expected_source",
                "mean_source_attention_margin",
                "spearman_attention_effect_corr",
                "pearson_attention_effect_corr",
                "positive_family_corr",
                "control_family_corr",
                "alignment_status",
            ]
        )
    rows: list[dict[str, Any]] = []
    table = examples.copy()
    table["effect_size_numeric"] = pd.to_numeric(table["effect_size"], errors="coerce")
    table["attention_to_expected_source_numeric"] = pd.to_numeric(
        table["attention_to_expected_source"],
        errors="coerce",
    )
    table["source_attention_margin_numeric"] = pd.to_numeric(
        table["source_attention_margin"],
        errors="coerce",
    )
    for key, group in table.groupby(["candidate_id", "candidate_group", "layer", "head"]):
        candidate_id, candidate_group, layer, head = key
        valid = group.dropna(
            subset=[
                "effect_size_numeric",
                "attention_to_expected_source_numeric",
                "source_attention_margin_numeric",
            ]
        )
        positive = valid[valid["heldout_family_type"] == "positive"]
        controls = valid[valid["heldout_family_type"] == "control"]
        positive_corr = pearson_corr(
            positive["attention_to_expected_source_numeric"],
            positive["effect_size_numeric"],
        )
        control_corr = pearson_corr(
            controls["attention_to_expected_source_numeric"],
            controls["effect_size_numeric"],
        )
        positive_margin = _safe_mean(positive["source_attention_margin_numeric"])
        control_margin = _safe_mean(controls["source_attention_margin_numeric"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_group": candidate_group,
                "layer": int(layer),
                "head": int(head),
                "n_examples": int(len(group)),
                "n_valid_examples": int(len(valid)),
                "mean_effect_size": _safe_mean(valid["effect_size_numeric"]),
                "mean_attention_to_expected_source": _safe_mean(
                    valid["attention_to_expected_source_numeric"]
                ),
                "mean_source_attention_margin": _safe_mean(
                    valid["source_attention_margin_numeric"]
                ),
                "spearman_attention_effect_corr": spearman_corr(
                    valid["attention_to_expected_source_numeric"],
                    valid["effect_size_numeric"],
                ),
                "pearson_attention_effect_corr": pearson_corr(
                    valid["attention_to_expected_source_numeric"],
                    valid["effect_size_numeric"],
                ),
                "positive_family_corr": positive_corr,
                "control_family_corr": control_corr,
                "alignment_status": classify_alignment(
                    positive_corr=positive_corr,
                    control_corr=control_corr,
                    positive_source_margin=positive_margin,
                    control_source_margin=control_margin,
                    n_valid_examples=len(valid),
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["alignment_status", "mean_effect_size"],
        ascending=[True, False],
        na_position="last",
    )


def run_attention_effect_alignment(
    model: Any,
    *,
    heldout_run: str | Path,
    candidate_set: str | Path,
    output_dir: str | Path,
    example_limit: int | None = None,
) -> dict[str, Any]:
    run_dir = resolve_repo_path(heldout_run)
    output_root = resolve_repo_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(run_dir / "heldout_robustness_results.csv")
    candidates = pd.read_csv(resolve_repo_path(candidate_set))
    prompts = {record.example_id: record for record in read_prompts_csv(run_dir / "prompts.csv")}
    candidate_keys = {
        (int(row.layer), int(row.head)): str(row.candidate_group)
        for row in candidates.itertuples()
        if bool(row.include_in_main)
    }
    selected = results[
        results.apply(lambda row: (int(row["layer"]), int(row["head"])) in candidate_keys, axis=1)
    ].copy()
    if example_limit is not None:
        selected = (
            selected.sort_values(["candidate_id", "family", "example_id"])
            .groupby(["candidate_id", "family"], as_index=False)
            .head(example_limit)
        )
    attention_lookup = _compute_attention_lookup(model, selected, prompts)
    examples = _merge_effects_with_attention(selected, prompts, attention_lookup)
    by_candidate = summarize_attention_effect_by_candidate(examples)
    summary = _alignment_summary(examples, by_candidate, run_dir, candidate_set)
    _write_alignment_artifacts(output_root, examples, by_candidate, summary)
    return summary


def _compute_attention_lookup(
    model: Any,
    selected: pd.DataFrame,
    prompts: dict[str, PromptRecord],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    heads_by_layer: dict[int, set[int]] = {}
    for row in selected.itertuples(index=False):
        heads_by_layer.setdefault(int(row.layer), set()).add(int(row.head))
    hook_names = [attention_pattern_hook_name(layer) for layer in sorted(heads_by_layer)]
    unique_examples = selected["example_id"].drop_duplicates().tolist()
    for example_id in tqdm(unique_examples, desc="Attention/effect alignment"):
        record = prompts[str(example_id)]
        tokens = model.to_tokens(record.prompt)
        target_position = int(tokens.shape[1]) - 1
        previous_position = None
        if record.expected_source_position_hint is not None:
            spans = prompt_word_token_spans(model.tokenizer, tokens, record.prompt_tokens_text)
            previous_position = spans[int(record.expected_source_position_hint)][1]
            target_position = spans[-1][1]
        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)
        for layer, heads in heads_by_layer.items():
            pattern = cache[attention_pattern_hook_name(layer)][0].detach().cpu()
            for head in heads:
                distribution = pattern[int(head), target_position, : target_position + 1].numpy()
                source_attention = (
                    float(distribution[previous_position])
                    if previous_position is not None
                    else None
                )
                distractor_attention = _best_distractor_attention(
                    distribution,
                    target_position=target_position,
                    source_position=previous_position,
                )
                lookup[(str(example_id), int(layer), int(head))] = {
                    "attention_to_expected_source": source_attention,
                    "attention_to_best_distractor": distractor_attention,
                    "source_attention_margin": source_attention_margin(
                        source_attention,
                        distractor_attention,
                    ),
                    "target_token": record.true_expected_next_token,
                    "source_token": record.expected_source_token,
                }
    return lookup


def _merge_effects_with_attention(
    results: pd.DataFrame,
    prompts: dict[str, PromptRecord],
    attention_lookup: dict[tuple[str, int, int], dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in results.itertuples(index=False):
        record = prompts[str(row.example_id)]
        attention = attention_lookup.get((str(row.example_id), int(row.layer), int(row.head)), {})
        rows.append(
            {
                "seed": int(row.seed),
                "candidate_id": str(row.candidate_id),
                "candidate_group": str(row.candidate_group),
                "layer": int(row.layer),
                "head": int(row.head),
                "family": str(row.family),
                "heldout_family_type": str(row.heldout_family_type),
                "example_id": str(row.example_id),
                "intervention": str(row.intervention),
                "position_label": str(row.position_label),
                "effect_size": row.effect_size,
                "effect_size_status": str(row.effect_size_status),
                "attention_to_expected_source": attention.get("attention_to_expected_source"),
                "attention_to_best_distractor": attention.get("attention_to_best_distractor"),
                "source_attention_margin": attention.get("source_attention_margin"),
                "target_token": attention.get("target_token", record.true_expected_next_token),
                "source_token": attention.get("source_token", record.expected_source_token),
                "clean_prompt": str(row.clean_prompt),
                "corrupt_prompt": str(row.corrupt_prompt),
            }
        )
    return pd.DataFrame(rows, columns=ALIGNMENT_COLUMNS)


def _best_distractor_attention(
    distribution: np.ndarray,
    *,
    target_position: int,
    source_position: int | None,
) -> float | None:
    if target_position <= 0:
        return None
    mask = np.ones(target_position + 1, dtype=bool)
    mask[target_position] = False
    mask[0] = False
    if source_position is not None and 0 <= source_position < len(mask):
        mask[source_position] = False
    values = distribution[mask]
    if len(values) == 0:
        return None
    return float(np.max(values))


def _alignment_summary(
    examples: pd.DataFrame,
    by_candidate: pd.DataFrame,
    heldout_run: Path,
    candidate_set: str | Path,
) -> dict[str, Any]:
    return {
        "heldout_run": str(heldout_run),
        "candidate_set": str(resolve_repo_path(candidate_set)),
        "n_example_rows": int(len(examples)),
        "n_candidates": int(by_candidate["candidate_id"].nunique()) if not by_candidate.empty else 0,
        "alignment_status_counts": (
            by_candidate["alignment_status"].value_counts().to_dict()
            if not by_candidate.empty
            else {}
        ),
        "top_aligned_candidates": _records(
            by_candidate.sort_values(
                "pearson_attention_effect_corr",
                ascending=False,
                na_position="last",
            ).head(10)
        ),
        "interpretation_note": (
            "Attention/effect alignment is stronger than raw attention alone, but it is "
            "still not a circuit or induction-head discovery claim."
        ),
    }


def _write_alignment_artifacts(
    output_root: Path,
    examples: pd.DataFrame,
    by_candidate: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    examples.to_csv(output_root / "attention_effect_examples.csv", index=False)
    by_candidate.to_csv(output_root / "attention_effect_by_candidate.csv", index=False)
    (output_root / "attention_effect_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "attention_effect_alignment.md").write_text(
        _alignment_markdown(summary, by_candidate),
        encoding="utf-8",
    )
    figures = output_root / "figures"
    figures.mkdir(exist_ok=True)
    for head_label in ["L7H7", "L9H11", "L7H11"]:
        layer, head = _parse_head_label(head_label)
        fig = plot_attention_effect_scatter(
            examples[(examples["layer"] == layer) & (examples["head"] == head)],
            head_label,
        )
        png_path = figures / f"attention_effect_scatter_{head_label}.png"
        svg_path = figures / f"attention_effect_scatter_{head_label}.svg"
        fig.savefig(png_path, dpi=160, bbox_inches="tight")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        plt.close(fig)


def plot_attention_effect_scatter(examples: pd.DataFrame, head_label: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if examples.empty:
        ax.text(0.5, 0.5, "No rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    rows = examples.copy()
    rows["effect_size_numeric"] = pd.to_numeric(rows["effect_size"], errors="coerce")
    rows["attention_to_expected_source_numeric"] = pd.to_numeric(
        rows["attention_to_expected_source"],
        errors="coerce",
    )
    colors = {"positive": "#0b6e4f", "control": "#8f2d56"}
    for family_type, group in rows.groupby("heldout_family_type"):
        ax.scatter(
            group["attention_to_expected_source_numeric"],
            group["effect_size_numeric"],
            s=16,
            alpha=0.65,
            label=str(family_type),
            color=colors.get(str(family_type), "#4b5563"),
        )
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Attention to expected source")
    ax.set_ylabel("Effect size")
    ax.set_title(f"Attention/effect alignment: {head_label}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    return fig


def _alignment_markdown(summary: dict[str, Any], by_candidate: pd.DataFrame) -> str:
    lines = [
        "# Attention/Effect Alignment",
        "",
        f"- Held-out run: `{summary['heldout_run']}`",
        f"- Candidates: `{summary['n_candidates']}`",
        f"- Example rows: `{summary['n_example_rows']}`",
        f"- Alignment status counts: `{summary['alignment_status_counts']}`",
        "",
        "Attention/effect alignment is a local diagnostic. It is not a circuit claim.",
        "",
        "## Top Rows",
        "",
        "| candidate | head | status | pearson corr | mean source margin |",
        "| --- | --- | --- | --- | --- |",
    ]
    if by_candidate.empty:
        lines.append("| none |  |  |  |  |")
    else:
        top = by_candidate.sort_values(
            "pearson_attention_effect_corr",
            ascending=False,
            na_position="last",
        ).head(10)
        for row in top.itertuples(index=False):
            lines.append(
                f"| {row.candidate_id} | L{int(row.layer)}H{int(row.head)} | "
                f"{row.alignment_status} | {_fmt(row.pearson_attention_effect_corr)} | "
                f"{_fmt(row.mean_source_attention_margin)} |"
            )
    return "\n".join(lines) + "\n"


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.where(pd.notna(df), None).to_json(orient="records"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _parse_head_label(label: str) -> tuple[int, int]:
    normalized = label.strip().upper()
    layer_text, head_text = normalized[1:].split("H", maxsplit=1)
    return int(layer_text), int(head_text)


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}"
