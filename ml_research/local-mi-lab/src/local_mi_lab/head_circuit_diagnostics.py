from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch

from local_mi_lab.heldout_prompts import generate_heldout_induction_prompts
from local_mi_lab.induction_metrics import resolve_induction_token_ids
from local_mi_lab.paths import resolve_repo_path
from local_mi_lab.tokens import token_id_for_single_token
from local_mi_lab.types import PromptRecord

DEFAULT_DIAGNOSTIC_FAMILIES = [
    "heldout_symbolic_longer",
    "heldout_word_sequences",
    "heldout_number_sequences",
    "heldout_double_repeat",
]


def ov_margin_from_logits(
    logits: torch.Tensor,
    *,
    true_token_id: int,
    wrong_token_id: int,
) -> dict[str, float | int]:
    scores = logits.detach().float().cpu()
    true_score = float(scores[true_token_id])
    wrong_score = float(scores[wrong_token_id])
    rank = int((scores > scores[true_token_id]).sum().item()) + 1
    return {
        "ov_true_token_score": true_score,
        "ov_wrong_token_score": wrong_score,
        "ov_copy_margin": true_score - wrong_score,
        "ov_rank_of_true_token": rank,
    }


def qk_margin_from_scores(
    scores: torch.Tensor,
    *,
    expected_source_position: int,
    distractor_positions: list[int],
) -> dict[str, float | None]:
    values = scores.detach().float().cpu()
    if expected_source_position < 0 or expected_source_position >= len(values):
        return {
            "mean_qk_expected_source_score": None,
            "mean_qk_best_distractor_score": None,
            "qk_source_margin": None,
        }
    source_score = float(values[expected_source_position])
    valid_distractors = [
        position for position in distractor_positions if 0 <= position < len(values)
    ]
    if not valid_distractors:
        return {
            "mean_qk_expected_source_score": source_score,
            "mean_qk_best_distractor_score": None,
            "qk_source_margin": None,
        }
    best_distractor = float(values[valid_distractors].max().item())
    return {
        "mean_qk_expected_source_score": source_score,
        "mean_qk_best_distractor_score": best_distractor,
        "qk_source_margin": source_score - best_distractor,
    }


def classify_ov_status(margin: float | None) -> str:
    if margin is None or pd.isna(margin):
        return "ov_unavailable"
    if margin > 0:
        return "ov_supports_copy"
    if margin < -0.1:
        return "ov_contradicts_copy"
    return "ov_weak"


def classify_qk_status(margin: float | None) -> str:
    if margin is None or pd.isna(margin):
        return "qk_unavailable"
    if margin > 0:
        return "qk_supports_source_selection"
    if margin < -0.1:
        return "qk_contradicts_source_selection"
    return "qk_weak"


def run_head_circuit_diagnostics(
    model: Any,
    config: dict[str, Any],
    candidate_set: str | Path,
    *,
    output_dir: str | Path,
    families: list[str] | None = None,
    examples_per_family: int = 12,
) -> dict[str, Any]:
    output_root = resolve_repo_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    seed = int(config["experiment"].get("seed", 0))
    records = _records_for_config(config, seed)
    selected_records = _selected_positive_records(
        records,
        families=families or DEFAULT_DIAGNOSTIC_FAMILIES,
        examples_per_family=examples_per_family,
        seed=seed,
    )
    candidates = pd.read_csv(resolve_repo_path(candidate_set))
    candidates = candidates[candidates["include_in_main"].astype(bool)].copy()
    example_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        rows = [
            _diagnose_example(model, candidate, record)
            for record in selected_records
        ]
        example_rows.extend(rows)
        candidate_rows.append(_aggregate_candidate_diagnostics(candidate, rows))
    examples = pd.DataFrame(example_rows)
    by_candidate = pd.DataFrame(candidate_rows)
    summary = _diagnostic_summary(examples, by_candidate, output_root)
    write_head_circuit_diagnostic_artifacts(output_root, examples, by_candidate, summary)
    return summary


def _diagnose_example(model: Any, candidate: Any, record: PromptRecord) -> dict[str, Any]:
    layer = int(candidate.layer)
    head = int(candidate.head)
    base = {
        "candidate_id": str(candidate.candidate_id),
        "candidate_group": str(candidate.candidate_group),
        "layer": layer,
        "head": head,
        "family": record.family,
        "example_id": record.example_id,
        "true_expected_next_token": record.true_expected_next_token,
        "wrong_or_control_token": record.wrong_or_control_token,
        "prompt": record.prompt,
    }
    try:
        token_ids = resolve_induction_token_ids(model.tokenizer, record)
        true_id = int(token_ids["true_token_id"])
        wrong_id = int(token_ids["wrong_or_control_token_id"])
        source_id = _source_token_id(model.tokenizer, record)
        ov = compute_ov_copy_diagnostic(model, layer, head, source_id, true_id, wrong_id)
        qk = compute_qk_source_diagnostic(model, layer, head, record)
    except (AttributeError, IndexError, KeyError, RuntimeError, ValueError) as exc:
        return {
            **base,
            "ov_true_token_score": None,
            "ov_wrong_token_score": None,
            "ov_copy_margin": None,
            "ov_rank_of_true_token": None,
            "ov_status": "ov_unavailable",
            "mean_qk_expected_source_score": None,
            "mean_qk_best_distractor_score": None,
            "qk_source_margin": None,
            "qk_status": "qk_unavailable",
            "diagnostic_error": str(exc),
        }
    return {
        **base,
        **ov,
        "ov_status": classify_ov_status(ov["ov_copy_margin"]),
        **qk,
        "qk_status": classify_qk_status(qk["qk_source_margin"]),
        "diagnostic_error": "",
    }


def _records_for_config(config: dict[str, Any], seed: int) -> list[PromptRecord]:
    if config["task"]["name"] == "candidate_characterization":
        from local_mi_lab.characterization_prompts import generate_characterization_prompts

        return generate_characterization_prompts(
            n_examples_per_family=int(config["task"]["n_examples_initial_per_family"]),
            families=list(config["task"]["families"]),
            seed=seed,
        )
    return generate_heldout_induction_prompts(
        n_examples_per_family=int(config["task"]["n_examples_initial_per_family"]),
        families=list(config["task"]["families"]),
        seed=seed,
    )


def compute_ov_copy_diagnostic(
    model: Any,
    layer: int,
    head: int,
    source_token_id: int,
    true_token_id: int,
    wrong_token_id: int,
) -> dict[str, float | int | None]:
    try:
        w_v = _head_weight(model.W_V, layer, head)
        w_o = _head_weight(model.W_O, layer, head)
        source_embed = model.W_E[source_token_id]
        ov_vector = source_embed @ w_v @ w_o
        logits = ov_vector @ model.W_U
    except (AttributeError, IndexError, RuntimeError) as exc:
        raise ValueError(f"OV diagnostic unavailable: {exc}") from exc
    return ov_margin_from_logits(
        logits,
        true_token_id=true_token_id,
        wrong_token_id=wrong_token_id,
    )


def compute_qk_source_diagnostic(
    model: Any,
    layer: int,
    head: int,
    record: PromptRecord,
) -> dict[str, float | None]:
    if record.expected_source_position_hint is None:
        raise ValueError("Record has no expected source position")
    try:
        w_q = _head_weight(model.W_Q, layer, head)
        w_k = _head_weight(model.W_K, layer, head)
        token_ids = model.to_tokens(record.prompt)[0]
        embeddings = model.W_E[token_ids]
        query_position = int(token_ids.shape[0]) - 1
        source_position = _model_position_for_prompt_index(
            record,
            seq_len=int(token_ids.shape[0]),
            prompt_index=int(record.expected_source_position_hint),
        )
        query = embeddings[query_position] @ w_q
        keys = embeddings @ w_k
        scores = keys @ query / math.sqrt(float(keys.shape[-1]))
    except (AttributeError, IndexError, RuntimeError) as exc:
        raise ValueError(f"QK diagnostic unavailable: {exc}") from exc
    distractors = [
        position
        for position in range(int(token_ids.shape[0]) - 1)
        if position not in {0, source_position}
    ]
    return qk_margin_from_scores(
        scores,
        expected_source_position=source_position,
        distractor_positions=distractors,
    )


def write_head_circuit_diagnostic_artifacts(
    output_root: Path,
    examples: pd.DataFrame,
    by_candidate: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    examples.to_csv(output_root / "head_circuit_diagnostics_examples.csv", index=False)
    by_candidate.to_csv(output_root / "head_circuit_diagnostics_by_candidate.csv", index=False)
    (output_root / "head_circuit_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "head_circuit_diagnostics.md").write_text(
        _diagnostic_markdown(summary, by_candidate),
        encoding="utf-8",
    )
    figures = output_root / "figures"
    figures.mkdir(exist_ok=True)
    _save_bar_figure(
        by_candidate,
        value_column="ov_copy_margin",
        title="OV copy margins",
        ylabel="True-minus-wrong OV logit margin",
        path_stem=figures / "ov_copy_margins",
    )
    _save_bar_figure(
        by_candidate,
        value_column="qk_source_margin",
        title="QK source margins",
        ylabel="Source-minus-distractor QK score",
        path_stem=figures / "qk_source_margins",
    )


def _aggregate_candidate_diagnostics(candidate: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    table = pd.DataFrame(rows)
    ov_margin = _safe_mean(table["ov_copy_margin"]) if "ov_copy_margin" in table else None
    qk_margin = _safe_mean(table["qk_source_margin"]) if "qk_source_margin" in table else None
    ov_true = _safe_mean(table["ov_true_token_score"]) if "ov_true_token_score" in table else None
    ov_wrong = _safe_mean(table["ov_wrong_token_score"]) if "ov_wrong_token_score" in table else None
    ov_rank = _safe_mean(table["ov_rank_of_true_token"]) if "ov_rank_of_true_token" in table else None
    qk_source = (
        _safe_mean(table["mean_qk_expected_source_score"])
        if "mean_qk_expected_source_score" in table
        else None
    )
    qk_distractor = (
        _safe_mean(table["mean_qk_best_distractor_score"])
        if "mean_qk_best_distractor_score" in table
        else None
    )
    return {
        "candidate_id": str(candidate.candidate_id),
        "candidate_group": str(candidate.candidate_group),
        "layer": int(candidate.layer),
        "head": int(candidate.head),
        "n_examples": int(len(rows)),
        "ov_true_token_score": ov_true,
        "ov_wrong_token_score": ov_wrong,
        "ov_copy_margin": ov_margin,
        "ov_rank_of_true_token": ov_rank,
        "ov_status": classify_ov_status(ov_margin),
        "mean_qk_expected_source_score": qk_source,
        "mean_qk_best_distractor_score": qk_distractor,
        "qk_source_margin": qk_margin,
        "qk_status": classify_qk_status(qk_margin),
    }


def _diagnostic_summary(
    examples: pd.DataFrame,
    by_candidate: pd.DataFrame,
    output_root: Path,
) -> dict[str, Any]:
    return {
        "output_dir": str(output_root),
        "n_example_rows": int(len(examples)),
        "n_candidates": int(by_candidate["candidate_id"].nunique()) if not by_candidate.empty else 0,
        "ov_status_counts": (
            by_candidate["ov_status"].value_counts().to_dict() if not by_candidate.empty else {}
        ),
        "qk_status_counts": (
            by_candidate["qk_status"].value_counts().to_dict() if not by_candidate.empty else {}
        ),
        "interpretation_note": (
            "OV and QK diagnostics are local signatures, not a complete circuit or an "
            "induction-head discovery."
        ),
    }


def _head_weight(weight: torch.Tensor, layer: int, head: int) -> torch.Tensor:
    if weight.ndim != 4:
        raise ValueError(f"Expected 4D head weight, got shape {tuple(weight.shape)}")
    return weight[layer, head]


def _source_token_id(tokenizer: Any, record: PromptRecord) -> int:
    if not record.expected_source_token:
        raise ValueError("Record has no expected source token")
    try:
        return token_id_for_single_token(tokenizer, f" {record.expected_source_token}")
    except ValueError:
        return token_id_for_single_token(tokenizer, record.expected_source_token)


def _model_position_for_prompt_index(
    record: PromptRecord,
    *,
    seq_len: int,
    prompt_index: int,
) -> int:
    bos_offset = max(seq_len - len(record.prompt_tokens_text), 0)
    return bos_offset + prompt_index


def _selected_positive_records(
    records: list[PromptRecord],
    *,
    families: list[str],
    examples_per_family: int,
    seed: int,
) -> list[PromptRecord]:
    rng = random.Random(seed)
    selected: list[PromptRecord] = []
    for family in families:
        rows = [
            record
            for record in records
            if record.family == family and record.is_positive_induction_example
        ]
        rows = sorted(rows, key=lambda record: record.family_index or 0)
        rng.shuffle(rows)
        selected.extend(sorted(rows[:examples_per_family], key=lambda record: record.family_index or 0))
    return selected


def _save_bar_figure(
    by_candidate: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    ylabel: str,
    path_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if by_candidate.empty:
        ax.text(0.5, 0.5, "No candidate rows", ha="center", va="center")
        ax.axis("off")
    else:
        rows = by_candidate.copy()
        rows[value_column] = pd.to_numeric(rows[value_column], errors="coerce")
        rows = rows.sort_values(value_column, ascending=False, na_position="last")
        labels = [f"L{int(row.layer)}H{int(row.head)}" for row in rows.itertuples()]
        ax.bar(labels, rows[value_column].fillna(0), color="#0b6e4f")
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def _diagnostic_markdown(summary: dict[str, Any], by_candidate: pd.DataFrame) -> str:
    lines = [
        "# Head Circuit Diagnostics",
        "",
        f"- Candidates: `{summary['n_candidates']}`",
        f"- Example rows: `{summary['n_example_rows']}`",
        f"- OV statuses: `{summary['ov_status_counts']}`",
        f"- QK statuses: `{summary['qk_status_counts']}`",
        "",
        "These diagnostics are local signatures only. They are not path patching or a circuit claim.",
        "",
        "| candidate | head | OV status | OV margin | QK status | QK margin |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    if by_candidate.empty:
        lines.append("| none |  |  |  |  |  |")
    else:
        for row in by_candidate.head(16).itertuples(index=False):
            lines.append(
                f"| {row.candidate_id} | L{int(row.layer)}H{int(row.head)} | "
                f"{row.ov_status} | {_fmt(row.ov_copy_margin)} | "
                f"{row.qk_status} | {_fmt(row.qk_source_margin)} |"
            )
    return "\n".join(lines) + "\n"


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}"
