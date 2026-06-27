from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from local_mi_lab.config import selected_layers
from local_mi_lab.models import n_layers
from local_mi_lab.plots import plot_attention_induction_scores
from local_mi_lab.tokens import decode_token, encode_text
from local_mi_lab.types import PromptRecord

ATTENTION_LIMITATION = (
    "Attention-pattern evidence is descriptive. It can suggest where to inspect, "
    "but it is not causal evidence by itself."
)


def attention_pattern_hook_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_pattern"


def attention_entropy(distribution: Any) -> float:
    values = np.asarray(distribution, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected one-dimensional attention distribution, got {values.shape}")
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probs = values / total
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log(nonzero)))


def require_induction_metadata(record: PromptRecord) -> None:
    if record.task != "induction":
        raise ValueError(f"Attention-pattern inspection expects induction records, got {record.task}")
    if not record.prompt_tokens_text:
        raise ValueError(
            f"Prompt {record.example_id} is missing prompt_tokens_text metadata. "
            "Regenerate prompts with scripts/build_toy_prompts.py or rerun baseline behavior."
        )
    if record.expected_source_position_hint is None:
        raise ValueError(
            f"Prompt {record.example_id} is missing expected_source_position_hint metadata."
        )
    if record.expected_source_position_hint < 0:
        raise ValueError(f"Prompt {record.example_id} has a negative source position")
    if record.expected_source_position_hint >= len(record.prompt_tokens_text):
        raise ValueError(f"Prompt {record.example_id} source position is outside prompt tokens")


def prompt_word_token_spans(
    tokenizer: Any,
    model_tokens: torch.Tensor,
    prompt_tokens_text: list[str],
) -> list[tuple[int, int]]:
    encoded_prompt_ids: list[int] = []
    spans_without_bos: list[tuple[int, int]] = []
    cursor = 0
    for word_index, token_text in enumerate(prompt_tokens_text):
        tokenizer_text = token_text if word_index == 0 else f" {token_text}"
        token_ids = encode_text(tokenizer, tokenizer_text)
        if not token_ids:
            raise ValueError(f"Prompt token {token_text!r} encoded to no model tokens")
        start = cursor
        end = cursor + len(token_ids) - 1
        spans_without_bos.append((start, end))
        encoded_prompt_ids.extend(token_ids)
        cursor += len(token_ids)

    actual_ids = [int(token_id) for token_id in model_tokens[0].detach().cpu().tolist()]
    if actual_ids == encoded_prompt_ids:
        offset = 0
    elif actual_ids[1:] == encoded_prompt_ids:
        offset = 1
    else:
        raise ValueError(
            "Prompt metadata does not reconstruct the model tokenization. "
            "Regenerate prompts or use simpler one-token prompt words."
        )
    return [(start + offset, end + offset) for start, end in spans_without_bos]


def summarize_attention_heads(df: pd.DataFrame, top_k: int = 10) -> dict[str, Any]:
    if df.empty:
        return {
            "mean_attention_to_previous_occurrence": None,
            "median_attention_to_previous_occurrence": None,
            "top_heads_by_previous_occurrence_attention": [],
        }
    grouped = (
        df.groupby(["layer", "head"], as_index=False)
        .agg(
            mean_attention_to_previous_occurrence=(
                "attention_to_previous_occurrence",
                "mean",
            ),
            median_attention_to_previous_occurrence=(
                "attention_to_previous_occurrence",
                "median",
            ),
            mean_attention_entropy=("attention_entropy", "mean"),
            n_examples=("example_id", "nunique"),
        )
        .sort_values("mean_attention_to_previous_occurrence", ascending=False)
    )
    return {
        "mean_attention_to_previous_occurrence": float(
            df["attention_to_previous_occurrence"].mean()
        ),
        "median_attention_to_previous_occurrence": float(
            df["attention_to_previous_occurrence"].median()
        ),
        "top_heads_by_previous_occurrence_attention": grouped.head(top_k).to_dict(
            orient="records"
        ),
    }


def compute_attention_patterns(
    model: Any,
    records: list[PromptRecord],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    layers = selected_layers(config, n_layers(model))
    hook_names = [attention_pattern_hook_name(layer) for layer in layers]
    rows: list[dict[str, Any]] = []

    for record in tqdm(records, desc="Attention patterns"):
        require_induction_metadata(record)
        tokens = model.to_tokens(record.prompt)
        word_spans = prompt_word_token_spans(model.tokenizer, tokens, record.prompt_tokens_text)
        target_position = word_spans[-1][1]
        previous_position = word_spans[int(record.expected_source_position_hint)][1]
        if previous_position >= target_position:
            raise ValueError(
                f"Prompt {record.example_id} previous occurrence must be before target position"
            )

        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        for layer in layers:
            pattern = cache[attention_pattern_hook_name(layer)][0].detach().cpu()
            n_heads = int(pattern.shape[0])
            for head in range(n_heads):
                distribution = pattern[head, target_position, :].numpy()
                attended_source_position = int(np.argmax(distribution[: target_position + 1]))
                rows.append(
                    {
                        "example_id": record.example_id,
                        "layer": layer,
                        "head": head,
                        "seq_len": int(tokens.shape[1]),
                        "expected_next_token": record.expected_next_token,
                        "attended_source_position": attended_source_position,
                        "attended_source_token": decode_token(
                            model.tokenizer,
                            int(tokens[0, attended_source_position]),
                        ),
                        "target_position": target_position,
                        "target_token": decode_token(model.tokenizer, int(tokens[0, target_position])),
                        "previous_occurrence_position": previous_position,
                        "previous_occurrence_token": decode_token(
                            model.tokenizer,
                            int(tokens[0, previous_position]),
                        ),
                        "attention_to_previous_occurrence": float(
                            distribution[previous_position]
                        ),
                        "attention_to_bos": float(distribution[0]),
                        "attention_entropy": attention_entropy(distribution),
                    }
                )

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "attention_patterns_by_head.csv", index=False)

    aggregate = summarize_attention_heads(df)
    summary = {
        "model": config["model"]["name"],
        "n_examples": len(records),
        "layers": layers,
        "n_rows": len(df),
        **aggregate,
        "interpretation_note": "High scores are induction-like attention pattern candidates, not identified induction heads.",
        "limitation": ATTENTION_LIMITATION,
    }
    (output_dir / "attention_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_attention_induction_scores(df, figures_dir / "attention_induction_scores.png")
    return df, summary
