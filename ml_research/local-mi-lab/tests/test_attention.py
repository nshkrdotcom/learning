from __future__ import annotations

import pandas as pd
import pytest
import torch

from local_mi_lab.attention import (
    attention_entropy,
    prompt_word_token_spans,
    summarize_attention_heads,
)
from local_mi_lab.prompts import generate_induction_prompts


def test_induction_metadata_maps_to_expected_source_positions() -> None:
    record = generate_induction_prompts(n_examples=1, seed=0)[0]
    assert record.expected_source_position_hint is not None
    source_position = record.expected_source_position_hint
    assert record.prompt_tokens_text[source_position] == record.expected_source_token
    assert source_position < len(record.prompt_tokens_text) - 1
    assert record.prompt_tokens_text[-1] == record.expected_source_token


def test_attention_entropy_math_on_synthetic_distribution() -> None:
    assert attention_entropy([0.5, 0.5, 0.0]) == pytest.approx(0.69314718056)
    assert attention_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_prompt_word_token_spans_allow_split_model_tokens() -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return {
                "north": [10, 11],
                " south": [12],
            }[text]

    spans = prompt_word_token_spans(
        FakeTokenizer(),
        torch.tensor([[0, 10, 11, 12]]),
        ["north", "south"],
    )
    assert spans == [(1, 2), (3, 3)]


def test_top_head_summary_ranks_previous_occurrence_attention_first() -> None:
    df = pd.DataFrame(
        [
            {
                "example_id": "a",
                "layer": 0,
                "head": 0,
                "attention_to_previous_occurrence": 0.1,
                "attention_entropy": 1.0,
            },
            {
                "example_id": "b",
                "layer": 0,
                "head": 0,
                "attention_to_previous_occurrence": 0.2,
                "attention_entropy": 1.0,
            },
            {
                "example_id": "a",
                "layer": 1,
                "head": 2,
                "attention_to_previous_occurrence": 0.9,
                "attention_entropy": 0.5,
            },
            {
                "example_id": "b",
                "layer": 1,
                "head": 2,
                "attention_to_previous_occurrence": 0.7,
                "attention_entropy": 0.6,
            },
        ]
    )
    summary = summarize_attention_heads(df)
    top = summary["top_heads_by_previous_occurrence_attention"][0]
    assert top["layer"] == 1
    assert top["head"] == 2
    assert top["mean_attention_to_previous_occurrence"] == pytest.approx(0.8)
