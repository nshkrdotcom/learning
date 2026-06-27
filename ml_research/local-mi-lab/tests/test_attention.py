from __future__ import annotations

import pandas as pd
import pytest
import torch

from local_mi_lab.attention import (
    attention_by_family,
    attention_entropy,
    previous_occurrence_attention,
    prompt_word_token_spans,
    summarize_attention_controls,
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


def test_controls_without_source_positions_do_not_fake_previous_occurrence_attention() -> None:
    assert previous_occurrence_attention([0.2, 0.8], None) is None


def test_attention_by_family_aggregation() -> None:
    df = _attention_df()
    by_family = attention_by_family(df)
    positive = by_family[
        (by_family["family"] == "positive_repeat_sequence")
        & (by_family["layer"] == 0)
        & (by_family["head"] == 0)
    ].iloc[0]
    assert positive["mean_attention_to_previous_occurrence"] == pytest.approx(0.8)


def test_positive_minus_control_attention_gap_ranks_specific_heads() -> None:
    summary = summarize_attention_controls(_attention_df(), top_k=5)
    top_gap = summary["top_heads_by_positive_minus_control_gap"][0]
    assert top_gap["layer"] == 0
    assert top_gap["head"] == 0
    assert top_gap["positive_minus_control_attention_gap"] == pytest.approx(0.6)


def test_attention_summary_distinguishes_raw_and_control_firing_heads() -> None:
    summary = summarize_attention_controls(_attention_df(), top_k=5)
    raw_top = summary["top_heads_on_positive_examples"][0]
    control_top = summary["top_heads_on_controls"][0]
    assert raw_top["mean_attention_to_previous_occurrence"] == pytest.approx(0.9)
    assert control_top["mean_attention_to_previous_occurrence"] == pytest.approx(0.7)
    assert summary["hardest_control_family_by_attention"]["family"] == (
        "same_token_frequency_control"
    )


def _attention_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "example_id": "p1",
                "family": "positive_repeat_sequence",
                "should_show_induction_behavior": True,
                "layer": 0,
                "head": 0,
                "attention_to_previous_occurrence": 0.8,
                "attention_to_bos": 0.1,
                "attention_entropy": 0.5,
            },
            {
                "example_id": "c1",
                "family": "shuffled_repeat_control",
                "should_show_induction_behavior": False,
                "layer": 0,
                "head": 0,
                "attention_to_previous_occurrence": 0.2,
                "attention_to_bos": 0.1,
                "attention_entropy": 0.7,
            },
            {
                "example_id": "p2",
                "family": "positive_repeat_sequence",
                "should_show_induction_behavior": True,
                "layer": 1,
                "head": 1,
                "attention_to_previous_occurrence": 0.9,
                "attention_to_bos": 0.1,
                "attention_entropy": 0.5,
            },
            {
                "example_id": "c2",
                "family": "same_token_frequency_control",
                "should_show_induction_behavior": False,
                "layer": 1,
                "head": 1,
                "attention_to_previous_occurrence": 0.7,
                "attention_to_bos": 0.1,
                "attention_entropy": 0.7,
            },
            {
                "example_id": "c3",
                "family": "no_repeat_control",
                "should_show_induction_behavior": False,
                "layer": 1,
                "head": 1,
                "attention_to_previous_occurrence": None,
                "attention_to_bos": 0.9,
                "attention_entropy": 0.2,
            },
        ]
    )
