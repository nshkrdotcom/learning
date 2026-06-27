from __future__ import annotations

import numpy as np
import pytest

from local_mi_lab.induction_metrics import (
    logit_diff_score,
    normalized_effect_size,
    paired_induction_score,
    probability_score,
    rank_score,
    resolve_induction_token_ids,
    target_logit_score,
)
from local_mi_lab.types import PromptRecord


class FakeTokenizer:
    vocab = {" A": 0, " B": 1, " C": 2, " D": 3, " X": 4}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in self.vocab:
            return [self.vocab[text]]
        return [self.vocab[f" {part}"] for part in text.strip().split() if f" {part}" in self.vocab]


def test_target_logit_score_from_synthetic_logits() -> None:
    logits = np.array([[0.1, 0.2, 3.0]])
    assert target_logit_score(logits, target_token_id=2) == 3.0


def test_logit_diff_score_from_synthetic_logits() -> None:
    logits = np.array([[[0.1, 2.0, -1.0]]])
    assert logit_diff_score(logits, positive_token_id=1, negative_token_id=2) == 3.0


def test_rank_score_from_synthetic_logits() -> None:
    logits = np.array([0.1, 4.0, 2.0, 3.0])
    assert rank_score(logits, target_token_id=2) == 3


def test_probability_score_from_synthetic_logits() -> None:
    logits = np.array([0.0, 0.0])
    assert probability_score(logits, target_token_id=1) == pytest.approx(0.5)


def test_normalized_effect_size_with_normal_denominator() -> None:
    result = normalized_effect_size(clean_score=5.0, corrupt_score=1.0, patched_score=3.0)
    assert result == {"effect_size": 0.5, "effect_size_status": "ok"}


def test_normalized_effect_size_with_zero_denominator() -> None:
    result = normalized_effect_size(clean_score=2.0, corrupt_score=2.0, patched_score=3.0)
    assert result == {"effect_size": None, "effect_size_status": "denominator_zero"}


def test_paired_induction_score_separates_positive_and_control() -> None:
    assert paired_induction_score(
        positive_true_score=6.0,
        positive_control_score=1.0,
        family_true_score=3.0,
        family_control_score=2.0,
    ) == 4.0


def test_token_resolution_validates_single_tokens() -> None:
    record = PromptRecord(
        example_id="example",
        task="induction_controls",
        family="positive_repeat_sequence",
        prompt="A B C D A B C",
        expected_next_token=" D",
        control_prompt="A B C D X Y Z",
        notes="",
        true_expected_next_token=" D",
        wrong_or_control_token=" X",
    )
    ids = resolve_induction_token_ids(FakeTokenizer(), record)
    assert ids["true_token_id"] == 3
    assert ids["wrong_or_control_token_id"] == 4


def test_token_resolution_fails_for_multi_token_control() -> None:
    record = PromptRecord(
        example_id="example",
        task="induction_controls",
        family="positive_repeat_sequence",
        prompt="A B C D A B C",
        expected_next_token=" D",
        control_prompt="A B C D X Y Z",
        notes="",
        true_expected_next_token=" D",
        wrong_or_control_token=" A B",
    )
    with pytest.raises(ValueError, match="one token"):
        resolve_induction_token_ids(FakeTokenizer(), record)
