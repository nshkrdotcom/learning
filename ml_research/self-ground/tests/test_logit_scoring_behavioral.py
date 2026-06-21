from __future__ import annotations

import pytest
import torch

from self_ground.behavioral_tasks import BehavioralTask
from self_ground.logit_scoring import (
    contrast_score_from_logits,
    score_behavioral_task_logits,
    token_group_logit_score,
    token_id_for_single_token_string,
)
from self_ground.task_validation import TokenValidationResult


class TinyTokenizerModel:
    def to_tokens(self, text: str, prepend_bos: bool = False):
        del prepend_bos
        mapping = {" good": [1], " bad": [2], " multi token": [3, 4]}
        return torch.tensor(mapping[text])


class TinyTokenizerAdapter:
    model = TinyTokenizerModel()


def test_single_token_validation_rejects_multi_token_strings() -> None:
    assert token_id_for_single_token_string(TinyTokenizerAdapter(), " good") == 1
    with pytest.raises(ValueError, match="exactly one token"):
        token_id_for_single_token_string(TinyTokenizerAdapter(), " multi token")


def test_contrast_score_math_mean_and_max() -> None:
    logits = torch.zeros((2, 1, 5))
    logits[:, -1, 1] = torch.tensor([2.0, 5.0])
    logits[:, -1, 2] = torch.tensor([1.0, 4.0])
    logits[:, -1, 3] = torch.tensor([3.0, 7.0])

    mean_score = contrast_score_from_logits(
        logits,
        target_token_ids=[1, 3],
        foil_token_ids=[2],
        reduction="mean",
    )
    max_score = contrast_score_from_logits(
        logits,
        target_token_ids=[1, 3],
        foil_token_ids=[2],
        reduction="max",
    )

    assert mean_score.tolist() == [1.5, 2.0]
    assert max_score.tolist() == [2.0, 3.0]


def test_token_group_rejects_empty_and_bad_shapes() -> None:
    with pytest.raises(ValueError, match="at least one"):
        token_group_logit_score(torch.zeros((1, 1, 4)), [])
    with pytest.raises(ValueError, match="2D or 3D"):
        token_group_logit_score(torch.zeros((4,)), [1])


def test_behavioral_task_scores_prompt_and_control() -> None:
    task = BehavioralTask(
        id="t",
        family="sentiment_negation",
        concept="movie",
        prompt="p",
        target_tokens=[" bad"],
        foil_tokens=[" good"],
        control_prompt="c",
        control_target_tokens=[" good"],
        control_foil_tokens=[" bad"],
        metadata={},
    )
    validation = TokenValidationResult(
        task_id="t",
        family="sentiment_negation",
        valid=True,
        prompt="p",
        control_prompt="c",
        control_type="matched_non_negation",
        target_token_ids=[2],
        foil_token_ids=[1],
        control_target_token_ids=[1],
        control_foil_token_ids=[2],
    )
    prompt_logits = torch.tensor([[[0.0, 1.0, 3.0]]])
    control_logits = torch.tensor([[[0.0, 4.0, 2.0]]])

    score = score_behavioral_task_logits(
        task=task,
        validation=validation,
        prompt_logits=prompt_logits,
        control_logits=control_logits,
    )

    assert score.prompt_result.contrast == 2.0
    assert score.control_result.contrast == 2.0
