from __future__ import annotations

import torch

from local_mi_lab.head_circuit_diagnostics import (
    _default_diagnostic_families,
    classify_ov_status,
    classify_qk_status,
    ov_margin_from_logits,
    qk_margin_from_scores,
)


def test_ov_margin_calculation() -> None:
    logits = torch.tensor([0.1, 2.0, 0.4, 1.0])

    result = ov_margin_from_logits(logits, true_token_id=1, wrong_token_id=3)

    assert result["ov_true_token_score"] == 2.0
    assert result["ov_wrong_token_score"] == 1.0
    assert result["ov_copy_margin"] == 1.0
    assert result["ov_rank_of_true_token"] == 1


def test_qk_margin_calculation() -> None:
    scores = torch.tensor([0.0, 0.2, 0.8, 0.4])

    result = qk_margin_from_scores(
        scores,
        expected_source_position=2,
        distractor_positions=[1, 3],
    )

    source_score = result["mean_qk_expected_source_score"]
    distractor_score = result["mean_qk_best_distractor_score"]
    assert source_score is not None
    assert distractor_score is not None
    assert abs(source_score - 0.8) < 1e-6
    assert abs(distractor_score - 0.4) < 1e-6
    assert abs(result["qk_source_margin"] - 0.4) < 1e-6  # type: ignore[operator]


def test_ov_statuses() -> None:
    assert classify_ov_status(0.1) == "ov_supports_copy"
    assert classify_ov_status(-0.2) == "ov_contradicts_copy"
    assert classify_ov_status(-0.01) == "ov_weak"
    assert classify_ov_status(None) == "ov_unavailable"


def test_qk_statuses() -> None:
    assert classify_qk_status(0.1) == "qk_supports_source_selection"
    assert classify_qk_status(-0.2) == "qk_contradicts_source_selection"
    assert classify_qk_status(-0.01) == "qk_weak"
    assert classify_qk_status(None) == "qk_unavailable"


def test_qk_missing_source_is_unavailable_shape() -> None:
    result = qk_margin_from_scores(
        torch.tensor([0.1, 0.2]),
        expected_source_position=4,
        distractor_positions=[1],
    )

    assert result["qk_source_margin"] is None
    assert classify_qk_status(result["qk_source_margin"]) == "qk_unavailable"


def test_candidate_characterization_default_diagnostic_families_are_positive() -> None:
    config = {
        "task": {
            "name": "candidate_characterization",
            "families": [
                "char_symbolic_short",
                "char_word_long",
                "char_reversed_control",
                "char_target_swap_control",
            ],
        }
    }

    assert _default_diagnostic_families(config) == [
        "char_symbolic_short",
        "char_word_long",
    ]
