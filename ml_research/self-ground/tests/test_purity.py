from __future__ import annotations

from self_ground.purity import score_control_purity


def test_intended_minimal_pair_scores_higher_than_topic_drift() -> None:
    clean = score_control_purity(
        "The dog is not friendly.",
        "The dog is friendly.",
        x_para="The dog isn't friendly.",
        x_decoy="The dog is often friendly.",
    )
    drifted = score_control_purity(
        "The dog is not friendly.",
        "A telescope was repaired yesterday.",
        x_para="The dog isn't friendly.",
        x_decoy="The dog is often friendly.",
    )

    assert clean > 0.70
    assert clean > drifted + 0.25


def test_paraphrase_purity_is_accepted_without_requiring_identity() -> None:
    pair_score = score_control_purity(
        "There is no leak in the pipe.",
        "There is a leak in the pipe.",
        x_para="There isn't a leak in the pipe.",
        x_decoy="There is sometimes a leak in the pipe.",
    )
    direct_score = score_control_purity(
        "There is no leak in the pipe.",
        "There is a leak in the pipe.",
    )

    assert pair_score > 0.60
    assert direct_score > 0.60
    assert pair_score != direct_score


def test_empty_strings_are_safe_and_stable() -> None:
    first = score_control_purity("", "", x_para="", x_decoy="")
    second = score_control_purity("", "", x_para="", x_decoy="")

    assert first == second
    assert 0.0 <= first <= 1.0
