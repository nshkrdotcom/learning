from __future__ import annotations

import pandas as pd

from local_mi_lab.attention_effect_alignment import (
    classify_alignment,
    source_attention_margin,
    summarize_attention_effect_by_candidate,
)


def test_source_attention_margin() -> None:
    margin = source_attention_margin(0.7, 0.2)
    assert margin is not None
    assert abs(margin - 0.5) < 1e-12
    assert source_attention_margin(None, 0.2) is None


def test_positive_alignment_specific_when_controls_weaker() -> None:
    examples = pd.DataFrame(
        [
            _row("positive", 0.1, 0.4, 0.1),
            _row("positive", 0.2, 0.5, 0.1),
            _row("positive", 0.3, 0.6, 0.1),
            _row("positive", 0.4, 0.7, 0.1),
            _row("control", 0.1, 0.2, 0.1),
            _row("control", 0.1, 0.2, 0.1),
        ]
    )

    summary = summarize_attention_effect_by_candidate(examples)

    assert summary.loc[0, "alignment_status"] == "aligned_positive_specific"
    assert summary.loc[0, "positive_family_corr"] > 0


def test_controls_matching_positives_are_control_like() -> None:
    examples = pd.DataFrame(
        [
            _row("positive", 0.1, 0.4, 0.1),
            _row("positive", 0.2, 0.5, 0.1),
            _row("positive", 0.3, 0.6, 0.1),
            _row("control", 0.1, 0.4, 0.1),
            _row("control", 0.2, 0.5, 0.1),
            _row("control", 0.3, 0.6, 0.1),
        ]
    )

    summary = summarize_attention_effect_by_candidate(examples)

    assert summary.loc[0, "alignment_status"] == "aligned_but_control_like"


def test_negative_correlation_is_not_aligned() -> None:
    examples = pd.DataFrame(
        [
            _row("positive", 0.4, 0.4, 0.1),
            _row("positive", 0.3, 0.5, 0.1),
            _row("positive", 0.2, 0.6, 0.1),
            _row("positive", 0.1, 0.7, 0.1),
        ]
    )

    summary = summarize_attention_effect_by_candidate(examples)

    assert summary.loc[0, "alignment_status"] == "not_aligned"


def test_insufficient_examples_is_explicit() -> None:
    status = classify_alignment(
        positive_corr=0.5,
        control_corr=None,
        positive_source_margin=0.2,
        control_source_margin=None,
        n_valid_examples=2,
    )

    assert status == "insufficient_examples"


def test_schema_retains_prompts_and_candidate_groups() -> None:
    examples = pd.DataFrame([_row("positive", 0.1, 0.4, 0.1)])

    assert examples.loc[0, "candidate_group"] == "replicated_candidate"
    assert examples.loc[0, "clean_prompt"]
    assert examples.loc[0, "corrupt_prompt"]


def _row(
    family_type: str,
    effect: float,
    source_attention: float,
    distractor_attention: float,
) -> dict[str, object]:
    return {
        "seed": 10,
        "candidate_id": "heldout_cand_001",
        "candidate_group": "replicated_candidate",
        "layer": 7,
        "head": 7,
        "family": "heldout_symbolic_longer"
        if family_type == "positive"
        else "heldout_no_structure_same_tokens",
        "heldout_family_type": family_type,
        "example_id": f"{family_type}_{effect}",
        "intervention": "head_clean_to_corrupt_patch",
        "position_label": "final",
        "effect_size": effect,
        "effect_size_status": "ok",
        "attention_to_expected_source": source_attention,
        "attention_to_best_distractor": distractor_attention,
        "source_attention_margin": source_attention - distractor_attention,
        "target_token": " D",
        "source_token": "C",
        "clean_prompt": "A B C D A B C",
        "corrupt_prompt": "A C B D C A B",
    }
