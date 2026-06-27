from __future__ import annotations

import pandas as pd

from local_mi_lab.position_characterization import (
    aggregate_position_by_candidate,
    classify_position_specificity,
)


def test_destination_specific_classification() -> None:
    assert (
        classify_position_specificity(
            final_effect=0.3,
            previous_occurrence_effect=-0.1,
            source_position_effect=None,
            distractor_position_effect=None,
        )
        == "destination_specific"
    )


def test_source_specific_classification() -> None:
    assert (
        classify_position_specificity(
            final_effect=-0.1,
            previous_occurrence_effect=0.2,
            source_position_effect=None,
            distractor_position_effect=None,
        )
        == "source_specific"
    )


def test_distractor_like_classification() -> None:
    assert (
        classify_position_specificity(
            final_effect=0.1,
            previous_occurrence_effect=0.2,
            source_position_effect=None,
            distractor_position_effect=0.3,
        )
        == "distractor_like"
    )


def test_position_nonspecific_classification() -> None:
    assert (
        classify_position_specificity(
            final_effect=0.2,
            previous_occurrence_effect=0.18,
            source_position_effect=None,
            distractor_position_effect=None,
        )
        == "position_nonspecific"
    )


def test_aggregate_position_matrix() -> None:
    results = pd.DataFrame(
        [
            _row("final", 0.3),
            _row("previous_occurrence", -0.1),
            _row("source_position", -0.1),
            _row("distractor_position", None, status="unavailable_for_family"),
        ]
    )

    summary = aggregate_position_by_candidate(results)

    assert summary.loc[0, "final_effect"] == 0.3
    assert summary.loc[0, "position_specificity_status"] == "destination_specific"


def test_unavailable_position_is_explicit() -> None:
    results = pd.DataFrame([_row("distractor_position", None, status="unavailable_for_family")])

    summary = aggregate_position_by_candidate(results)

    assert summary.loc[0, "position_specificity_status"] == "insufficient_positions"


def _row(position: str, effect: float | None, status: str = "ok") -> dict[str, object]:
    return {
        "seed": 10,
        "candidate_id": "heldout_cand_000",
        "candidate_group": "replicated_candidate",
        "layer": 7,
        "head": 7,
        "family": "heldout_symbolic_longer",
        "heldout_family_type": "positive",
        "example_id": "example_0000",
        "intervention": "head_clean_to_corrupt_patch",
        "position_label": position,
        "position_status": status,
        "head_specific_patch": True,
        "actual_patch_scope": "single_head_z",
        "metric": "true_vs_control_logit_diff",
        "effect_size": effect,
        "effect_size_status": "ok" if effect is not None else "position_unavailable",
        "clean_prompt": "A B C D A B C",
        "corrupt_prompt": "A C B D C A B",
    }
