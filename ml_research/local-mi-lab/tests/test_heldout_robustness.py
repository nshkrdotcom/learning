from __future__ import annotations

import pandas as pd

from local_mi_lab.heldout_prompts import generate_heldout_induction_prompts
from local_mi_lab.heldout_robustness import (
    RESULT_COLUMNS,
    aggregate_heldout_by_candidate,
    aggregate_heldout_by_family,
    classify_heldout_seed_status,
    expand_heldout_jobs,
)


def test_matrix_expansion_uses_fixed_candidates() -> None:
    records = generate_heldout_induction_prompts(n_examples_per_family=2, seed=10)
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "c0",
                "candidate_group": "replicated_candidate",
                "layer": 7,
                "head": 7,
            },
            {
                "candidate_id": "c1",
                "candidate_group": "negative_control_no_effect",
                "layer": 0,
                "head": 1,
            },
        ]
    )
    jobs = expand_heldout_jobs(
        records,
        candidates,
        families=["heldout_symbolic_longer", "heldout_wrong_target_same_prompt"],
        interventions=["head_clean_to_corrupt_patch", "head_zero_ablation"],
        positions=["final", "previous_occurrence"],
        examples_per_family=1,
        seed=10,
    )
    assert len(jobs) == 2 * 2 * 2 * 2
    assert {(job["layer"], job["head"]) for job in jobs} == {(7, 7), (0, 1)}


def test_survival_status_classification() -> None:
    assert (
        classify_heldout_seed_status(
            head_specific=True,
            positive_mean=0.3,
            max_control=0.1,
            gap=0.2,
            n_positive_families_with_gap_gt_0=2,
        )
        == "heldout_survives_seed"
    )
    assert (
        classify_heldout_seed_status(
            head_specific=True,
            positive_mean=0.3,
            max_control=0.35,
            gap=-0.05,
            n_positive_families_with_gap_gt_0=3,
        )
        == "falsified_controls_move"
    )
    assert (
        classify_heldout_seed_status(
            head_specific=True,
            positive_mean=0.3,
            max_control=0.1,
            gap=0.2,
            n_positive_families_with_gap_gt_0=1,
        )
        == "downgraded_weak_family_specific"
    )


def test_result_schema_contains_intervention_and_position_fields() -> None:
    assert "intervention" in RESULT_COLUMNS
    assert "intervention_status" in RESULT_COLUMNS
    assert "position_label" in RESULT_COLUMNS
    assert "position_status" in RESULT_COLUMNS


def test_controls_moving_falsification_dominates_candidate_summary() -> None:
    results = pd.DataFrame(
        [
            _row("p0", "heldout_symbolic_longer", "positive", 0.2),
            _row("p1", "heldout_word_sequences", "positive", 0.2),
            _row("c0", "heldout_wrong_target_same_prompt", "control", 0.4),
            _row("c1", "heldout_no_structure_same_tokens", "control", 0.1),
        ]
    )
    by_family = aggregate_heldout_by_family(results)
    by_candidate = aggregate_heldout_by_candidate(by_family)
    assert by_candidate.iloc[0]["survival_status"] == "falsified_controls_move"


def test_position_unavailable_is_counted_not_hidden() -> None:
    results = pd.DataFrame(
        [
            {
                **_row("c0", "heldout_no_structure_same_tokens", "control", None),
                "position_status": "unavailable_for_family",
                "effect_size_status": "position_unavailable",
            }
        ]
    )
    by_family = aggregate_heldout_by_family(results)
    assert by_family.iloc[0]["n_position_unavailable"] == 1


def _row(
    example_id: str,
    family: str,
    family_type: str,
    effect_size: float | None,
) -> dict[str, object]:
    return {
        "seed": 10,
        "candidate_id": "c0",
        "candidate_group": "replicated_candidate",
        "layer": 7,
        "head": 7,
        "family": family,
        "example_id": example_id,
        "heldout_family_type": family_type,
        "intervention": "head_clean_to_corrupt_patch",
        "intervention_status": "ok",
        "position_label": "final",
        "position_status": "ok",
        "head_specific_patch": True,
        "actual_patch_scope": "single_head_z",
        "metric": "true_vs_control_logit_diff",
        "clean_score": 1.0,
        "corrupt_score": 0.0,
        "patched_score": effect_size,
        "effect_size": effect_size,
        "effect_size_status": "ok" if effect_size is not None else "position_unavailable",
        "true_expected_next_token": " F",
        "wrong_or_control_token": " X",
        "clean_prompt": "A B C D E F A B C D E",
        "corrupt_prompt": "A C E B D F C A E B D",
    }
