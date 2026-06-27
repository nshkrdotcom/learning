from __future__ import annotations

import pandas as pd

from local_mi_lab.controlled_patching import (
    aggregate_controlled_patching_by_candidate,
    aggregate_controlled_patching_by_family,
    build_patching_jobs,
    classify_candidate_specificity,
    controlled_patching_summary,
)
from local_mi_lab.prompts import generate_induction_control_prompts


def test_candidate_example_pairing_uses_explicit_metadata() -> None:
    records = generate_induction_control_prompts(n_examples_per_family=2, seed=0)
    candidates = [_candidate("cand_000")]
    jobs = build_patching_jobs(
        records,
        candidates,
        families=["distractor_repeat_control"],
        examples_per_family=1,
        max_candidates=1,
        seed=0,
    )
    job = jobs[0]
    assert job["clean_prompt"] != job["corrupt_prompt"]
    assert job["record"].paired_positive_example_id.startswith("positive_repeat_sequence_")
    assert job["positive_minus_control_context"] == "positive_vs_distractor_repeat_control"


def test_random_expected_pairing_tests_wrong_target_specificity() -> None:
    records = generate_induction_control_prompts(n_examples_per_family=1, seed=0)
    jobs = build_patching_jobs(
        records,
        [_candidate("cand_000")],
        families=["random_expected_token_control"],
        examples_per_family=1,
        max_candidates=1,
    )
    job = jobs[0]
    assert job["clean_prompt"] == job["corrupt_prompt"]
    assert job["target_token"] == " X"
    assert job["wrong_or_control_token"] == " X"


def test_effect_size_aggregation_by_family() -> None:
    results = _results_df()
    by_family = aggregate_controlled_patching_by_family(results)
    positive = by_family[by_family["family"] == "positive_repeat_sequence"].iloc[0]
    assert positive["mean_effect_size"] == 0.6
    assert positive["fraction_positive_effect"] == 1.0


def test_positive_minus_control_effect_gap() -> None:
    by_candidate = aggregate_controlled_patching_by_candidate(_results_df(), [_candidate("cand_000")])
    row = by_candidate.iloc[0]
    assert row["positive_mean_effect_size"] == 0.6
    assert row["max_control_mean_effect_size"] == 0.3
    assert row["positive_minus_control_effect_gap"] == 0.3


def test_nonspecific_candidate_when_controls_move_as_much() -> None:
    status = classify_candidate_specificity(_results_df(control_effect=0.7), 0.6, 0.7, -0.1)
    assert status == "nonspecific_moves_controls"


def test_positive_specific_candidate_when_positive_moves_more() -> None:
    status = classify_candidate_specificity(_results_df(), 0.6, 0.3, 0.3)
    assert status == "positive_specific_candidate"


def test_denominator_zero_handling() -> None:
    df = _results_df()
    df["effect_size_status"] = "denominator_zero"
    df["effect_size"] = None
    status = classify_candidate_specificity(df, None, None, None)
    assert status == "denominator_problem"


def test_candidate_summary_schema() -> None:
    by_family = aggregate_controlled_patching_by_family(_results_df())
    by_candidate = aggregate_controlled_patching_by_candidate(_results_df(), [_candidate("cand_000")])
    summary = controlled_patching_summary(
        _results_df(),
        by_family,
        by_candidate,
        ["positive_repeat_sequence", "distractor_repeat_control"],
    )
    assert summary["n_candidates"] == 1
    assert "specificity_status_counts" in summary
    assert summary["head_specific_patch"] is False
    assert summary["actual_patch_scope"] == "full_attn_out_layer"
    assert summary["actual_patch_scopes"] == ["full_attn_out_layer"]


def test_dry_run_job_selection_does_not_require_model() -> None:
    records = generate_induction_control_prompts(n_examples_per_family=2, seed=0)
    jobs = build_patching_jobs(
        records,
        [_candidate("cand_000"), _candidate("cand_001", layer=1, head=2)],
        families=["positive_repeat_sequence"],
        examples_per_family=2,
        max_candidates=1,
        seed=0,
    )
    assert len(jobs) == 2


def _candidate(candidate_id: str, layer: int = 0, head: int = 1) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source": "top_raw_positive_attention",
        "layer": layer,
        "head": head,
        "component": "attn_out",
    }


def _results_df(control_effect: float = 0.3) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("p0", "positive_repeat_sequence", True, 0.6),
            _row("p1", "positive_repeat_sequence", True, 0.6),
            _row("c0", "distractor_repeat_control", False, control_effect),
            _row("c1", "distractor_repeat_control", False, control_effect),
        ]
    )


def _row(
    example_id: str,
    family: str,
    should_show: bool,
    effect_size: float | None,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "family": family,
        "control_family": "" if should_show else family,
        "should_show_induction_behavior": should_show,
        "candidate_id": "cand_000",
        "candidate_source": "top_raw_positive_attention",
        "layer": 0,
        "head": 1,
        "component": "attn_out",
        "position_label": "final",
        "patch_position": 6,
        "head_specific_patch": False,
        "actual_patch_scope": "full_attn_out_layer",
        "clean_prompt": "A B C D A B C",
        "corrupt_prompt": "A B C D A B X",
        "target_token": " D",
        "true_expected_next_token": " D",
        "wrong_or_control_token": "",
        "metric": "target_logit",
        "clean_score": 3.0,
        "corrupt_score": 1.0,
        "patched_score": 2.2,
        "effect_size": effect_size,
        "effect_size_status": "ok" if effect_size is not None else "denominator_zero",
        "positive_minus_control_context": "test",
    }
