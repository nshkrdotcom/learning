from __future__ import annotations

from pathlib import Path

import pandas as pd

from local_mi_lab.failure_taxonomy import (
    build_failure_taxonomy,
    classify_counterexample_rows,
    classify_single_counterexample,
    parse_summary_head_rows,
    summarize_failure_taxonomy_by_head,
    summary_has_negative_control_support,
)


def test_single_row_classifies_control_and_target_swap_leaks() -> None:
    row = _row(
        counterexample_type="wrong_target_control_moved",
        family="char_target_swap_control",
        heldout_family_type="control",
        effect=0.4,
    )

    modes = classify_single_counterexample(row)

    assert "control_moved" in modes
    assert "target_swap_leak" in modes


def test_single_row_classifies_reversed_control_leak() -> None:
    row = _row(
        counterexample_type="control_moved",
        family="char_reversed_control",
        heldout_family_type="control",
        effect=0.2,
    )

    modes = classify_single_counterexample(row)

    assert "control_moved" in modes
    assert "reversed_control_leak" in modes


def test_domain_and_length_flip_use_effect_sign_context() -> None:
    rows = pd.DataFrame(
        [
            _row(
                counterexample_type="token_domain_or_length_failure",
                token_domain="word",
                sequence_length_bucket="short",
                effect=-0.3,
            ),
            _row(
                counterexample_type="strongest_positive_success",
                token_domain="symbolic",
                sequence_length_bucket="long",
                effect=0.2,
            ),
        ]
    )
    classified = classify_counterexample_rows(rows)

    assert "domain_flip" in set(classified["failure_mode"])
    assert "length_flip" in set(classified["failure_mode"])


def test_position_and_intervention_mismatch_use_effect_sign_context() -> None:
    rows = pd.DataFrame(
        [
            _row(
                counterexample_type="position_intervention_mismatch",
                intervention="head_zero_ablation",
                position_label="final",
                effect=-0.2,
            ),
            _row(
                counterexample_type="strongest_positive_success",
                intervention="head_clean_to_corrupt_patch",
                position_label="previous_occurrence",
                effect=0.4,
            ),
        ]
    )
    classified = classify_counterexample_rows(rows)

    assert "intervention_disagreement" in set(classified["failure_mode"])
    assert "position_mismatch" in set(classified["failure_mode"])


def test_summary_parser_extracts_head_rows_and_negative_control_support() -> None:
    text = """
| head | candidate_id | group | final status | mean gap | mean corr | OV | QK |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L7H7 | heldout_cand_000 | random_comparison_replicated | falsified_candidate | -0.0550 | 0.0370 | ov_supports_copy | qk_weak |
| L4H1 | heldout_cand_012 | negative_control_no_effect | falsified_candidate | -0.0067 | -0.0640 | ov_supports_copy | qk_weak |
| L4H1 | negative_control_no_effect | -0.0640 | characterization_falsifies,characterization_supports |
"""
    rows = parse_summary_head_rows(text)

    assert rows["L7H7"].mean_attention_effect_corr == 0.037
    assert rows["L7H7"].ov_statuses == "ov_supports_copy"
    assert summary_has_negative_control_support(text)


def test_by_head_adds_attention_decoupled_ov_qk_and_negative_control_caution() -> None:
    taxonomy = pd.DataFrame(
        [
            {
                "head_label": "L7H7",
                "candidate_id": "heldout_cand_000",
                "candidate_group": "random_comparison_replicated",
                "failure_mode": "control_moved",
            }
        ]
    )
    summary_rows = parse_summary_head_rows(
        "| L7H7 | heldout_cand_000 | random_comparison_replicated | falsified_candidate | -0.0550 | 0.0370 | ov_supports_copy | qk_weak |\n"
    )

    by_head = summarize_failure_taxonomy_by_head(
        taxonomy,
        summary_rows=summary_rows,
        negative_control_support=True,
    )
    row = by_head.iloc[0]

    assert row["control_moved"] == 1
    assert row["attention_effect_decoupled"] == 1
    assert row["ov_qk_local_only"] == 1
    assert row["negative_control_support"] == 1


def test_build_failure_taxonomy_writes_outputs(tmp_path: Path) -> None:
    counterexamples = tmp_path / "counterexamples"
    counterexamples.mkdir()
    pd.DataFrame(
        [
            _row(
                counterexample_type="control_moved",
                family="char_reversed_control",
                heldout_family_type="control",
                effect=0.3,
            )
        ]
    ).to_csv(counterexamples / "counterexamples_L7H7.csv", index=False)
    summary_path = tmp_path / "summary.md"
    summary_path.write_text(
        "| L7H7 | heldout_cand_000 | random_comparison_replicated | falsified_candidate | -0.0550 | 0.0370 | ov_supports_copy | qk_weak |\n",
        encoding="utf-8",
    )

    paths = build_failure_taxonomy(
        counterexamples_dir=counterexamples,
        summary_path=summary_path,
        output_dir=tmp_path / "taxonomy",
        tracked_summary_path=tmp_path / "docs" / "taxonomy.md",
    )

    assert paths["by_row"].exists()
    assert paths["by_head"].exists()
    assert paths["summary"].exists()
    assert "No induction-head discovery" not in paths["tracked_summary"].read_text(
        encoding="utf-8"
    )
    assert "does not show an induction head" in paths["tracked_summary"].read_text(
        encoding="utf-8"
    )


def _row(
    *,
    counterexample_type: str,
    family: str = "char_word_short",
    heldout_family_type: str = "positive",
    token_domain: str = "word",
    sequence_length_bucket: str = "short",
    intervention: str = "head_clean_to_corrupt_patch",
    position_label: str = "final",
    effect: float,
) -> dict[str, object]:
    return {
        "counterexample_type": counterexample_type,
        "seed": 20,
        "candidate_id": "heldout_cand_000",
        "candidate_group": "random_comparison_replicated",
        "layer": 7,
        "head": 7,
        "head_label": "L7H7",
        "source_file": "counterexamples_L7H7.csv",
        "family": family,
        "heldout_family_type": heldout_family_type,
        "example_id": "ex",
        "token_domain": token_domain,
        "sequence_length_bucket": sequence_length_bucket,
        "intervention": intervention,
        "position_label": position_label,
        "effect_size": effect,
        "effect_size_status": "ok",
        "true_expected_next_token": " D",
        "wrong_or_control_token": " X",
        "clean_prompt": "A B C A B",
        "corrupt_prompt": "A B X A B",
    }
