from __future__ import annotations

from pathlib import Path

import pandas as pd

from local_mi_lab.candidate_characterization_counterexamples import (
    build_counterexample_table,
    inspect_candidate_characterization_counterexamples,
)


def test_counterexample_table_includes_controls_and_failures() -> None:
    rows = pd.DataFrame(
        [
            _row("positive", 0.5, family="char_word_short"),
            _row("positive", -0.4, family="char_number_long", domain="number"),
            _row("control", 0.6, family="char_target_swap_control"),
        ]
    )

    table = build_counterexample_table(rows, top_k=2)

    assert "strongest_positive_failure" in set(table["counterexample_type"])
    assert "control_moved" in set(table["counterexample_type"])
    assert "wrong_target_control_moved" in set(table["counterexample_type"])


def test_inspection_writes_csv_and_markdown(tmp_path: Path) -> None:
    report = tmp_path / "report"
    seed = report / "seed20"
    seed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "candidate_id": "heldout_cand_000",
                "candidate_group": "random_comparison_replicated",
                "layer": 7,
                "head": 7,
                "mean_positive_minus_control_gap": -0.1,
                "final_characterization_status": "falsified_candidate",
            },
            {
                "candidate_id": "negative",
                "candidate_group": "negative_control_no_effect",
                "layer": 11,
                "head": 0,
                "mean_positive_minus_control_gap": 0.1,
                "final_characterization_status": "falsified_candidate",
            },
        ]
    ).to_csv(report / "candidate_characterization_multiseed_by_candidate.csv", index=False)
    pd.DataFrame(
        [
            _result_row("heldout_cand_000", "positive", -0.5),
            _result_row("heldout_cand_000", "control", 0.4, family="char_target_swap_control"),
        ]
    ).to_csv(seed / "candidate_characterization_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "example_id": "ex_positive",
                "token_domain": "word",
                "sequence_length_bucket": "short",
                "characterization_axis": "token_domain",
                "distractor_position_hint": "",
            },
            {
                "example_id": "ex_control",
                "token_domain": "word",
                "sequence_length_bucket": "short",
                "characterization_axis": "target_swap",
                "distractor_position_hint": "",
            },
        ]
    ).to_csv(seed / "prompts.csv", index=False)

    paths = inspect_candidate_characterization_counterexamples(
        candidate="L7H7",
        report_dir=report,
        output_dir=tmp_path / "out",
    )

    assert paths["csv"].exists()
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "What not to claim" in markdown
    assert "Do not claim induction-head discovery" in markdown


def test_missing_candidate_fails_clearly(tmp_path: Path) -> None:
    report = tmp_path / "report"
    report.mkdir()
    pd.DataFrame(
        [
            {
                "candidate_id": "heldout_cand_001",
                "candidate_group": "replicated_candidate",
                "layer": 9,
                "head": 11,
                "final_characterization_status": "falsified_candidate",
            }
        ]
    ).to_csv(report / "candidate_characterization_multiseed_by_candidate.csv", index=False)

    try:
        inspect_candidate_characterization_counterexamples(
            candidate="L7H7",
            report_dir=report,
            output_dir=tmp_path / "out",
        )
    except ValueError as exc:
        assert "L7H7" in str(exc)
    else:
        raise AssertionError("Expected missing candidate failure")


def _row(
    family_type: str,
    effect: float,
    *,
    family: str = "char_word_short",
    domain: str = "word",
) -> dict[str, object]:
    return {
        "seed": 20,
        "candidate_id": "c",
        "candidate_group": "replicated_candidate",
        "layer": 7,
        "head": 7,
        "family": family,
        "heldout_family_type": family_type,
        "example_id": f"ex_{family_type}_{effect}",
        "token_domain": domain,
        "sequence_length_bucket": "long",
        "intervention": "head_clean_to_corrupt_patch",
        "position_label": "final",
        "effect_size": effect,
        "effect_size_status": "ok",
        "true_expected_next_token": "D",
        "wrong_or_control_token": "X",
        "clean_prompt": "A B C A B",
        "corrupt_prompt": "A B X A B",
    }


def _result_row(candidate_id: str, family_type: str, effect: float, *, family: str = "char_word_short") -> dict[str, object]:
    row = _row(family_type, effect, family=family)
    row["candidate_id"] = candidate_id
    row["example_id"] = f"ex_{family_type}"
    return row
