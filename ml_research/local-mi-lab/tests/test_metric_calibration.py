from __future__ import annotations

import pandas as pd

from local_mi_lab.metric_calibration import (
    CONTROL_FAMILIES,
    POSITIVE_FAMILIES,
    aggregate_metric_calibration_by_family,
    build_calibration_prompt_bank,
    classify_metric_calibration_status,
    summarize_metric_calibration,
    validate_calibration_prompt_bank,
)


def test_calibration_prompt_bank_has_positive_and_control_families() -> None:
    rows = build_calibration_prompt_bank()
    families = set(rows["family"])

    assert POSITIVE_FAMILIES.issubset(families)
    assert CONTROL_FAMILIES.issubset(families)
    assert set(rows["token_domain"]) == {"symbolic", "word", "number"}
    assert set(rows["sequence_length_bucket"]) == {"short", "medium", "long"}


def test_validation_catches_missing_or_multitoken() -> None:
    rows = pd.DataFrame(
        [
            {
                "example_id": "bad",
                "family": "calib_clean_repeat_symbolic",
                "prompt": "A B C A B",
                "true_expected_next_token": " missing",
                "wrong_or_control_token": " A",
                "should_show_induction_behavior": True,
            }
        ]
    )
    validated = validate_calibration_prompt_bank(FakeTokenizer(), rows)

    assert validated.iloc[0]["validation_status"] == "tokenization_error"


def test_family_aggregation_and_status_pass() -> None:
    rows = pd.DataFrame(
        [
            _score_row("pos_symbolic", "calib_clean_repeat_symbolic", True, 1.2, "symbolic", "short"),
            _score_row("pos_word", "calib_clean_repeat_word", True, 1.0, "word", "medium"),
            _score_row("pos_number", "calib_clean_repeat_number", True, 0.9, "number", "long"),
            _score_row("control", "calib_target_swap_control", False, -0.3, "symbolic", "short"),
        ]
    )
    family = aggregate_metric_calibration_by_family(rows)

    assert classify_metric_calibration_status(rows, family) == "metric_calibrated_for_next_spec"


def test_status_fails_when_control_matches_positive() -> None:
    rows = pd.DataFrame(
        [
            _score_row("pos_symbolic", "calib_clean_repeat_symbolic", True, 0.6, "symbolic", "short"),
            _score_row("pos_word", "calib_clean_repeat_word", True, 0.6, "word", "medium"),
            _score_row("pos_number", "calib_clean_repeat_number", True, 0.6, "number", "long"),
            _score_row("control", "calib_target_swap_control", False, 0.55, "symbolic", "short"),
        ]
    )
    family = aggregate_metric_calibration_by_family(rows)
    summary = summarize_metric_calibration(rows, family)

    assert summary["status"] == "metric_needs_revision"
    assert summary["search_allowed"] is False


def test_status_marks_prompt_bank_revision_when_domain_flips() -> None:
    rows = pd.DataFrame(
        [
            _score_row("pos_symbolic", "calib_clean_repeat_symbolic", True, 0.8, "symbolic", "short"),
            _score_row("pos_word", "calib_clean_repeat_word", True, -0.1, "word", "medium"),
            _score_row("pos_number", "calib_clean_repeat_number", True, 0.8, "number", "long"),
            _score_row("control", "calib_target_swap_control", False, -0.5, "symbolic", "short"),
        ]
    )
    family = aggregate_metric_calibration_by_family(rows)

    assert classify_metric_calibration_status(rows, family) == "prompt_bank_needs_revision"


class FakeTokenizer:
    vocab = {" A": 0, " B": 1, " C": 2}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in self.vocab:
            return [self.vocab[text]]
        if text == " missing":
            return []
        return [0, 1]


def _score_row(
    example_id: str,
    family: str,
    should_show: bool,
    diff: float,
    domain: str,
    bucket: str,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "family": family,
        "token_domain": domain,
        "sequence_length_bucket": bucket,
        "should_show_induction_behavior": should_show,
        "validation_status": "ok",
        "metric_status": "ok",
        "true_vs_control_logit_diff": diff,
        "probability_gap": diff / 10,
        "target_rank": 1 if diff > 0 else 20,
    }
