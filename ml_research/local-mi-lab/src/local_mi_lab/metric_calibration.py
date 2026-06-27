from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import torch

from local_mi_lab.induction_metrics import (
    logit_diff_score,
    probability_score,
    rank_score,
    target_logit_score,
)
from local_mi_lab.tokens import token_id_for_single_token

POSITIVE_FAMILIES = {
    "calib_clean_repeat_symbolic",
    "calib_clean_repeat_word",
    "calib_clean_repeat_number",
    "calib_clean_repeat_format_variant",
}

CONTROL_FAMILIES = {
    "calib_wrong_target_same_prompt",
    "calib_target_swap_control",
    "calib_same_token_frequency_control",
    "calib_reversed_order_control",
    "calib_no_repeat_control",
    "calib_frequency_trap_control",
}

SUMMARY_THRESHOLDS = {
    "min_positive_minus_max_control_gap": 0.25,
    "min_positive_fraction_diff_positive": 0.80,
    "max_control_fraction_diff_positive": 0.20,
}


def build_calibration_prompt_bank(seed: int = 0) -> pd.DataFrame:
    del seed
    rows: list[dict[str, Any]] = []
    for base in _base_sequences():
        rows.extend(_rows_for_base_sequence(base))
    return pd.DataFrame(rows)


def validate_calibration_prompt_bank(tokenizer: Any, rows: pd.DataFrame) -> pd.DataFrame:
    validated: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        status = "ok"
        error = ""
        true_id: int | None = None
        control_id: int | None = None
        try:
            true_id = token_id_for_single_token(tokenizer, str(row["true_expected_next_token"]))
            control_id = token_id_for_single_token(tokenizer, str(row["wrong_or_control_token"]))
            if true_id == control_id:
                raise ValueError("expected and control tokens resolve to the same token id")
            encoded_prompt = tokenizer.encode(str(row["prompt"]), add_special_tokens=False)
            if len(encoded_prompt) == 0:
                raise ValueError("prompt tokenized to an empty sequence")
        except Exception as exc:  # noqa: BLE001 - validation artifacts need exact failure text.
            status = "tokenization_error"
            error = str(exc)
        updated = dict(row)
        updated.update(
            {
                "true_token_id": true_id,
                "control_token_id": control_id,
                "validation_status": status,
                "validation_error": error,
            }
        )
        validated.append(updated)
    return pd.DataFrame(validated)


def score_metric_calibration_examples(model: Any, tokenizer: Any, rows: pd.DataFrame) -> pd.DataFrame:
    validated = validate_calibration_prompt_bank(tokenizer, rows)
    scored: list[dict[str, Any]] = []
    for row in validated.to_dict("records"):
        if row["validation_status"] != "ok":
            scored.append(_unscored_row(row))
            continue
        tokens = model.to_tokens(row["prompt"])
        with torch.inference_mode():
            logits = model(tokens)[0, -1, :]
        true_id = int(row["true_token_id"])
        control_id = int(row["control_token_id"])
        true_probability = probability_score(logits, true_id)
        control_probability = probability_score(logits, control_id)
        scored.append(
            {
                **row,
                "metric_status": "ok",
                "target_logit": target_logit_score(logits, true_id),
                "control_logit": target_logit_score(logits, control_id),
                "true_vs_control_logit_diff": logit_diff_score(logits, true_id, control_id),
                "target_probability": true_probability,
                "control_probability": control_probability,
                "probability_gap": true_probability - control_probability,
                "target_rank": rank_score(logits, true_id),
                "control_rank": rank_score(logits, control_id),
            }
        )
    return pd.DataFrame(scored)


def aggregate_metric_calibration_by_family(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=_family_columns())
    valid = rows[rows["metric_status"] == "ok"].copy()
    if valid.empty:
        return pd.DataFrame(columns=_family_columns())
    summaries: list[dict[str, Any]] = []
    for family, group in valid.groupby("family", sort=True):
        diffs = pd.to_numeric(group["true_vs_control_logit_diff"], errors="coerce")
        probs = pd.to_numeric(group["probability_gap"], errors="coerce")
        ranks = pd.to_numeric(group["target_rank"], errors="coerce")
        should_show = bool(group["should_show_induction_behavior"].iloc[0])
        summaries.append(
            {
                "family": family,
                "n_examples": int(len(group)),
                "token_domains": ",".join(sorted(group["token_domain"].dropna().astype(str).unique())),
                "sequence_length_buckets": ",".join(
                    sorted(group["sequence_length_bucket"].dropna().astype(str).unique())
                ),
                "should_show_induction_behavior": should_show,
                "mean_true_vs_control_logit_diff": _mean_or_none(diffs),
                "median_true_vs_control_logit_diff": _median_or_none(diffs),
                "fraction_diff_positive": float((diffs > 0).mean()),
                "mean_probability_gap": _mean_or_none(probs),
                "median_target_rank": _median_or_none(ranks),
            }
        )
    return pd.DataFrame(summaries, columns=_family_columns())


def summarize_metric_calibration(rows: pd.DataFrame, family_rows: pd.DataFrame) -> dict[str, Any]:
    validation_counts = Counter(rows.get("validation_status", pd.Series(dtype=str)).astype(str))
    metric_counts = Counter(rows.get("metric_status", pd.Series(dtype=str)).astype(str))
    if validation_counts.get("tokenization_error", 0) > 0:
        status = "blocked_tokenization"
    elif family_rows.empty:
        status = "metric_needs_revision"
    else:
        status = classify_metric_calibration_status(rows, family_rows)

    positives = family_rows[family_rows["should_show_induction_behavior"] == True]  # noqa: E712
    controls = family_rows[family_rows["should_show_induction_behavior"] == False]  # noqa: E712
    positive_mean = _mean_or_none(positives["mean_true_vs_control_logit_diff"]) if not positives.empty else None
    max_control = (
        float(controls["mean_true_vs_control_logit_diff"].max()) if not controls.empty else None
    )
    weakest_positive = (
        float(positives["mean_true_vs_control_logit_diff"].min()) if not positives.empty else None
    )
    positive_minus_max_control = (
        None if positive_mean is None or max_control is None else float(positive_mean - max_control)
    )
    return {
        "model": "gpt2-small",
        "primary_metric": "true_vs_control_logit_diff",
        "n_examples": int(len(rows)),
        "validation_counts": dict(validation_counts),
        "metric_counts": dict(metric_counts),
        "positive_mean_true_vs_control_logit_diff": positive_mean,
        "max_control_mean_true_vs_control_logit_diff": max_control,
        "weakest_positive_family_mean": weakest_positive,
        "positive_minus_max_control_gap": positive_minus_max_control,
        "hardest_control_family": hardest_metric_control_family(family_rows),
        "positive_domain_means": _group_mean(rows, "token_domain", positive_only=True),
        "positive_length_means": _group_mean(rows, "sequence_length_bucket", positive_only=True),
        "thresholds": SUMMARY_THRESHOLDS,
        "status": status,
        "search_allowed": status == "metric_calibrated_for_next_spec",
        "refused_claims": [
            "This does not show an induction head.",
            "This does not show a circuit.",
            "This does not establish a broad GPT-2 property.",
            "Calibration success would only permit a tighter next spec.",
        ],
    }


def classify_metric_calibration_status(rows: pd.DataFrame, family_rows: pd.DataFrame) -> str:
    positives = family_rows[family_rows["should_show_induction_behavior"] == True]  # noqa: E712
    controls = family_rows[family_rows["should_show_induction_behavior"] == False]  # noqa: E712
    if positives.empty or controls.empty:
        return "prompt_bank_needs_revision"
    positive_mean = float(positives["mean_true_vs_control_logit_diff"].mean())
    max_control = float(controls["mean_true_vs_control_logit_diff"].max())
    weakest_positive = float(positives["mean_true_vs_control_logit_diff"].min())
    positive_fraction = float(positives["fraction_diff_positive"].mean())
    max_control_fraction = float(controls["fraction_diff_positive"].max())
    positive_domain_means = _group_mean(rows, "token_domain", positive_only=True)
    positive_length_means = _group_mean(rows, "sequence_length_bucket", positive_only=True)
    if any(value <= 0 for value in positive_domain_means.values()) or any(
        value <= 0 for value in positive_length_means.values()
    ):
        return "prompt_bank_needs_revision"
    if (
        positive_mean - max_control
        <= SUMMARY_THRESHOLDS["min_positive_minus_max_control_gap"]
        or max_control >= weakest_positive
        or positive_fraction < SUMMARY_THRESHOLDS["min_positive_fraction_diff_positive"]
        or max_control_fraction > SUMMARY_THRESHOLDS["max_control_fraction_diff_positive"]
    ):
        return "metric_needs_revision"
    return "metric_calibrated_for_next_spec"


def hardest_metric_control_family(family_rows: pd.DataFrame) -> dict[str, Any] | None:
    if family_rows.empty:
        return None
    controls = family_rows[family_rows["should_show_induction_behavior"] == False]  # noqa: E712
    if controls.empty:
        return None
    row = controls.sort_values("mean_true_vs_control_logit_diff", ascending=False).iloc[0]
    return {
        "family": str(row["family"]),
        "mean_true_vs_control_logit_diff": float(row["mean_true_vs_control_logit_diff"]),
        "fraction_diff_positive": float(row["fraction_diff_positive"]),
    }


def _base_sequences() -> list[dict[str, Any]]:
    return [
        _base("symbolic", "short", ["A", "B", "C", "D"], ["X", "Y", "Z"]),
        _base("symbolic", "medium", ["E", "F", "G", "H", "I"], ["J", "K", "L", "M"]),
        _base("symbolic", "long", ["N", "P", "Q", "R", "S", "T"], ["U", "V", "W", "X", "Y"]),
        _base("word", "short", ["red", "blue", "green", "yellow"], ["black", "white", "gray"]),
        _base("word", "medium", ["north", "south", "east", "west", "center"], ["left", "right", "up", "down"]),
        _base("word", "long", ["cat", "dog", "bird", "fish", "horse", "cow"], ["mouse", "lion", "tiger", "bear", "wolf"]),
        _base("number", "short", ["one", "two", "three", "four"], ["five", "six", "seven"]),
        _base("number", "medium", ["two", "four", "six", "eight", "ten"], ["one", "three", "five", "seven"]),
        _base("number", "long", ["three", "six", "nine", "twelve", "fifteen", "eighteen"], ["one", "two", "four", "five", "seven"]),
    ]


def _base(token_domain: str, length_bucket: str, sequence: list[str], distractors: list[str]) -> dict[str, Any]:
    return {
        "token_domain": token_domain,
        "sequence_length_bucket": length_bucket,
        "sequence_tokens": sequence,
        "distractors": distractors,
    }


def _rows_for_base_sequence(base: dict[str, Any]) -> list[dict[str, Any]]:
    sequence = list(base["sequence_tokens"])
    distractors = list(base["distractors"])
    prompt = _positive_prompt(sequence)
    expected = _next_token_text(sequence[-1])
    control = _next_token_text(sequence[0])
    second_control = _next_token_text(sequence[1])
    prefix = _metadata_prefix(base)
    rows = [
        _prompt_row(prefix, "calib_clean_repeat_" + base["token_domain"], prompt, expected, control, True),
        _prompt_row(prefix, "calib_clean_repeat_format_variant", _format_variant_prompt(sequence), expected, control, True),
        _prompt_row(prefix, "calib_wrong_target_same_prompt", prompt, control, expected, False),
        _prompt_row(prefix, "calib_target_swap_control", prompt, second_control, expected, False),
        _prompt_row(
            prefix,
            "calib_same_token_frequency_control",
            _same_frequency_prompt(sequence),
            expected,
            control,
            False,
        ),
        _prompt_row(prefix, "calib_reversed_order_control", _reversed_prompt(sequence), expected, control, False),
        _prompt_row(prefix, "calib_no_repeat_control", _no_repeat_prompt(sequence, distractors), expected, control, False),
        _prompt_row(prefix, "calib_frequency_trap_control", _frequency_trap_prompt(sequence), expected, control, False),
    ]
    return rows


def _prompt_row(
    prefix: dict[str, Any],
    family: str,
    prompt: str,
    true_expected_next_token: str,
    wrong_or_control_token: str,
    should_show: bool,
) -> dict[str, Any]:
    example_id = (
        f"{family}_{prefix['token_domain']}_{prefix['sequence_length_bucket']}_{prefix['family_index']:02d}"
    )
    return {
        "example_id": example_id,
        "task": "induction_metric_calibration",
        "family": family,
        "prompt": prompt,
        "true_expected_next_token": true_expected_next_token,
        "wrong_or_control_token": wrong_or_control_token,
        "expected_next_token": true_expected_next_token,
        "token_domain": prefix["token_domain"],
        "sequence_length_bucket": prefix["sequence_length_bucket"],
        "should_show_induction_behavior": should_show,
        "expected_behavior": "high_true_vs_control_logit_diff" if should_show else "low_true_vs_control_logit_diff",
        "control_rationale": _control_rationale(family),
    }


def _metadata_prefix(base: dict[str, Any]) -> dict[str, Any]:
    token_domain = str(base["token_domain"])
    bucket = str(base["sequence_length_bucket"])
    family_index = {"short": 0, "medium": 1, "long": 2}[bucket]
    return {
        "token_domain": token_domain,
        "sequence_length_bucket": bucket,
        "family_index": family_index,
    }


def _positive_prompt(sequence: list[str]) -> str:
    return " ".join([*sequence, *sequence[:-1]])


def _format_variant_prompt(sequence: list[str]) -> str:
    return " ".join(sequence) + "\n" + " ".join(sequence[:-1])


def _same_frequency_prompt(sequence: list[str]) -> str:
    return " ".join([*sequence, sequence[-2], sequence[0], sequence[1]])


def _reversed_prompt(sequence: list[str]) -> str:
    return " ".join([*sequence, *list(reversed(sequence[:-1]))])


def _no_repeat_prompt(sequence: list[str], distractors: list[str]) -> str:
    return " ".join([*sequence, *distractors[: len(sequence) - 1]])


def _frequency_trap_prompt(sequence: list[str]) -> str:
    return " ".join([*sequence, *([sequence[-1]] * (len(sequence) - 1))])


def _next_token_text(token: str) -> str:
    return f" {token}"


def _control_rationale(family: str) -> str:
    rationales = {
        "calib_wrong_target_same_prompt": "Same prompt but scored against a deliberately wrong target.",
        "calib_target_swap_control": "Same prompt with target swapped to another sequence token.",
        "calib_same_token_frequency_control": "Approximate token counts preserved without repeated-prefix structure.",
        "calib_reversed_order_control": "Repeated material appears in reversed order.",
        "calib_no_repeat_control": "Same initial vocabulary followed by distractors and no repeated prefix.",
        "calib_frequency_trap_control": "Expected token frequency is high without the target induction structure.",
    }
    return rationales.get(family, "Positive repeated-prefix prompt.")


def _unscored_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "metric_status": "not_scored",
        "target_logit": None,
        "control_logit": None,
        "true_vs_control_logit_diff": None,
        "target_probability": None,
        "control_probability": None,
        "probability_gap": None,
        "target_rank": None,
        "control_rank": None,
    }


def _group_mean(rows: pd.DataFrame, column: str, *, positive_only: bool) -> dict[str, float]:
    if rows.empty or column not in rows:
        return {}
    subset = rows
    if positive_only:
        subset = subset[subset["should_show_induction_behavior"] == True]  # noqa: E712
    subset = subset[subset["metric_status"] == "ok"].copy()
    if subset.empty:
        return {}
    subset["true_vs_control_logit_diff"] = pd.to_numeric(
        subset["true_vs_control_logit_diff"],
        errors="coerce",
    )
    grouped = subset.groupby(column, dropna=False)["true_vs_control_logit_diff"].mean()
    return {str(key): float(value) for key, value in grouped.to_dict().items()}


def _mean_or_none(values: Any) -> float | None:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return None
    return float(np.mean(series))


def _median_or_none(values: Any) -> float | None:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return None
    return float(np.median(series))


def _family_columns() -> list[str]:
    return [
        "family",
        "n_examples",
        "token_domains",
        "sequence_length_buckets",
        "should_show_induction_behavior",
        "mean_true_vs_control_logit_diff",
        "median_true_vs_control_logit_diff",
        "fraction_diff_positive",
        "mean_probability_gap",
        "median_target_rank",
    ]
