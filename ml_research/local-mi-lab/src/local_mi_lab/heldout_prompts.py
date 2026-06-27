from __future__ import annotations

import random
from typing import Any

from local_mi_lab.tokens import token_id_for_single_token
from local_mi_lab.types import PromptRecord

HELDOUT_POSITIVE_FAMILIES = [
    "heldout_symbolic_longer",
    "heldout_word_sequences",
    "heldout_number_sequences",
    "heldout_double_repeat",
]

HELDOUT_CONTROL_FAMILIES = [
    "heldout_wrong_target_same_prompt",
    "heldout_no_structure_same_tokens",
]

HELDOUT_FAMILIES = [*HELDOUT_POSITIVE_FAMILIES, *HELDOUT_CONTROL_FAMILIES]

SYMBOLIC_SEQUENCES = [
    ("A", "B", "C", "D", "E", "F"),
    ("G", "H", "I", "J", "K", "L"),
    ("M", "N", "O", "P", "Q", "R"),
    ("S", "T", "U", "V", "W", "X"),
]

WORD_SEQUENCES = [
    ("red", "blue", "green", "yellow"),
    ("cat", "dog", "bird", "fish"),
    ("north", "south", "east", "west"),
    ("spring", "summer", "autumn", "winter"),
    ("iron", "copper", "silver", "gold"),
    ("circle", "square", "triangle", "star"),
]

NUMBER_SEQUENCES = [
    ("one", "two", "three", "four"),
    ("two", "four", "six", "eight"),
    ("three", "five", "seven", "nine"),
    ("ten", "eleven", "twelve", "thirteen"),
]

DOUBLE_REPEAT_SEQUENCES = [
    ("A", "B", "C", "X", "Y"),
    ("red", "blue", "green", "black", "white"),
    ("one", "two", "three", "zero", "four"),
    ("cat", "dog", "bird", "fish", "horse"),
]


def generate_heldout_induction_prompts(
    n_examples_per_family: int,
    families: list[str] | None = None,
    seed: int = 10,
) -> list[PromptRecord]:
    if n_examples_per_family <= 0:
        raise ValueError("n_examples_per_family must be positive")
    selected_families = families or HELDOUT_FAMILIES
    unknown = sorted(set(selected_families) - set(HELDOUT_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown held-out families: {unknown}")

    rng = random.Random(seed)
    records: list[PromptRecord] = []
    for family in selected_families:
        pool = _pool_for_family(family, rng)
        for index in range(n_examples_per_family):
            sequence = pool[index % len(pool)]
            records.append(_heldout_record(family, sequence, index))
    return records


def validate_heldout_expected_tokens(records: list[PromptRecord], tokenizer: Any) -> None:
    for record in records:
        token_id_for_single_token(tokenizer, record.expected_next_token)
        token_id_for_single_token(tokenizer, record.true_expected_next_token)
        if record.wrong_or_control_token:
            token_id_for_single_token(tokenizer, record.wrong_or_control_token)


def _pool_for_family(family: str, rng: random.Random) -> list[tuple[str, ...]]:
    if family in {
        "heldout_symbolic_longer",
        "heldout_wrong_target_same_prompt",
        "heldout_no_structure_same_tokens",
    }:
        base = SYMBOLIC_SEQUENCES
    elif family == "heldout_word_sequences":
        base = WORD_SEQUENCES
    elif family == "heldout_number_sequences":
        base = NUMBER_SEQUENCES
    elif family == "heldout_double_repeat":
        base = DOUBLE_REPEAT_SEQUENCES
    else:
        raise ValueError(f"Unknown held-out family {family!r}")
    pool: list[tuple[str, ...]] = []
    for sequence in base:
        for offset in range(len(sequence)):
            pool.append(sequence[offset:] + sequence[:offset])
    rng.shuffle(pool)
    return pool


def _heldout_record(family: str, sequence: tuple[str, ...], index: int) -> PromptRecord:
    positive_family = _paired_positive_family(family)
    positive_id = f"{positive_family}_{index:04d}"
    base_id = f"heldout_sequence_{index:04d}"

    if family == "heldout_symbolic_longer":
        return _positive_record(
            family=family,
            sequence=sequence,
            index=index,
            base_id=base_id,
            construction_note="Longer symbolic repeated-prefix prompt.",
        )
    if family == "heldout_word_sequences":
        return _positive_record(
            family=family,
            sequence=sequence,
            index=index,
            base_id=base_id,
            construction_note="Common-word repeated-prefix prompt.",
        )
    if family == "heldout_number_sequences":
        return _positive_record(
            family=family,
            sequence=sequence,
            index=index,
            base_id=base_id,
            construction_note="Number-word repeated-prefix prompt.",
        )
    if family == "heldout_double_repeat":
        prompt_tokens = [*sequence, *sequence[:-1]]
        expected = f" {sequence[-1]}"
        control_tokens = [*sequence, sequence[0], sequence[2], sequence[1]]
        source_position = len(sequence) - 2
        return PromptRecord(
            example_id=f"{family}_{index:04d}",
            task="induction_heldout",
            family=family,
            prompt=" ".join(prompt_tokens),
            expected_next_token=expected,
            control_prompt=" ".join(control_tokens),
            notes="Two repeated subsequences are present; only the final repeated prefix licenses the target.",
            prompt_tokens_text=list(prompt_tokens),
            sequence_tokens=list(sequence),
            repeated_prefix_length=len(sequence) - 1,
            target_position_label="final",
            expected_source_token=sequence[source_position],
            expected_source_position_hint=source_position,
            is_positive_induction_example=True,
            control_family="",
            should_show_induction_behavior=True,
            base_sequence_id=base_id,
            family_index=index,
            true_expected_next_token=expected,
            paired_positive_example_id=f"{family}_{index:04d}",
            wrong_or_control_token=_wrong_token(sequence, expected),
            heldout_family_type="positive",
            heldout_construction_note="Double-repeat positive prompt with a distractor pattern.",
        )
    if family == "heldout_wrong_target_same_prompt":
        prompt_tokens = [*sequence, *sequence[:-1]]
        true_expected = f" {sequence[-1]}"
        wrong = _wrong_token(sequence, true_expected)
        return PromptRecord(
            example_id=f"{family}_{index:04d}",
            task="induction_heldout",
            family=family,
            prompt=" ".join(prompt_tokens),
            expected_next_token=wrong,
            control_prompt=" ".join(prompt_tokens),
            notes="Target-specificity control: same prompt as positive but scored against a wrong target.",
            prompt_tokens_text=list(prompt_tokens),
            sequence_tokens=list(sequence),
            repeated_prefix_length=len(sequence) - 1,
            target_position_label="final",
            expected_source_token="",
            expected_source_position_hint=None,
            is_positive_induction_example=False,
            control_family=family,
            should_show_induction_behavior=False,
            base_sequence_id=base_id,
            family_index=index,
            true_expected_next_token=true_expected,
            paired_positive_example_id=positive_id,
            wrong_or_control_token=wrong,
            heldout_family_type="control",
            heldout_construction_note="Wrong-target same-prompt control.",
        )
    if family == "heldout_no_structure_same_tokens":
        positive_tokens = [*sequence, *sequence[:-1]]
        prompt_tokens = _no_structure_tokens(sequence)
        true_expected = f" {sequence[-1]}"
        return PromptRecord(
            example_id=f"{family}_{index:04d}",
            task="induction_heldout",
            family=family,
            prompt=" ".join(prompt_tokens),
            expected_next_token=true_expected,
            control_prompt=" ".join(positive_tokens),
            notes="Structure-specificity control: same token bag without ordered repeated-prefix structure.",
            prompt_tokens_text=list(prompt_tokens),
            sequence_tokens=list(sequence),
            repeated_prefix_length=0,
            target_position_label="final",
            expected_source_token="",
            expected_source_position_hint=None,
            is_positive_induction_example=False,
            control_family=family,
            should_show_induction_behavior=False,
            base_sequence_id=base_id,
            family_index=index,
            true_expected_next_token=true_expected,
            paired_positive_example_id=positive_id,
            wrong_or_control_token=_wrong_token(sequence, true_expected),
            heldout_family_type="control",
            heldout_construction_note="Same-token-bag control with no induction structure.",
        )
    raise ValueError(f"Unknown held-out family {family!r}")


def _positive_record(
    *,
    family: str,
    sequence: tuple[str, ...],
    index: int,
    base_id: str,
    construction_note: str,
) -> PromptRecord:
    prompt_tokens = [*sequence, *sequence[:-1]]
    control_tokens = _no_structure_tokens(sequence)
    expected = f" {sequence[-1]}"
    source_position = len(sequence) - 2
    return PromptRecord(
        example_id=f"{family}_{index:04d}",
        task="induction_heldout",
        family=family,
        prompt=" ".join(prompt_tokens),
        expected_next_token=expected,
        control_prompt=" ".join(control_tokens),
        notes=construction_note,
        prompt_tokens_text=list(prompt_tokens),
        sequence_tokens=list(sequence),
        repeated_prefix_length=len(sequence) - 1,
        target_position_label="final",
        expected_source_token=sequence[source_position],
        expected_source_position_hint=source_position,
        is_positive_induction_example=True,
        control_family="",
        should_show_induction_behavior=True,
        base_sequence_id=base_id,
        family_index=index,
        true_expected_next_token=expected,
        paired_positive_example_id=f"{family}_{index:04d}",
        wrong_or_control_token=_wrong_token(sequence, expected),
        heldout_family_type="positive",
        heldout_construction_note=construction_note,
    )


def _paired_positive_family(family: str) -> str:
    if family in HELDOUT_POSITIVE_FAMILIES:
        return family
    return "heldout_symbolic_longer"


def _no_structure_tokens(sequence: tuple[str, ...]) -> list[str]:
    if len(sequence) >= 6:
        return [sequence[0], sequence[2], sequence[4], sequence[1], sequence[3], sequence[5], sequence[2], sequence[0], sequence[4], sequence[1], sequence[3]]
    return [*sequence, sequence[-2], sequence[0], sequence[1], sequence[-3]]


def _wrong_token(sequence: tuple[str, ...], true_expected: str) -> str:
    for token in [*sequence[:-1], " X", " zero", " other"]:
        spaced = token if token.startswith(" ") else f" {token}"
        if spaced != true_expected:
            return spaced
    return " X"
