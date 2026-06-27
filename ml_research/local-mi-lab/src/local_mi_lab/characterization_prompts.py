from __future__ import annotations

import random
from typing import Any

from local_mi_lab.tokens import token_id_for_single_token
from local_mi_lab.types import PromptRecord

CHAR_POSITIVE_FAMILIES = [
    "char_symbolic_short",
    "char_symbolic_long",
    "char_word_short",
    "char_word_long",
    "char_number_short",
    "char_number_long",
    "char_multi_distractor",
]

CHAR_CONTROL_FAMILIES = [
    "char_reversed_control",
    "char_target_swap_control",
]

CHARACTERIZATION_FAMILIES = [*CHAR_POSITIVE_FAMILIES, *CHAR_CONTROL_FAMILIES]

SYMBOLIC_SHORT = [
    ("A", "B", "C", "D"),
    ("E", "F", "G", "H"),
    ("J", "K", "L", "M"),
    ("N", "P", "Q", "R"),
]

SYMBOLIC_LONG = [
    ("A", "B", "C", "D", "E", "F"),
    ("G", "H", "I", "J", "K", "L"),
    ("M", "N", "O", "P", "Q", "R"),
    ("S", "T", "U", "V", "W", "X"),
]

WORD_SHORT = [
    ("red", "blue", "green", "yellow"),
    ("cat", "dog", "bird", "fish"),
    ("north", "south", "east", "west"),
    ("iron", "copper", "silver", "gold"),
]

WORD_LONG = [
    ("red", "blue", "green", "yellow", "black", "white"),
    ("spring", "summer", "autumn", "winter", "dawn", "night"),
    ("circle", "square", "triangle", "star", "line", "point"),
    ("alpha", "beta", "gamma", "delta", "omega", "lambda"),
]

NUMBER_SHORT = [
    ("one", "two", "three", "four"),
    ("two", "four", "six", "eight"),
    ("three", "five", "seven", "nine"),
    ("ten", "eleven", "twelve", "thirteen"),
]

NUMBER_LONG = [
    ("one", "two", "three", "four", "five", "six"),
    ("two", "four", "six", "eight", "ten", "twelve"),
    ("three", "five", "seven", "nine", "eleven", "thirteen"),
    ("zero", "one", "two", "three", "four", "five"),
]

MULTI_DISTRACTOR = [
    ("A", "B", "C", "X", "Y"),
    ("red", "blue", "green", "black", "white"),
    ("one", "two", "three", "zero", "four"),
    ("cat", "dog", "bird", "fish", "horse"),
]


def generate_characterization_prompts(
    n_examples_per_family: int,
    families: list[str] | None = None,
    seed: int = 20,
) -> list[PromptRecord]:
    if n_examples_per_family <= 0:
        raise ValueError("n_examples_per_family must be positive")
    selected_families = families or CHARACTERIZATION_FAMILIES
    unknown = sorted(set(selected_families) - set(CHARACTERIZATION_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown characterization families: {unknown}")
    symbolic_pool = _sequence_pool(SYMBOLIC_SHORT, random.Random(f"{seed}:symbolic_short"))
    records: list[PromptRecord] = []
    for family in selected_families:
        pool = symbolic_pool if family in CHAR_CONTROL_FAMILIES else _family_pool(family, seed)
        for index in range(n_examples_per_family):
            records.append(_characterization_record(family, pool[index % len(pool)], index))
    return records


def validate_characterization_expected_tokens(records: list[PromptRecord], tokenizer: Any) -> None:
    for record in records:
        token_id_for_single_token(tokenizer, record.expected_next_token)
        token_id_for_single_token(tokenizer, record.true_expected_next_token)
        if record.wrong_or_control_token:
            token_id_for_single_token(tokenizer, record.wrong_or_control_token)


def _family_pool(family: str, seed: int) -> list[tuple[str, ...]]:
    base = {
        "char_symbolic_short": SYMBOLIC_SHORT,
        "char_symbolic_long": SYMBOLIC_LONG,
        "char_word_short": WORD_SHORT,
        "char_word_long": WORD_LONG,
        "char_number_short": NUMBER_SHORT,
        "char_number_long": NUMBER_LONG,
        "char_multi_distractor": MULTI_DISTRACTOR,
    }[family]
    return _sequence_pool(base, random.Random(f"{seed}:{family}"))


def _sequence_pool(
    sequences: list[tuple[str, ...]],
    rng: random.Random,
) -> list[tuple[str, ...]]:
    pool: list[tuple[str, ...]] = []
    for sequence in sequences:
        for offset in range(len(sequence)):
            pool.append(sequence[offset:] + sequence[:offset])
    rng.shuffle(pool)
    return pool


def _characterization_record(
    family: str,
    sequence: tuple[str, ...],
    index: int,
) -> PromptRecord:
    if family in CHAR_POSITIVE_FAMILIES:
        return _positive_record(family, sequence, index)
    if family == "char_reversed_control":
        return _reversed_control(sequence, index)
    if family == "char_target_swap_control":
        return _target_swap_control(sequence, index)
    raise ValueError(f"Unknown characterization family {family!r}")


def _positive_record(family: str, sequence: tuple[str, ...], index: int) -> PromptRecord:
    if family == "char_multi_distractor":
        prompt_tokens = [*sequence, *sequence[:-1], sequence[0], sequence[-2]]
        distractor_position = len(prompt_tokens) - 2
    else:
        prompt_tokens = [*sequence, *sequence[:-1]]
        distractor_position = None
    control_tokens = _reversed_tokens(sequence)
    expected = f" {sequence[-1]}"
    wrong = _wrong_token(sequence, expected)
    source_position = len(sequence) - 2
    return PromptRecord(
        example_id=f"{family}_{index:04d}",
        task="candidate_characterization",
        family=family,
        prompt=" ".join(prompt_tokens),
        expected_next_token=expected,
        control_prompt=" ".join(control_tokens),
        notes="Characterization positive repeated-prefix prompt.",
        prompt_tokens_text=list(prompt_tokens),
        sequence_tokens=list(sequence),
        repeated_prefix_length=len(sequence) - 1,
        target_position_label="final",
        expected_source_token=sequence[source_position],
        expected_source_position_hint=source_position,
        distractor_position_hint=distractor_position,
        is_positive_induction_example=True,
        control_family="",
        should_show_induction_behavior=True,
        base_sequence_id=f"char_sequence_{index:04d}",
        family_index=index,
        true_expected_next_token=expected,
        paired_positive_example_id=f"{family}_{index:04d}",
        wrong_or_control_token=wrong,
        heldout_family_type="positive",
        heldout_construction_note="Candidate characterization repeated-prefix positive.",
        characterization_axis=_axis_for_family(family),
        sequence_length_bucket=_length_bucket(family),
        token_domain=_token_domain(family),
    )


def _reversed_control(sequence: tuple[str, ...], index: int) -> PromptRecord:
    prompt_tokens = _reversed_tokens(sequence)
    expected = f" {sequence[-1]}"
    return PromptRecord(
        example_id=f"char_reversed_control_{index:04d}",
        task="candidate_characterization",
        family="char_reversed_control",
        prompt=" ".join(prompt_tokens),
        expected_next_token=expected,
        control_prompt=" ".join([*sequence, *sequence[:-1]]),
        notes="Control with same symbolic-short token set but reversed structure.",
        prompt_tokens_text=list(prompt_tokens),
        sequence_tokens=list(sequence),
        repeated_prefix_length=0,
        target_position_label="final",
        expected_source_token="",
        expected_source_position_hint=None,
        is_positive_induction_example=False,
        control_family="char_reversed_control",
        should_show_induction_behavior=False,
        base_sequence_id=f"char_sequence_{index:04d}",
        family_index=index,
        true_expected_next_token=expected,
        paired_positive_example_id=f"char_symbolic_short_{index:04d}",
        wrong_or_control_token=_wrong_token(sequence, expected),
        heldout_family_type="control",
        heldout_construction_note="Reversed-order control.",
        characterization_axis="structure_control",
        sequence_length_bucket="short",
        token_domain="symbolic",
    )


def _target_swap_control(sequence: tuple[str, ...], index: int) -> PromptRecord:
    prompt_tokens = [*sequence, *sequence[:-1]]
    true_expected = f" {sequence[-1]}"
    wrong = _wrong_token(sequence, true_expected)
    return PromptRecord(
        example_id=f"char_target_swap_control_{index:04d}",
        task="candidate_characterization",
        family="char_target_swap_control",
        prompt=" ".join(prompt_tokens),
        expected_next_token=wrong,
        control_prompt=" ".join(prompt_tokens),
        notes="Control with positive prompt but deliberately wrong scored target.",
        prompt_tokens_text=list(prompt_tokens),
        sequence_tokens=list(sequence),
        repeated_prefix_length=len(sequence) - 1,
        target_position_label="final",
        expected_source_token="",
        expected_source_position_hint=None,
        is_positive_induction_example=False,
        control_family="char_target_swap_control",
        should_show_induction_behavior=False,
        base_sequence_id=f"char_sequence_{index:04d}",
        family_index=index,
        true_expected_next_token=true_expected,
        paired_positive_example_id=f"char_symbolic_short_{index:04d}",
        wrong_or_control_token=wrong,
        heldout_family_type="control",
        heldout_construction_note="Wrong-target same-prompt control.",
        characterization_axis="target_control",
        sequence_length_bucket="short",
        token_domain="symbolic",
    )


def _reversed_tokens(sequence: tuple[str, ...]) -> list[str]:
    return [*sequence, *reversed(sequence[:-1])]


def _wrong_token(sequence: tuple[str, ...], expected: str) -> str:
    for token in [*sequence, "X", "zero", "other"]:
        candidate = f" {token}"
        if candidate != expected:
            return candidate
    return " X"


def _axis_for_family(family: str) -> str:
    if "multi_distractor" in family:
        return "distractor"
    if "long" in family:
        return "sequence_length"
    if "short" in family:
        return "token_domain"
    return "token_domain"


def _length_bucket(family: str) -> str:
    if "long" in family:
        return "long"
    if "multi_distractor" in family:
        return "medium"
    return "short"


def _token_domain(family: str) -> str:
    if "word" in family:
        return "word"
    if "number" in family:
        return "number"
    return "symbolic"
