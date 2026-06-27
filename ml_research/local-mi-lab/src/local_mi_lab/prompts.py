from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from local_mi_lab.paths import resolve_repo_path
from local_mi_lab.types import PromptRecord

PROMPT_COLUMNS = [
    "example_id",
    "task",
    "family",
    "prompt",
    "expected_next_token",
    "control_prompt",
    "notes",
    "prompt_tokens_text",
    "sequence_tokens",
    "repeated_prefix_length",
    "target_position_label",
    "expected_source_token",
    "expected_source_position_hint",
    "is_positive_induction_example",
    "control_family",
    "should_show_induction_behavior",
    "base_sequence_id",
    "family_index",
    "true_expected_next_token",
    "paired_positive_example_id",
    "wrong_or_control_token",
]

CONTROL_FAMILIES = [
    "positive_repeat_sequence",
    "no_repeat_control",
    "shuffled_repeat_control",
    "distractor_repeat_control",
    "same_token_frequency_control",
    "random_expected_token_control",
]

BASE_SEQUENCES: tuple[tuple[str, ...], ...] = (
    ("A", "B", "C", "D"),
    ("red", "blue", "green"),
    ("one", "two", "three"),
    ("cat", "dog", "bird"),
    ("north", "south", "east", "west"),
    ("Monday", "Tuesday", "Wednesday"),
    ("Paris", "London", "Rome"),
    ("alpha", "beta", "gamma", "delta"),
    ("apple", "banana", "cherry"),
    ("spring", "summer", "autumn", "winter"),
    ("iron", "copper", "silver"),
    ("circle", "square", "triangle"),
)


def generate_induction_prompts(n_examples: int, seed: int = 0) -> list[PromptRecord]:
    if n_examples <= 0:
        raise ValueError("n_examples must be positive")
    rng = random.Random(seed)
    pool: list[tuple[str, ...]] = []
    for sequence in BASE_SEQUENCES:
        for offset in range(len(sequence)):
            rotated = sequence[offset:] + sequence[:offset]
            pool.append(rotated)
    rng.shuffle(pool)

    records: list[PromptRecord] = []
    for i in range(n_examples):
        sequence = pool[i % len(pool)]
        prompt_tokens = [*sequence, *sequence[:-1]]
        control_tokens = [*sequence, *sequence[:-2], _distractor_for(sequence)]
        expected = f" {sequence[-1]}"
        expected_source_position = len(sequence) - 2
        records.append(
            PromptRecord(
                example_id=f"induction_{i:04d}",
                task="induction",
                family="positive_repeat_sequence",
                prompt=" ".join(prompt_tokens),
                expected_next_token=expected,
                control_prompt=" ".join(control_tokens),
                notes="Repeated-token induction practice prompt with a simple nonmatching control.",
                prompt_tokens_text=list(prompt_tokens),
                sequence_tokens=list(sequence),
                repeated_prefix_length=len(sequence) - 1,
                target_position_label="final",
                expected_source_token=sequence[expected_source_position],
                expected_source_position_hint=expected_source_position,
                is_positive_induction_example=True,
                control_family="",
                should_show_induction_behavior=True,
                base_sequence_id=f"induction_sequence_{i:04d}",
                family_index=i,
                true_expected_next_token=expected,
                paired_positive_example_id=f"induction_{i:04d}",
                wrong_or_control_token=" X",
            )
        )
    return records


def generate_induction_control_prompts(
    n_examples_per_family: int,
    families: list[str] | None = None,
    seed: int = 0,
) -> list[PromptRecord]:
    if n_examples_per_family <= 0:
        raise ValueError("n_examples_per_family must be positive")
    selected_families = families or CONTROL_FAMILIES
    unknown = sorted(set(selected_families) - set(CONTROL_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown control families: {unknown}")
    rng = random.Random(seed)
    pool = _sequence_pool(rng)
    records: list[PromptRecord] = []
    for family in selected_families:
        for i in range(n_examples_per_family):
            sequence = pool[i % len(pool)]
            records.append(_control_record(family, sequence, i))
    return records


def generate_text_prompts() -> list[PromptRecord]:
    return [
        PromptRecord(
            example_id="text_0000",
            task="basic_next_token",
            family="basic_next_token",
            prompt="The capital of France is",
            expected_next_token=" Paris",
            control_prompt="The capital of France is not",
            notes="Human-readable factual next-token prompt for tokenizer inspection.",
            is_positive_induction_example=False,
            should_show_induction_behavior=False,
            true_expected_next_token=" Paris",
            wrong_or_control_token=" London",
        ),
        PromptRecord(
            example_id="text_0001",
            task="basic_next_token",
            family="basic_next_token",
            prompt="Once upon a time there was a",
            expected_next_token=" little",
            control_prompt="Once upon a time there was not a",
            notes="Human-readable continuation prompt for tokenizer inspection.",
            is_positive_induction_example=False,
            should_show_induction_behavior=False,
            true_expected_next_token=" little",
            wrong_or_control_token=" big",
        ),
    ]


def write_prompt_dataset(
    records: list[PromptRecord],
    output_dir: str | Path,
    dataset_name: str,
) -> Path:
    root = resolve_repo_path(output_dir) / dataset_name
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "prompts.csv"
    summary_path = root / "summary.json"
    write_prompts_csv(records, csv_path)
    summary = {
        "dataset": dataset_name,
        "n_prompts": len(records),
        "tasks": sorted({record.task for record in records}),
        "families": _counts_by_family(records),
        "columns": PROMPT_COLUMNS,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return root


def write_prompts_csv(records: list[PromptRecord], path: str | Path) -> None:
    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROMPT_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_dict())


def read_prompts_csv(path: str | Path) -> list[PromptRecord]:
    input_path = resolve_repo_path(path)
    with input_path.open("r", newline="", encoding="utf-8") as f:
        return [PromptRecord.from_mapping(row) for row in csv.DictReader(f)]


def _distractor_for(sequence: tuple[str, ...]) -> str:
    for candidate in ("X", "zero", "plain", "other"):
        if candidate not in sequence:
            return candidate
    return "other"


def _sequence_pool(rng: random.Random) -> list[tuple[str, ...]]:
    pool: list[tuple[str, ...]] = []
    for sequence in BASE_SEQUENCES:
        for offset in range(len(sequence)):
            pool.append(sequence[offset:] + sequence[:offset])
    rng.shuffle(pool)
    return pool


def _counts_by_family(records: list[PromptRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.family] = counts.get(record.family, 0) + 1
    return dict(sorted(counts.items()))


def _control_record(family: str, sequence: tuple[str, ...], index: int) -> PromptRecord:
    positive_prompt_tokens = [*sequence, *sequence[:-1]]
    true_expected = f" {sequence[-1]}"
    control_prompt_tokens = [*sequence, "X", "Y", "Z"]
    source_position: int | None
    source_token: str
    prompt_tokens: list[str]
    expected = true_expected
    wrong_or_control_token = " X"
    repeated_prefix_length = 0

    if family == "positive_repeat_sequence":
        prompt_tokens = positive_prompt_tokens
        source_position = len(sequence) - 2
        source_token = sequence[source_position]
        repeated_prefix_length = len(sequence) - 1
        is_positive = True
        should_show = True
        notes = "Positive repeated-prefix induction prompt."
    elif family == "no_repeat_control":
        prompt_tokens = control_prompt_tokens
        source_position = None
        source_token = ""
        is_positive = False
        should_show = False
        notes = "No repeated prefix licenses the expected token."
    elif family == "shuffled_repeat_control":
        prompt_tokens = [*sequence, sequence[1], sequence[0]]
        source_position = 0
        source_token = sequence[0]
        wrong_or_control_token = _spaced_nontrue_token(sequence[1], true_expected)
        is_positive = False
        should_show = False
        notes = "Repeated tokens are shuffled, so prior occurrence does not license the expected token."
    elif family == "distractor_repeat_control":
        prompt_tokens = [*sequence, sequence[0], sequence[1]]
        source_position = 1
        source_token = sequence[1]
        wrong_or_control_token = _spaced_nontrue_token(sequence[2 % len(sequence)], true_expected)
        repeated_prefix_length = 2
        is_positive = False
        should_show = False
        notes = "Repeated subsequence points toward a different next token than the scored target."
    elif family == "same_token_frequency_control":
        if len(sequence) >= 4:
            prompt_tokens = [*sequence, sequence[-2], sequence[0], sequence[1]]
            source_position = 1
            source_token = sequence[1]
        else:
            prompt_tokens = [*sequence, sequence[-2], sequence[0]]
            source_position = 0
            source_token = sequence[0]
        wrong_or_control_token = _spaced_nontrue_token(source_token, true_expected)
        is_positive = False
        should_show = False
        notes = "Approximate token frequencies are preserved without the positive induction structure."
    elif family == "random_expected_token_control":
        prompt_tokens = positive_prompt_tokens
        expected = " X"
        wrong_or_control_token = expected
        source_position = len(sequence) - 2
        source_token = sequence[source_position]
        repeated_prefix_length = len(sequence) - 1
        is_positive = False
        should_show = False
        notes = (
            "Positive repeated-prefix prompt with a deliberately wrong scored expected token; "
            f"true positive expected token would be {true_expected!r}."
        )
    else:
        raise ValueError(f"Unknown control family {family!r}")

    return PromptRecord(
        example_id=f"{family}_{index:04d}",
        task="induction_controls",
        family=family,
        prompt=" ".join(prompt_tokens),
        expected_next_token=expected,
        control_prompt=" ".join(control_prompt_tokens),
        notes=notes,
        prompt_tokens_text=list(prompt_tokens),
        sequence_tokens=list(sequence),
        repeated_prefix_length=repeated_prefix_length,
        target_position_label="final",
        expected_source_token=source_token,
        expected_source_position_hint=source_position,
        is_positive_induction_example=is_positive,
        control_family="" if is_positive else family,
        should_show_induction_behavior=should_show,
        base_sequence_id=f"base_sequence_{index:04d}",
        family_index=index,
        true_expected_next_token=true_expected,
        paired_positive_example_id=f"positive_repeat_sequence_{index:04d}",
        wrong_or_control_token=wrong_or_control_token,
    )


def _spaced_nontrue_token(token: str, true_expected: str) -> str:
    spaced = f" {token}"
    return spaced if spaced != true_expected else " X"
