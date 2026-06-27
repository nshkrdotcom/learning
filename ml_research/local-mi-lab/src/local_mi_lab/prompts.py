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
    "prompt",
    "expected_next_token",
    "control_prompt",
    "notes",
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
        records.append(
            PromptRecord(
                example_id=f"induction_{i:04d}",
                task="induction",
                prompt=" ".join(prompt_tokens),
                expected_next_token=expected,
                control_prompt=" ".join(control_tokens),
                notes="Repeated-token induction practice prompt with a simple nonmatching control.",
            )
        )
    return records


def generate_text_prompts() -> list[PromptRecord]:
    return [
        PromptRecord(
            example_id="text_0000",
            task="basic_next_token",
            prompt="The capital of France is",
            expected_next_token=" Paris",
            control_prompt="The capital of France is not",
            notes="Human-readable factual next-token prompt for tokenizer inspection.",
        ),
        PromptRecord(
            example_id="text_0001",
            task="basic_next_token",
            prompt="Once upon a time there was a",
            expected_next_token=" little",
            control_prompt="Once upon a time there was not a",
            notes="Human-readable continuation prompt for tokenizer inspection.",
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
            writer.writerow(record.to_dict())


def read_prompts_csv(path: str | Path) -> list[PromptRecord]:
    input_path = resolve_repo_path(path)
    with input_path.open("r", newline="", encoding="utf-8") as f:
        return [PromptRecord.from_mapping(row) for row in csv.DictReader(f)]


def _distractor_for(sequence: tuple[str, ...]) -> str:
    for candidate in ("X", "zero", "plain", "other"):
        if candidate not in sequence:
            return candidate
    return "other"
