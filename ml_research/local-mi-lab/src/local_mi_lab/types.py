from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PromptRecord:
    example_id: str
    task: str
    prompt: str
    expected_next_token: str
    control_prompt: str
    notes: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> PromptRecord:
        return cls(
            example_id=str(row["example_id"]),
            task=str(row["task"]),
            prompt=str(row["prompt"]),
            expected_next_token=str(row["expected_next_token"]),
            control_prompt=str(row["control_prompt"]),
            notes=str(row.get("notes", "")),
        )
