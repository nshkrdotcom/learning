from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptRecord:
    example_id: str
    task: str
    prompt: str
    expected_next_token: str
    control_prompt: str
    notes: str
    prompt_tokens_text: list[str] = field(default_factory=list)
    sequence_tokens: list[str] = field(default_factory=list)
    repeated_prefix_length: int | None = None
    target_position_label: str = ""
    expected_source_token: str = ""
    expected_source_position_hint: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_csv_dict(self) -> dict[str, str | int | None]:
        row = self.to_dict()
        row["prompt_tokens_text"] = json.dumps(self.prompt_tokens_text)
        row["sequence_tokens"] = json.dumps(self.sequence_tokens)
        return row

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> PromptRecord:
        return cls(
            example_id=str(row["example_id"]),
            task=str(row["task"]),
            prompt=str(row["prompt"]),
            expected_next_token=str(row["expected_next_token"]),
            control_prompt=str(row["control_prompt"]),
            notes=str(row.get("notes", "")),
            prompt_tokens_text=_parse_string_list(row.get("prompt_tokens_text")),
            sequence_tokens=_parse_string_list(row.get("sequence_tokens")),
            repeated_prefix_length=_parse_optional_int(row.get("repeated_prefix_length")),
            target_position_label=str(row.get("target_position_label") or ""),
            expected_source_token=str(row.get("expected_source_token") or ""),
            expected_source_position_hint=_parse_optional_int(
                row.get("expected_source_position_hint")
            ),
        )


def _parse_string_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, got {value!r}")
    return [str(item) for item in parsed]


def _parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
