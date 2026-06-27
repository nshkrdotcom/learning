from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptRecord:
    example_id: str
    task: str
    family: str
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
    is_positive_induction_example: bool = True
    control_family: str = ""
    should_show_induction_behavior: bool = True
    base_sequence_id: str = ""
    family_index: int | None = None
    true_expected_next_token: str = ""
    paired_positive_example_id: str = ""
    wrong_or_control_token: str = ""
    heldout_family_type: str = ""
    heldout_construction_note: str = ""
    distractor_position_hint: int | None = None
    characterization_axis: str = ""
    sequence_length_bucket: str = ""
    token_domain: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_csv_dict(self) -> dict[str, str | int | None]:
        row = self.to_dict()
        row["prompt_tokens_text"] = json.dumps(self.prompt_tokens_text)
        row["sequence_tokens"] = json.dumps(self.sequence_tokens)
        return row

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> PromptRecord:
        family = str(row.get("family") or "positive_repeat_sequence")
        sequence_tokens = _parse_string_list(row.get("sequence_tokens"))
        family_index = _parse_family_index(row)
        true_expected_next_token = str(
            row.get("true_expected_next_token")
            or (f" {sequence_tokens[-1]}" if sequence_tokens else row["expected_next_token"])
        )
        paired_positive_example_id = str(
            row.get("paired_positive_example_id")
            or _default_paired_positive_example_id(family, family_index, str(row["example_id"]))
        )
        wrong_or_control_token = str(
            row.get("wrong_or_control_token")
            or _default_wrong_or_control_token(str(row["expected_next_token"]), true_expected_next_token)
        )
        return cls(
            example_id=str(row["example_id"]),
            task=str(row["task"]),
            family=family,
            prompt=str(row["prompt"]),
            expected_next_token=str(row["expected_next_token"]),
            control_prompt=str(row["control_prompt"]),
            notes=str(row.get("notes", "")),
            prompt_tokens_text=_parse_string_list(row.get("prompt_tokens_text")),
            sequence_tokens=sequence_tokens,
            repeated_prefix_length=_parse_optional_int(row.get("repeated_prefix_length")),
            target_position_label=str(row.get("target_position_label") or ""),
            expected_source_token=str(row.get("expected_source_token") or ""),
            expected_source_position_hint=_parse_optional_int(
                row.get("expected_source_position_hint")
            ),
            is_positive_induction_example=_parse_bool(
                row.get("is_positive_induction_example", True)
            ),
            control_family=str(row.get("control_family") or ""),
            should_show_induction_behavior=_parse_bool(
                row.get("should_show_induction_behavior", True)
            ),
            base_sequence_id=str(row.get("base_sequence_id") or _default_base_sequence_id(family_index)),
            family_index=family_index,
            true_expected_next_token=true_expected_next_token,
            paired_positive_example_id=paired_positive_example_id,
            wrong_or_control_token=wrong_or_control_token,
            heldout_family_type=str(row.get("heldout_family_type") or ""),
            heldout_construction_note=str(row.get("heldout_construction_note") or ""),
            distractor_position_hint=_parse_optional_int(row.get("distractor_position_hint")),
            characterization_axis=str(row.get("characterization_axis") or ""),
            sequence_length_bucket=str(row.get("sequence_length_bucket") or ""),
            token_domain=str(row.get("token_domain") or ""),
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


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def _parse_family_index(row: dict[str, Any]) -> int | None:
    parsed = _parse_optional_int(row.get("family_index"))
    if parsed is not None:
        return parsed
    example_id = str(row.get("example_id") or "")
    suffix = example_id.rsplit("_", maxsplit=1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _default_base_sequence_id(family_index: int | None) -> str:
    return f"base_sequence_{family_index:04d}" if family_index is not None else ""


def _default_paired_positive_example_id(
    family: str,
    family_index: int | None,
    example_id: str,
) -> str:
    if family_index is None:
        return example_id if family == "positive_repeat_sequence" else ""
    return f"positive_repeat_sequence_{family_index:04d}"


def _default_wrong_or_control_token(expected_next_token: str, true_expected_next_token: str) -> str:
    if expected_next_token != true_expected_next_token:
        return expected_next_token
    return " X"
