from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict

from self_ground.behavioral_tasks import BehavioralTask
from self_ground.logit_scoring import token_id_for_single_token_string


class TokenValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    family: str
    valid: bool
    prompt: str
    control_prompt: str
    control_type: Literal["matched_non_negation"]
    target_token_ids: list[int]
    foil_token_ids: list[int]
    control_target_token_ids: list[int]
    control_foil_token_ids: list[int]
    excluded_reason: str | None = None


class BehavioralTaskValidationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_tasks: int
    valid_tasks: int
    excluded_tasks: int
    valid_by_family: dict[str, int]
    excluded_by_family: dict[str, int]
    min_valid_tasks_per_family: int
    passes_minimum: bool


def _token_ids(model_adapter, token_strings: list[str], label: str) -> list[int]:
    if not token_strings:
        raise ValueError(f"{label} must contain at least one token string")
    return [token_id_for_single_token_string(model_adapter, token) for token in token_strings]


def _validate_one(model_adapter, task: BehavioralTask) -> TokenValidationResult:
    target_ids: list[int] = []
    foil_ids: list[int] = []
    control_target_ids: list[int] = []
    control_foil_ids: list[int] = []
    try:
        if not task.prompt or not task.control_prompt:
            raise ValueError("prompt and control_prompt must be nonempty")
        if task.control_type != "matched_non_negation":
            raise ValueError("control_type must be matched_non_negation")
        target_ids = _token_ids(model_adapter, task.target_tokens, "target_tokens")
        foil_ids = _token_ids(model_adapter, task.foil_tokens, "foil_tokens")
        control_target_ids = _token_ids(
            model_adapter,
            task.control_target_tokens,
            "control_target_tokens",
        )
        control_foil_ids = _token_ids(
            model_adapter,
            task.control_foil_tokens,
            "control_foil_tokens",
        )
        if set(target_ids) & set(foil_ids):
            raise ValueError("target and foil token ids overlap")
        if set(control_target_ids) & set(control_foil_ids):
            raise ValueError("control target and foil token ids overlap")
    except Exception as exc:
        return TokenValidationResult(
            task_id=task.id,
            family=task.family,
            valid=False,
            prompt=task.prompt,
            control_prompt=task.control_prompt,
            control_type="matched_non_negation",
            target_token_ids=target_ids,
            foil_token_ids=foil_ids,
            control_target_token_ids=control_target_ids,
            control_foil_token_ids=control_foil_ids,
            excluded_reason=str(exc),
        )
    return TokenValidationResult(
        task_id=task.id,
        family=task.family,
        valid=True,
        prompt=task.prompt,
        control_prompt=task.control_prompt,
        control_type="matched_non_negation",
        target_token_ids=target_ids,
        foil_token_ids=foil_ids,
        control_target_token_ids=control_target_ids,
        control_foil_token_ids=control_foil_ids,
    )


def validate_behavioral_tasks(
    *,
    model_adapter,
    tasks: list[BehavioralTask],
    min_valid_tasks_per_family: int = 2,
) -> tuple[list[BehavioralTask], list[TokenValidationResult], BehavioralTaskValidationSummary]:
    if min_valid_tasks_per_family < 1:
        raise ValueError("min_valid_tasks_per_family must be >= 1")
    task_by_id = {task.id: task for task in tasks}
    results = [_validate_one(model_adapter, task) for task in tasks]
    valid_tasks = [task_by_id[result.task_id] for result in results if result.valid]
    valid_counter = Counter(result.family for result in results if result.valid)
    excluded_counter = Counter(result.family for result in results if not result.valid)
    families = {task.family for task in tasks}
    passes_minimum = bool(families) and all(
        valid_counter.get(family, 0) >= min_valid_tasks_per_family for family in families
    )
    summary = BehavioralTaskValidationSummary(
        total_tasks=len(tasks),
        valid_tasks=len(valid_tasks),
        excluded_tasks=len(tasks) - len(valid_tasks),
        valid_by_family={family: int(valid_counter.get(family, 0)) for family in sorted(families)},
        excluded_by_family={
            family: int(excluded_counter.get(family, 0)) for family in sorted(families)
        },
        min_valid_tasks_per_family=min_valid_tasks_per_family,
        passes_minimum=passes_minimum,
    )
    return valid_tasks, results, summary
