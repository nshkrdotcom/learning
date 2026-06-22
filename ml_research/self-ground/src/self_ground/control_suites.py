from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from self_ground.behavioral_tasks import TASK_FAMILY_ORDER, BehavioralTask
from self_ground.io import write_config, write_jsonl
from self_ground.logit_scoring import token_id_for_single_token_string

ControlSuiteMode = Literal[
    "matched_non_negation_current",
    "lexical_identity_control",
    "semantic_unrelated_control",
    "shuffled_target_control",
    "hard_negative_control",
    "multi_control",
]

SINGLE_CONTROL_SUITES = [
    "matched_non_negation_current",
    "lexical_identity_control",
    "semantic_unrelated_control",
    "shuffled_target_control",
    "hard_negative_control",
]


class ControlCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    family: str
    control_suite: str
    control_case_id: str
    control_prompt: str
    control_type: str
    control_target_tokens: list[str]
    control_foil_tokens: list[str]
    control_target_token_ids: list[int]
    control_foil_token_ids: list[int]
    source_task_id: str | None = None
    metadata: dict[str, Any] = {}


class ControlSuiteValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_mode: str
    expanded_suites: list[str]
    total_tasks: int
    total_control_cases: int
    valid_control_cases: int
    excluded_control_cases: int
    min_control_cases_per_family: int
    valid_cases_by_family: dict[str, int]
    valid_cases_by_suite: dict[str, int]
    excluded_by_reason: dict[str, int]
    missing_required_families: list[str]
    passes_minimum: bool


def _stable_case_id(task_id: str, suite: str, prompt: str, tokens: list[str]) -> str:
    payload = f"{task_id}|{suite}|{prompt}|{','.join(tokens)}"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"{task_id}_{suite}_{digest[:10]}"


def _token_ids(model_adapter, token_strings: list[str], label: str) -> list[int]:
    if not token_strings:
        raise ValueError(f"{label} must contain at least one token string")
    return [token_id_for_single_token_string(model_adapter, token) for token in token_strings]


def _token_text(token: str) -> str:
    return token.strip().replace('"', "'")


def _same_family_alternate(
    tasks_by_family: dict[str, list[BehavioralTask]],
    task: BehavioralTask,
) -> BehavioralTask | None:
    candidates = [
        candidate for candidate in tasks_by_family.get(task.family, []) if candidate.id != task.id
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda candidate: candidate.id)[0]


def _other_family_task(tasks: list[BehavioralTask], task: BehavioralTask) -> BehavioralTask | None:
    candidates = [candidate for candidate in tasks if candidate.family != task.family]
    if not candidates:
        return None
    return sorted(candidates, key=lambda candidate: (candidate.family, candidate.id))[0]


def _case_payload(
    *,
    task: BehavioralTask,
    suite: str,
    tasks: list[BehavioralTask],
    tasks_by_family: dict[str, list[BehavioralTask]],
) -> dict[str, Any] | None:
    if suite == "matched_non_negation_current":
        return {
            "prompt": task.control_prompt,
            "target_tokens": task.control_target_tokens,
            "foil_tokens": task.control_foil_tokens,
            "control_type": task.control_type,
            "source_task_id": task.id,
            "metadata": {"control_design": "current_matched_non_negation"},
        }
    if suite == "lexical_identity_control":
        target = _token_text(task.target_tokens[0])
        foil = _token_text(task.foil_tokens[0])
        return {
            "prompt": f'The word "{foil}" contrasts with the word',
            "target_tokens": task.target_tokens,
            "foil_tokens": task.foil_tokens,
            "control_type": "lexical_identity",
            "source_task_id": task.id,
            "metadata": {
                "control_design": "same_target_foil_without_task_negation_scope",
                "foil_word": foil,
                "target_word": target,
            },
        }
    if suite == "semantic_unrelated_control":
        other = _other_family_task(tasks, task)
        if other is None:
            return None
        return {
            "prompt": other.control_prompt,
            "target_tokens": other.control_target_tokens,
            "foil_tokens": other.control_foil_tokens,
            "control_type": "semantic_unrelated",
            "source_task_id": other.id,
            "metadata": {
                "control_design": "different_family_matched_control",
                "source_family": other.family,
            },
        }
    if suite == "shuffled_target_control":
        other = _same_family_alternate(tasks_by_family, task)
        if other is None:
            return None
        return {
            "prompt": task.control_prompt,
            "target_tokens": other.target_tokens,
            "foil_tokens": other.foil_tokens,
            "control_type": "shuffled_target",
            "source_task_id": other.id,
            "metadata": {
                "control_design": "same_family_different_target_foil",
                "source_family": other.family,
            },
        }
    if suite == "hard_negative_control":
        return {
            "prompt": task.control_prompt,
            "target_tokens": task.control_target_tokens,
            "foil_tokens": task.control_foil_tokens,
            "control_type": "hard_negative",
            "source_task_id": task.id,
            "metadata": {
                "control_design": "baseline_calibrated_non_negated_semantic_control",
            },
        }
    raise ValueError(f"unknown control suite: {suite}")


def expanded_control_suites(mode: ControlSuiteMode) -> list[str]:
    if mode == "multi_control":
        return list(SINGLE_CONTROL_SUITES)
    if mode not in SINGLE_CONTROL_SUITES:
        raise ValueError(f"unknown control_suite: {mode}")
    return [mode]


def build_control_cases(
    *,
    tasks: list[BehavioralTask],
    model_adapter,
    mode: ControlSuiteMode = "matched_non_negation_current",
    min_control_cases_per_family: int = 1,
    required_families: list[str] | None = None,
) -> tuple[list[ControlCase], ControlSuiteValidation, list[dict[str, Any]]]:
    if min_control_cases_per_family < 1:
        raise ValueError("min_control_cases_per_family must be >= 1")
    suites = expanded_control_suites(mode)
    tasks_by_family: dict[str, list[BehavioralTask]] = defaultdict(list)
    for task in tasks:
        tasks_by_family[task.family].append(task)

    cases: list[ControlCase] = []
    excluded: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    required = list(TASK_FAMILY_ORDER if required_families is None else required_families)
    for task in tasks:
        for suite in suites:
            try:
                payload = _case_payload(
                    task=task,
                    suite=suite,
                    tasks=tasks,
                    tasks_by_family=tasks_by_family,
                )
                if payload is None:
                    raise ValueError("control case could not be generated")
                target_ids = _token_ids(
                    model_adapter,
                    list(payload["target_tokens"]),
                    "control_target_tokens",
                )
                foil_ids = _token_ids(
                    model_adapter,
                    list(payload["foil_tokens"]),
                    "control_foil_tokens",
                )
                if set(target_ids) & set(foil_ids):
                    raise ValueError("control target and foil token ids overlap")
                prompt = str(payload["prompt"])
                if not prompt:
                    raise ValueError("control_prompt must be nonempty")
                cases.append(
                    ControlCase(
                        task_id=task.id,
                        family=task.family,
                        control_suite=suite,
                        control_case_id=_stable_case_id(
                            task.id,
                            suite,
                            prompt,
                            list(payload["target_tokens"]),
                        ),
                        control_prompt=prompt,
                        control_type=str(payload["control_type"]),
                        control_target_tokens=list(payload["target_tokens"]),
                        control_foil_tokens=list(payload["foil_tokens"]),
                        control_target_token_ids=target_ids,
                        control_foil_token_ids=foil_ids,
                        source_task_id=payload.get("source_task_id"),
                        metadata=dict(payload.get("metadata") or {}),
                    )
                )
            except Exception as exc:
                reason = str(exc)
                reason_counts[reason] += 1
                excluded.append(
                    {
                        "task_id": task.id,
                        "family": task.family,
                        "control_suite": suite,
                        "excluded_reason": reason,
                    }
                )

    family_counts = Counter(case.family for case in cases)
    suite_counts = Counter(case.control_suite for case in cases)
    missing = [
        family
        for family in required
        if family_counts.get(family, 0) < min_control_cases_per_family
    ]
    validation = ControlSuiteValidation(
        requested_mode=mode,
        expanded_suites=suites,
        total_tasks=len(tasks),
        total_control_cases=len(tasks) * len(suites),
        valid_control_cases=len(cases),
        excluded_control_cases=len(excluded),
        min_control_cases_per_family=min_control_cases_per_family,
        valid_cases_by_family={family: int(family_counts.get(family, 0)) for family in required},
        valid_cases_by_suite={suite: int(suite_counts.get(suite, 0)) for suite in suites},
        excluded_by_reason=dict(sorted(reason_counts.items())),
        missing_required_families=missing,
        passes_minimum=not missing and len(cases) == len(tasks) * len(suites),
    )
    return cases, validation, excluded


def write_control_suite_artifacts(
    *,
    out_dir: str | Path,
    mode: ControlSuiteMode,
    cases: list[ControlCase],
    validation: ControlSuiteValidation,
    excluded: list[dict[str, Any]],
) -> None:
    path = Path(out_dir)
    write_config(
        {
            "control_suite": mode,
            "expanded_suites": validation.expanded_suites,
            "n_control_cases": len(cases),
            "claim_gate_role": (
                "multi_control must pass all configured controls for candidate evidence"
                if mode == "multi_control"
                else "single control suite"
            ),
        },
        path / "control_suite.json",
    )
    write_jsonl(cases, path / "control_task_mapping.jsonl")
    write_jsonl(excluded, path / "excluded_control_task_mapping.jsonl")
    write_config(validation.model_dump(mode="json"), path / "control_suite_validation.json")
