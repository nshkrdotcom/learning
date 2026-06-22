from __future__ import annotations

import torch

from self_ground.behavioral_tasks import BehavioralTask
from self_ground.control_suites import build_control_cases


class Tokenizer:
    def to_tokens(self, text: str, prepend_bos: bool = False):
        del prepend_bos
        mapping = {
            " bad": [1],
            " good": [2],
            " safe": [3],
            " dangerous": [4],
            " off": [5],
            " on": [6],
            " light": [7],
            " heavy": [8],
        }
        return torch.tensor(mapping[text])


class Adapter:
    model = Tokenizer()


def _task(idx: int, family: str, target: str = " bad", foil: str = " good") -> BehavioralTask:
    return BehavioralTask(
        id=f"{family}_{idx}",
        family=family,
        concept=f"concept_{idx}",
        prompt=f"The item {idx} was not good. The item was",
        target_tokens=[target],
        foil_tokens=[foil],
        control_prompt=f"The item {idx} was good. The item was",
        control_type="matched_non_negation",
        control_target_tokens=[foil],
        control_foil_tokens=[target],
        expected_baseline_direction="positive",
        metadata={"template_family": family, "template_id": f"{family}_{idx}"},
    )


def _tasks() -> list[BehavioralTask]:
    rows = []
    for family in ["sentiment_negation", "property_negation", "state_negation"]:
        rows.append(_task(0, family))
        rows.append(_task(1, family, target=" safe", foil=" dangerous"))
    return rows


def test_default_control_suite_preserves_current_controls() -> None:
    cases, validation, excluded = build_control_cases(
        tasks=_tasks(),
        model_adapter=Adapter(),
        mode="matched_non_negation_current",
    )

    assert validation.passes_minimum is True
    assert not excluded
    assert len(cases) == 6
    assert {case.control_suite for case in cases} == {"matched_non_negation_current"}
    assert cases[0].control_type == "matched_non_negation"


def test_multi_control_expands_each_task_across_suites() -> None:
    tasks = _tasks()
    cases, validation, excluded = build_control_cases(
        tasks=tasks,
        model_adapter=Adapter(),
        mode="multi_control",
    )

    assert validation.passes_minimum is True
    assert not excluded
    assert len(cases) == len(tasks) * 5
    assert set(validation.valid_cases_by_suite) == {
        "matched_non_negation_current",
        "lexical_identity_control",
        "semantic_unrelated_control",
        "shuffled_target_control",
        "hard_negative_control",
    }


def test_shuffled_control_fails_closed_without_same_family_alternate() -> None:
    tasks = [
        _task(0, "sentiment_negation"),
        _task(0, "property_negation"),
        _task(0, "state_negation"),
    ]

    cases, validation, excluded = build_control_cases(
        tasks=tasks,
        model_adapter=Adapter(),
        mode="shuffled_target_control",
    )

    assert cases == []
    assert validation.passes_minimum is False
    assert len(excluded) == 3
