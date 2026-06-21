from __future__ import annotations

import json

from self_ground.behavioral_tasks import BehavioralTask
from self_ground.task_validation import validate_behavioral_tasks


class TinyTokenizerModel:
    def to_tokens(self, text: str, prepend_bos: bool = False):
        del prepend_bos
        mapping = {
            " good": [1],
            " bad": [2],
            " safe": [3],
            " dangerous": [4],
            " multi token": [5, 6],
        }
        return TinyTokenTensor(mapping.get(text, [99]))


class TinyTokenTensor:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def flatten(self):
        return self

    def numel(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> int:
        return self.values[index]


class TinyTokenizerAdapter:
    model = TinyTokenizerModel()


def _task(
    *,
    task_id: str = "task",
    family: str = "sentiment_negation",
    target_tokens: list[str] | None = None,
    foil_tokens: list[str] | None = None,
    control_type: str = "matched_non_negation",
) -> BehavioralTask:
    return BehavioralTask(
        id=task_id,
        family=family,
        concept="movie",
        prompt="The movie was not good. The movie was",
        target_tokens=target_tokens or [" bad"],
        foil_tokens=foil_tokens or [" good"],
        control_prompt="The movie was good. The movie was",
        control_type=control_type,  # type: ignore[arg-type]
        control_target_tokens=[" good"],
        control_foil_tokens=[" bad"],
        expected_baseline_direction="positive",
        metadata={"template_family": family},
    )


def test_valid_single_token_task_passes() -> None:
    valid, results, summary = validate_behavioral_tasks(
        model_adapter=TinyTokenizerAdapter(),
        tasks=[_task()],
        min_valid_tasks_per_family=1,
    )

    assert [task.id for task in valid] == ["task"]
    assert results[0].valid is True
    assert summary.valid_tasks == 1
    assert summary.passes_minimum is True


def test_multi_token_and_overlap_are_excluded_with_reasons() -> None:
    tasks = [
        _task(task_id="multi", target_tokens=[" multi token"]),
        _task(task_id="overlap", target_tokens=[" good"], foil_tokens=[" good"]),
    ]

    valid, results, summary = validate_behavioral_tasks(
        model_adapter=TinyTokenizerAdapter(),
        tasks=tasks,
        min_valid_tasks_per_family=1,
    )

    assert valid == []
    reasons = {result.task_id: result.excluded_reason for result in results}
    assert "exactly one token" in reasons["multi"]
    assert "overlap" in reasons["overlap"]
    assert summary.excluded_tasks == 2
    assert summary.passes_minimum is False


def test_wrong_control_type_fails_validation() -> None:
    task = _task().model_copy(update={"control_type": "unrelated"})

    valid, results, _ = validate_behavioral_tasks(
        model_adapter=TinyTokenizerAdapter(),
        tasks=[task],
        min_valid_tasks_per_family=1,
    )

    assert valid == []
    assert results[0].valid is False
    assert "matched_non_negation" in results[0].excluded_reason


def test_validation_summary_serializes_cleanly() -> None:
    _, _, summary = validate_behavioral_tasks(
        model_adapter=TinyTokenizerAdapter(),
        tasks=[_task()],
        min_valid_tasks_per_family=1,
    )

    assert json.loads(summary.model_dump_json())["valid_tasks"] == 1
