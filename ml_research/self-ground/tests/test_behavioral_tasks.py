from __future__ import annotations

from self_ground.behavioral_tasks import (
    BehavioralTask,
    generate_behavioral_tasks,
    read_behavioral_tasks_jsonl,
    write_behavioral_tasks_jsonl,
)


def test_behavioral_task_generation_is_deterministic_and_stable() -> None:
    first = generate_behavioral_tasks(per_family=3, seed=7)
    second = generate_behavioral_tasks(per_family=3, seed=7)

    assert first == second
    assert [task.id for task in first[:3]] == [
        "sentiment_negation_2bd066e8d9e8",
        "sentiment_negation_8681908cb457",
        "sentiment_negation_929a30008f09",
    ]


def test_behavioral_tasks_cover_required_families_and_controls() -> None:
    tasks = generate_behavioral_tasks(per_family=2, seed=11)

    assert {task.family for task in tasks} == {
        "sentiment_negation",
        "property_negation",
        "state_negation",
    }
    assert len({task.id for task in tasks}) == len(tasks)
    for task in tasks:
        assert task.prompt
        assert task.control_prompt
        assert task.control_type == "matched_non_negation"
        assert task.target_tokens
        assert task.foil_tokens
        assert task.control_target_tokens
        assert task.control_foil_tokens
        assert not set(task.target_tokens) & set(task.foil_tokens)
        assert not set(task.control_target_tokens) & set(task.control_foil_tokens)
        assert task.metadata["template_family"] == task.family


def test_behavioral_tasks_jsonl_roundtrip(tmp_path) -> None:
    path = tmp_path / "tasks.jsonl"
    tasks = generate_behavioral_tasks(per_family=1, seed=7)

    write_behavioral_tasks_jsonl(tasks, path)
    loaded = read_behavioral_tasks_jsonl(path)

    assert [BehavioralTask.model_validate(task.model_dump()) for task in loaded] == tasks
