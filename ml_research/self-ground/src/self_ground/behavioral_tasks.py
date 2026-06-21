from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from self_ground.io import read_jsonl, write_jsonl


class BehavioralTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    family: str
    concept: str
    prompt: str
    target_tokens: list[str]
    foil_tokens: list[str]
    control_prompt: str
    control_type: Literal["matched_non_negation"] = "matched_non_negation"
    control_target_tokens: list[str]
    control_foil_tokens: list[str]
    expected_baseline_direction: Literal["positive", "negative", "unknown"] = "unknown"
    metadata: dict[str, str | int | float | bool]


TASK_FAMILY_ORDER = [
    "sentiment_negation",
    "property_negation",
    "state_negation",
]


_TASK_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "sentiment_negation": [
        {
            "concept": "movie_good",
            "prompt": "The movie was not good. The movie was",
            "target_tokens": [" bad"],
            "foil_tokens": [" good"],
            "control_prompt": "The movie was good. The movie was",
            "control_target_tokens": [" good"],
            "control_foil_tokens": [" bad"],
        },
        {
            "concept": "review_great",
            "prompt": "The review was not great. The review was",
            "target_tokens": [" bad"],
            "foil_tokens": [" great"],
            "control_prompt": "The review was great. The review was",
            "control_target_tokens": [" great"],
            "control_foil_tokens": [" bad"],
        },
        {
            "concept": "meal_bad",
            "prompt": "The meal was not bad. The meal was",
            "target_tokens": [" good"],
            "foil_tokens": [" bad"],
            "control_prompt": "The meal was bad. The meal was",
            "control_target_tokens": [" bad"],
            "control_foil_tokens": [" good"],
        },
        {
            "concept": "show_fun",
            "prompt": "The show was not fun. The show was",
            "target_tokens": [" bad"],
            "foil_tokens": [" fun"],
            "control_prompt": "The show was fun. The show was",
            "control_target_tokens": [" fun"],
            "control_foil_tokens": [" bad"],
        },
    ],
    "property_negation": [
        {
            "concept": "animal_dangerous",
            "prompt": "The animal is not dangerous. The animal is",
            "target_tokens": [" safe"],
            "foil_tokens": [" dangerous"],
            "control_prompt": "The animal is dangerous. The animal is",
            "control_target_tokens": [" dangerous"],
            "control_foil_tokens": [" safe"],
        },
        {
            "concept": "task_simple",
            "prompt": "The task is not simple. The task is",
            "target_tokens": [" complex"],
            "foil_tokens": [" simple"],
            "control_prompt": "The task is simple. The task is",
            "control_target_tokens": [" simple"],
            "control_foil_tokens": [" complex"],
        },
        {
            "concept": "room_clean",
            "prompt": "The room is not clean. The room is",
            "target_tokens": [" dirty"],
            "foil_tokens": [" clean"],
            "control_prompt": "The room is clean. The room is",
            "control_target_tokens": [" clean"],
            "control_foil_tokens": [" dirty"],
        },
        {
            "concept": "route_safe",
            "prompt": "The route is not safe. The route is",
            "target_tokens": [" dangerous"],
            "foil_tokens": [" safe"],
            "control_prompt": "The route is safe. The route is",
            "control_target_tokens": [" safe"],
            "control_foil_tokens": [" dangerous"],
        },
    ],
    "state_negation": [
        {
            "concept": "switch_on",
            "prompt": "The switch is not on. The switch is",
            "target_tokens": [" off"],
            "foil_tokens": [" on"],
            "control_prompt": "The switch is on. The switch is",
            "control_target_tokens": [" on"],
            "control_foil_tokens": [" off"],
        },
        {
            "concept": "door_open",
            "prompt": "The door is not open. The door is",
            "target_tokens": [" closed"],
            "foil_tokens": [" open"],
            "control_prompt": "The door is open. The door is",
            "control_target_tokens": [" open"],
            "control_foil_tokens": [" closed"],
        },
        {
            "concept": "box_full",
            "prompt": "The box is not full. The box is",
            "target_tokens": [" empty"],
            "foil_tokens": [" full"],
            "control_prompt": "The box is full. The box is",
            "control_target_tokens": [" full"],
            "control_foil_tokens": [" empty"],
        },
        {
            "concept": "light_off",
            "prompt": "The light is not off. The light is",
            "target_tokens": [" on"],
            "foil_tokens": [" off"],
            "control_prompt": "The light is off. The light is",
            "control_target_tokens": [" off"],
            "control_foil_tokens": [" on"],
        },
    ],
}


def _stable_task_id(family: str, concept: str, prompt: str) -> str:
    digest = hashlib.sha256(f"{family}|{concept}|{prompt}".encode()).hexdigest()
    return f"{family}_{digest[:12]}"


def generate_behavioral_tasks(
    *,
    per_family: int = 10,
    seed: int = 7,
) -> list[BehavioralTask]:
    if per_family < 1:
        raise ValueError("per_family must be >= 1")
    rng = random.Random(seed)
    tasks: list[BehavioralTask] = []
    for family in TASK_FAMILY_ORDER:
        templates = list(_TASK_TEMPLATES[family])
        offset = rng.randrange(len(templates))
        ordered = templates[offset:] + templates[:offset]
        for idx in range(per_family):
            template = ordered[idx % len(ordered)]
            concept = str(template["concept"])
            prompt = str(template["prompt"])
            task_id = _stable_task_id(family, f"{concept}_{idx}", prompt)
            tasks.append(
                BehavioralTask(
                    id=task_id,
                    family=family,
                    concept=concept,
                    prompt=prompt,
                    target_tokens=list(template["target_tokens"]),
                    foil_tokens=list(template["foil_tokens"]),
                    control_prompt=str(template["control_prompt"]),
                    control_target_tokens=list(template["control_target_tokens"]),
                    control_foil_tokens=list(template["control_foil_tokens"]),
                    expected_baseline_direction="positive",
                    metadata={
                        "template_family": family,
                        "template_index": idx,
                        "source": "deterministic_phase3_templates",
                    },
                )
            )
    return tasks


def write_behavioral_tasks_jsonl(tasks: list[BehavioralTask], path: str | Path) -> None:
    write_jsonl(tasks, path)


def read_behavioral_tasks_jsonl(path: str | Path) -> list[BehavioralTask]:
    return [BehavioralTask.model_validate(row) for row in read_jsonl(path)]
