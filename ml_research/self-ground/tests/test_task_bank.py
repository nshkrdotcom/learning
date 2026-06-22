from __future__ import annotations

import torch

from self_ground.behavioral_tasks import TASK_FAMILY_ORDER
from self_ground.task_bank import (
    CandidateTaskTemplate,
    build_candidate_task_bank,
    generate_candidate_templates,
    validate_candidate_template_tokens,
)


class AnySingleTokenModel:
    def to_tokens(self, text: str, prepend_bos: bool = False):
        del prepend_bos
        if text == " multi token":
            return torch.tensor([1, 2])
        return torch.tensor([abs(hash(text)) % 1000 + 1])


class AnySingleTokenAdapter:
    model = AnySingleTokenModel()


def test_task_bank_generation_is_deterministic_and_covers_required_families() -> None:
    first = generate_candidate_templates(per_family_candidates=12)
    second = generate_candidate_templates(per_family_candidates=12)

    assert [row.model_dump() for row in first] == [row.model_dump() for row in second]
    families = {row.family for row in first}
    assert families == set(TASK_FAMILY_ORDER)
    assert all(row.target_token.startswith(" ") for row in first)
    assert all(row.foil_token.startswith(" ") for row in first)


def test_task_bank_filters_multitoken_targets() -> None:
    ok, rejection = validate_candidate_template_tokens(
        model_adapter=AnySingleTokenAdapter(),
        template=CandidateTaskTemplate(
            template_id="bad_token",
            family="property_negation",
            prompt_template="The object was not heavy. The object was",
            target_token=" multi token",
            foil_token=" heavy",
        ),
    )

    assert ok is False
    assert rejection is not None
    assert rejection["reason"] == "tokenization_failed"
    assert rejection["field"] == "target_token"


def test_task_bank_property_family_has_large_surviving_pool_with_valid_tokenizer() -> None:
    bank, tasks, rejections = build_candidate_task_bank(
        model_adapter=AnySingleTokenAdapter(),
        model_name="test-local",
        per_family_candidates=80,
    )

    assert rejections == []
    assert bank.metadata["accepted_by_family"]["property_negation"] >= 50
    assert {task.family for task in tasks} == set(TASK_FAMILY_ORDER)
