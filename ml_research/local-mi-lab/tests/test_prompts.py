from __future__ import annotations

from local_mi_lab.prompts import generate_induction_prompts


def test_induction_prompts_are_deterministic() -> None:
    first = generate_induction_prompts(n_examples=8, seed=0)
    second = generate_induction_prompts(n_examples=8, seed=0)
    assert [record.to_dict() for record in first] == [record.to_dict() for record in second]


def test_each_prompt_has_expected_next_token() -> None:
    records = generate_induction_prompts(n_examples=16, seed=0)
    assert all(record.expected_next_token for record in records)
    assert all(record.expected_next_token.startswith(" ") for record in records)


def test_controls_are_present() -> None:
    records = generate_induction_prompts(n_examples=16, seed=0)
    assert all(record.control_prompt for record in records)
    assert any(record.prompt != record.control_prompt for record in records)
