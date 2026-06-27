from __future__ import annotations

from pathlib import Path

import pytest

from local_mi_lab.heldout_prompts import (
    HELDOUT_FAMILIES,
    generate_heldout_induction_prompts,
    validate_heldout_expected_tokens,
)
from local_mi_lab.prompts import read_prompts_csv, write_prompts_csv


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == " bad multi":
            return [1, 2]
        if not text.startswith(" "):
            return [99, 100]
        return [abs(hash(text)) % 1000]


def test_all_heldout_families_generated_balanced() -> None:
    records = generate_heldout_induction_prompts(n_examples_per_family=3, seed=10)
    counts = {family: 0 for family in HELDOUT_FAMILIES}
    for record in records:
        counts[record.family] += 1
    assert counts == {family: 3 for family in HELDOUT_FAMILIES}


def test_positive_and_control_pairing_metadata_valid() -> None:
    records = generate_heldout_induction_prompts(n_examples_per_family=4, seed=10)
    by_id = {record.example_id: record for record in records}
    for record in records:
        assert record.base_sequence_id
        assert record.family_index is not None
        if record.should_show_induction_behavior:
            assert record.is_positive_induction_example
            assert record.expected_source_token
            assert record.expected_source_position_hint is not None
            assert record.paired_positive_example_id == record.example_id
        else:
            assert not record.is_positive_induction_example
            assert record.paired_positive_example_id in by_id
            assert by_id[record.paired_positive_example_id].should_show_induction_behavior


def test_wrong_target_controls_differ_from_true_expected_token() -> None:
    records = generate_heldout_induction_prompts(
        n_examples_per_family=5,
        families=["heldout_wrong_target_same_prompt"],
        seed=10,
    )
    assert records
    for record in records:
        assert record.expected_next_token != record.true_expected_next_token
        assert record.wrong_or_control_token == record.expected_next_token
        assert not record.should_show_induction_behavior


def test_no_structure_controls_are_marked_as_controls() -> None:
    records = generate_heldout_induction_prompts(
        n_examples_per_family=5,
        families=["heldout_no_structure_same_tokens"],
        seed=10,
    )
    for record in records:
        assert record.heldout_family_type == "control"
        assert record.expected_source_position_hint is None
        assert not record.should_show_induction_behavior


def test_csv_roundtrip_preserves_heldout_metadata(tmp_path: Path) -> None:
    records = generate_heldout_induction_prompts(n_examples_per_family=2, seed=11)
    path = tmp_path / "prompts.csv"
    write_prompts_csv(records, path)
    loaded = read_prompts_csv(path)
    assert loaded[0].heldout_family_type == records[0].heldout_family_type
    assert loaded[0].heldout_construction_note == records[0].heldout_construction_note
    assert loaded[-1].paired_positive_example_id == records[-1].paired_positive_example_id


def test_generation_is_deterministic_under_seed() -> None:
    first = generate_heldout_induction_prompts(n_examples_per_family=4, seed=12)
    second = generate_heldout_induction_prompts(n_examples_per_family=4, seed=12)
    assert [record.to_dict() for record in first] == [record.to_dict() for record in second]


def test_expected_token_validation_boundary() -> None:
    records = generate_heldout_induction_prompts(n_examples_per_family=1, seed=10)
    validate_heldout_expected_tokens(records, FakeTokenizer())
    bad = [records[0]]
    bad[0] = bad[0].__class__(**{**bad[0].to_dict(), "expected_next_token": " bad multi"})
    with pytest.raises(ValueError):
        validate_heldout_expected_tokens(bad, FakeTokenizer())
