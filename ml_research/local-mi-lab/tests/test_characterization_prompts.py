from __future__ import annotations

from pathlib import Path

from local_mi_lab.characterization_prompts import (
    CHAR_CONTROL_FAMILIES,
    CHAR_POSITIVE_FAMILIES,
    CHARACTERIZATION_FAMILIES,
    generate_characterization_prompts,
)
from local_mi_lab.prompts import read_prompts_csv, write_prompts_csv


def test_all_characterization_families_generated() -> None:
    records = generate_characterization_prompts(3, seed=20)

    assert {record.family for record in records} == set(CHARACTERIZATION_FAMILIES)
    assert len(records) == len(CHARACTERIZATION_FAMILIES) * 3


def test_family_counts_are_balanced() -> None:
    records = generate_characterization_prompts(4, seed=20)
    counts = {family: 0 for family in CHARACTERIZATION_FAMILIES}
    for record in records:
        counts[record.family] += 1

    assert set(counts.values()) == {4}


def test_domains_lengths_and_pairing_metadata() -> None:
    records = generate_characterization_prompts(2, seed=20)
    by_id = {record.example_id: record for record in records}

    for record in records:
        assert record.token_domain
        assert record.sequence_length_bucket
        assert record.characterization_axis
        if record.family in CHAR_POSITIVE_FAMILIES:
            assert record.should_show_induction_behavior is True
            assert record.paired_positive_example_id == record.example_id
            assert record.expected_source_position_hint is not None
            assert record.expected_source_position_hint < len(record.prompt_tokens_text)
        else:
            assert record.family in CHAR_CONTROL_FAMILIES
            assert record.should_show_induction_behavior is False
            assert record.paired_positive_example_id in by_id


def test_multi_distractor_has_valid_distractor_position() -> None:
    records = generate_characterization_prompts(
        2,
        families=["char_multi_distractor"],
        seed=20,
    )

    assert all(record.distractor_position_hint is not None for record in records)
    assert all(record.distractor_position_hint < len(record.prompt_tokens_text) for record in records)


def test_target_swap_control_differs_from_true_expected() -> None:
    records = generate_characterization_prompts(
        3,
        families=["char_target_swap_control"],
        seed=20,
    )

    assert all(record.expected_next_token != record.true_expected_next_token for record in records)


def test_characterization_prompts_are_deterministic() -> None:
    first = [record.to_dict() for record in generate_characterization_prompts(3, seed=21)]
    second = [record.to_dict() for record in generate_characterization_prompts(3, seed=21)]

    assert first == second


def test_characterization_csv_roundtrip_preserves_metadata(tmp_path: Path) -> None:
    records = generate_characterization_prompts(2, seed=20)
    path = tmp_path / "prompts.csv"

    write_prompts_csv(records, path)
    loaded = read_prompts_csv(path)

    assert loaded[0].distractor_position_hint == records[0].distractor_position_hint
    assert loaded[0].characterization_axis == records[0].characterization_axis
    assert loaded[0].sequence_length_bucket == records[0].sequence_length_bucket
    assert loaded[0].token_domain == records[0].token_domain
