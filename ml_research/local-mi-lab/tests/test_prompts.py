from __future__ import annotations

from local_mi_lab.prompts import (
    CONTROL_FAMILIES,
    generate_induction_control_prompts,
    generate_induction_prompts,
    read_prompts_csv,
    write_prompts_csv,
)


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


def test_induction_prompts_include_source_position_metadata() -> None:
    records = generate_induction_prompts(n_examples=16, seed=0)
    for record in records:
        assert record.prompt_tokens_text
        assert record.sequence_tokens
        assert record.repeated_prefix_length is not None
        assert record.target_position_label == "final"
        assert record.expected_source_position_hint is not None
        assert record.prompt_tokens_text[record.expected_source_position_hint] == (
            record.expected_source_token
        )


def test_prompt_metadata_survives_csv_roundtrip(tmp_path) -> None:
    path = tmp_path / "prompts.csv"
    records = generate_induction_prompts(n_examples=4, seed=0)
    write_prompts_csv(records, path)
    loaded = read_prompts_csv(path)
    assert [record.to_dict() for record in loaded] == [record.to_dict() for record in records]


def test_controlled_induction_families_are_generated() -> None:
    records = generate_induction_control_prompts(n_examples_per_family=3, seed=0)
    assert sorted({record.family for record in records}) == sorted(CONTROL_FAMILIES)


def test_controlled_family_counts_are_balanced() -> None:
    records = generate_induction_control_prompts(n_examples_per_family=4, seed=0)
    counts = {family: 0 for family in CONTROL_FAMILIES}
    for record in records:
        counts[record.family] += 1
    assert set(counts.values()) == {4}


def test_positive_examples_have_source_metadata_and_controls_are_labeled() -> None:
    records = generate_induction_control_prompts(n_examples_per_family=2, seed=0)
    for record in records:
        if record.family == "positive_repeat_sequence":
            assert record.is_positive_induction_example is True
            assert record.should_show_induction_behavior is True
            assert record.expected_source_position_hint is not None
            assert record.expected_source_token
        else:
            assert record.is_positive_induction_example is False
            assert record.should_show_induction_behavior is False
            assert record.control_family == record.family


def test_controls_have_correct_or_null_source_metadata() -> None:
    records = generate_induction_control_prompts(n_examples_per_family=1, seed=0)
    by_family = {record.family: record for record in records}
    assert by_family["no_repeat_control"].expected_source_position_hint is None
    assert by_family["no_repeat_control"].expected_source_token == ""
    for family in [
        "shuffled_repeat_control",
        "distractor_repeat_control",
        "same_token_frequency_control",
        "random_expected_token_control",
    ]:
        record = by_family[family]
        assert record.expected_source_position_hint is not None
        assert record.prompt_tokens_text[record.expected_source_position_hint] == (
            record.expected_source_token
        )


def test_random_expected_token_controls_differ_from_true_expected_tokens() -> None:
    records = generate_induction_control_prompts(n_examples_per_family=3, seed=0)
    random_records = [r for r in records if r.family == "random_expected_token_control"]
    assert random_records
    for record in random_records:
        assert record.expected_next_token == " X"
        assert record.expected_next_token != f" {record.sequence_tokens[-1]}"


def test_control_metadata_survives_csv_roundtrip(tmp_path) -> None:
    path = tmp_path / "control_prompts.csv"
    records = generate_induction_control_prompts(n_examples_per_family=2, seed=0)
    write_prompts_csv(records, path)
    loaded = read_prompts_csv(path)
    assert [record.to_dict() for record in loaded] == [record.to_dict() for record in records]
