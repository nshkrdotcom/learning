from __future__ import annotations

from local_mi_lab.prompts import generate_induction_prompts, read_prompts_csv, write_prompts_csv


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
