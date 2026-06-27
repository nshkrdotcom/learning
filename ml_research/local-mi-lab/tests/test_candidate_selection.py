from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from local_mi_lab.candidate_selection import (
    select_controlled_patching_candidates,
    write_candidate_artifacts,
)


def test_selects_top_raw_positive_candidates(tmp_path: Path) -> None:
    _write_attention_artifacts(tmp_path)
    candidates = select_controlled_patching_candidates(
        tmp_path,
        top_k_raw=1,
        top_k_control=0,
        top_k_gap=0,
        n_random=0,
    )
    assert candidates[0]["source"] == "top_raw_positive_attention"
    assert candidates[0]["layer"] == 0
    assert candidates[0]["head"] == 1


def test_selects_top_control_firing_candidates(tmp_path: Path) -> None:
    _write_attention_artifacts(tmp_path)
    candidates = select_controlled_patching_candidates(
        tmp_path,
        top_k_raw=0,
        top_k_control=1,
        top_k_gap=0,
        n_random=0,
    )
    assert candidates[0]["source"] == "top_control_firing_attention"
    assert candidates[0]["layer"] == 2
    assert candidates[0]["head"] == 3


def test_handles_zero_gap_honestly(tmp_path: Path) -> None:
    _write_attention_artifacts(tmp_path)
    candidates = select_controlled_patching_candidates(
        tmp_path,
        top_k_raw=1,
        top_k_control=0,
        top_k_gap=5,
        n_random=0,
    )
    assert all(not candidate["gap_positive"] for candidate in candidates)
    assert candidates[0]["expected_specificity"] == "nonspecific_attention_candidate"
    assert all(candidate["source"] != "top_positive_minus_control_gap" for candidate in candidates)


def test_deduplicates_repeated_layer_head_component(tmp_path: Path) -> None:
    _write_attention_artifacts(tmp_path, duplicate_control=True)
    candidates = select_controlled_patching_candidates(
        tmp_path,
        top_k_raw=1,
        top_k_control=1,
        top_k_gap=0,
        n_random=0,
    )
    keys = {(candidate["layer"], candidate["head"], candidate["component"]) for candidate in candidates}
    assert len(keys) == len(candidates)


def test_adds_random_comparison_candidates_deterministically(tmp_path: Path) -> None:
    _write_attention_artifacts(tmp_path)
    first = select_controlled_patching_candidates(
        tmp_path,
        top_k_raw=0,
        top_k_control=0,
        top_k_gap=0,
        n_random=2,
        seed=7,
    )
    second = select_controlled_patching_candidates(
        tmp_path,
        top_k_raw=0,
        top_k_control=0,
        top_k_gap=0,
        n_random=2,
        seed=7,
    )
    assert first == second
    assert {candidate["source"] for candidate in first} == {"random_comparison"}


def test_writes_candidate_csv_json_md(tmp_path: Path) -> None:
    _write_attention_artifacts(tmp_path)
    candidates = select_controlled_patching_candidates(tmp_path, n_random=1)
    paths = write_candidate_artifacts(
        tmp_path,
        candidates,
        top_k_raw=5,
        top_k_control=5,
        top_k_gap=5,
        n_random=1,
        seed=0,
    )
    assert paths["csv"].exists()
    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert "Raw attention candidates are not causal candidates" in paths["markdown"].read_text()
    payload = json.loads(paths["json"].read_text())
    assert payload["candidates"][0]["candidate_id"] == "cand_000"
    assert "candidate_id" in pd.read_csv(paths["csv"]).columns


def _write_attention_artifacts(tmp_path: Path, duplicate_control: bool = False) -> None:
    control_head = {"layer": 0, "head": 1} if duplicate_control else {"layer": 2, "head": 3}
    summary = {
        "top_heads_on_positive_examples": [
            {"layer": 0, "head": 1, "mean_attention_to_previous_occurrence": 0.5},
            {"layer": 1, "head": 2, "mean_attention_to_previous_occurrence": 0.4},
        ],
        "top_heads_on_controls": [
            {**control_head, "mean_attention_to_previous_occurrence": 0.6},
            {"layer": 1, "head": 2, "mean_attention_to_previous_occurrence": 0.4},
        ],
        "top_heads_by_positive_minus_control_gap": [
            {
                "layer": 0,
                "head": 1,
                "positive_mean_attention_to_previous_occurrence": 0.5,
                "max_control_mean_attention_to_previous_occurrence": 0.5,
                "max_control_family": "random_expected_token_control",
                "positive_minus_control_attention_gap": 0.0,
            }
        ],
    }
    (tmp_path / "attention_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(
        [
            _family_row("positive_repeat_sequence", 0, 1, 0.5),
            _family_row("random_expected_token_control", 0, 1, 0.5),
            _family_row("positive_repeat_sequence", 1, 2, 0.4),
            _family_row("distractor_repeat_control", 1, 2, 0.1),
            _family_row("positive_repeat_sequence", 2, 3, 0.2),
            _family_row("distractor_repeat_control", 2, 3, 0.6),
            _family_row("positive_repeat_sequence", 4, 4, 0.05),
            _family_row("distractor_repeat_control", 4, 4, 0.02),
        ]
    ).to_csv(tmp_path / "attention_by_family.csv", index=False)


def _family_row(family: str, layer: int, head: int, attention: float) -> dict[str, object]:
    return {
        "family": family,
        "layer": layer,
        "head": head,
        "n_examples": 2,
        "mean_attention_to_previous_occurrence": attention,
        "median_attention_to_previous_occurrence": attention,
        "mean_attention_to_bos": 0.1,
        "mean_attention_entropy": 1.0,
    }
