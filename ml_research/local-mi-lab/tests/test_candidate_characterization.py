from __future__ import annotations

from pathlib import Path

import pandas as pd

from local_mi_lab.candidate_characterization import (
    classify_characterization_seed_status,
    summarize_candidate_characterization,
)


def test_support_downgrade_falsification_classification() -> None:
    support = {
        "positive_minus_control_gap": 0.2,
        "mean_true_vs_control_effect": 0.3,
        "max_control_effect": 0.1,
        "spearman_attention_effect_corr": 0.4,
        "position_specificity_status": "destination_specific",
        "ov_status": "ov_supports_copy",
        "qk_status": "qk_weak",
    }
    downgrade = {**support, "ov_status": "ov_weak", "qk_status": "qk_weak"}
    falsify = {**support, "max_control_effect": 0.4}

    assert classify_characterization_seed_status(support) == "characterization_supports"
    assert classify_characterization_seed_status(downgrade) == "characterization_downgrades"
    assert classify_characterization_seed_status(falsify) == "characterization_falsifies"


def test_summary_merges_diagnostics() -> None:
    results = pd.DataFrame(
        [
            _result("primary", "positive", 0.3),
            _result("primary", "control", 0.1),
        ]
    )
    attention = pd.DataFrame(
        [
            {
                "candidate_id": "primary",
                "spearman_attention_effect_corr": 0.5,
                "mean_source_attention_margin": 0.2,
            }
        ]
    )
    position = pd.DataFrame(
        [{"candidate_id": "primary", "position_specificity_status": "destination_specific"}]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "candidate_id": "primary",
                "ov_copy_margin": 0.1,
                "ov_status": "ov_supports_copy",
                "qk_source_margin": -0.1,
                "qk_status": "qk_weak",
            }
        ]
    )

    summary = summarize_candidate_characterization(results, attention, position, diagnostics)

    assert summary.loc[0, "characterization_seed_status"] == "characterization_supports"
    assert summary.loc[0, "candidate_id"] == "primary"


def test_negative_controls_are_retained() -> None:
    results = pd.DataFrame(
        [
            _result("negative", "positive", 0.0, group="negative_control_no_effect"),
            _result("negative", "control", 0.0, group="negative_control_no_effect"),
        ]
    )
    attention = pd.DataFrame(
        [{"candidate_id": "negative", "spearman_attention_effect_corr": None, "mean_source_attention_margin": None}]
    )
    position = pd.DataFrame(
        [{"candidate_id": "negative", "position_specificity_status": "no_position_effect"}]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "candidate_id": "negative",
                "ov_copy_margin": -0.2,
                "ov_status": "ov_contradicts_copy",
                "qk_source_margin": -0.2,
                "qk_status": "qk_contradicts_source_selection",
            }
        ]
    )

    summary = summarize_candidate_characterization(results, attention, position, diagnostics)

    assert summary.loc[0, "candidate_group"] == "negative_control_no_effect"
    assert summary.loc[0, "characterization_seed_status"] == "characterization_falsifies"


def test_expected_output_paths_documented(tmp_path: Path) -> None:
    root = tmp_path / "seed20"
    expected = [
        root / "prompts.csv",
        root / "candidate_characterization_results.csv",
        root / "attention_effect_alignment",
        root / "position_characterization",
        root / "head_circuit_diagnostics",
        root / "candidate_characterization_summary.json",
        root / "candidate_characterization.md",
    ]

    assert len(expected) == 7


def _result(
    candidate_id: str,
    family_type: str,
    effect: float,
    group: str = "replicated_candidate",
) -> dict[str, object]:
    return {
        "seed": 20,
        "candidate_id": candidate_id,
        "candidate_group": group,
        "layer": 7,
        "head": 7,
        "family": f"{family_type}_family",
        "heldout_family_type": family_type,
        "example_id": f"{candidate_id}_{family_type}",
        "intervention": "head_clean_to_corrupt_patch",
        "position_label": "final",
        "effect_size": effect,
        "effect_size_status": "ok",
    }
