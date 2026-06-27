from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from local_mi_lab.publication_figures import generate_head_specific_publication_figures


def test_generates_vector_publication_figures(tmp_path: Path) -> None:
    report = tmp_path / "report"
    report.mkdir()
    run = tmp_path / "run0"
    run.mkdir()
    _write_multiseed_table(report)
    _write_seed_table(run)
    (report / "run_manifest.json").write_text(
        json.dumps({"runs": [{"seed": 0, "run_dir": str(run), "n_heads": 3}]}),
        encoding="utf-8",
    )
    (report / "head_specific_multiseed_summary.json").write_text(
        json.dumps(
            {
                "replicated_candidates": [
                    {"layer": 7, "head": 7},
                    {"layer": 9, "head": 11},
                ]
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "seed": 0,
                "family": "positive_repeat_sequence",
                "effect_size": 0.2,
            },
            {
                "seed": 0,
                "family": "same_token_frequency_control",
                "effect_size": -0.1,
            },
        ]
    ).to_csv(report / "replicated_head_L7H7_examples.csv", index=False)

    manifest = generate_head_specific_publication_figures(report, tmp_path / "figures")

    assert (tmp_path / "figures" / "figure_1_multiseed_candidate_gaps.svg").exists()
    assert (tmp_path / "figures" / "figure_1_multiseed_candidate_gaps.pdf").exists()
    assert (tmp_path / "figures" / "manifest.json").exists()
    assert manifest["primary_metric"] == "true_vs_control_logit_diff"


def _write_multiseed_table(report: Path) -> None:
    pd.DataFrame(
        [
            {
                "layer": 7,
                "head": 7,
                "mean_positive_minus_control_gap": 0.08,
                "replication_status": "replicated_head_specific_candidate",
                "raw_attention_candidate_in_any_seed": False,
                "random_comparison_candidate_in_any_seed": True,
            },
            {
                "layer": 9,
                "head": 11,
                "mean_positive_minus_control_gap": 0.03,
                "replication_status": "replicated_head_specific_candidate",
                "raw_attention_candidate_in_any_seed": False,
                "random_comparison_candidate_in_any_seed": False,
            },
            {
                "layer": 0,
                "head": 1,
                "mean_positive_minus_control_gap": -0.13,
                "replication_status": "no_effect",
                "raw_attention_candidate_in_any_seed": True,
                "random_comparison_candidate_in_any_seed": False,
            },
        ]
    ).to_csv(report / "head_specific_multiseed_by_head.csv", index=False)


def _write_seed_table(run: Path) -> None:
    pd.DataFrame(
        [
            {"seed": 0, "specificity_status": "head_specific_positive_candidate"},
            {"seed": 0, "specificity_status": "nonspecific_moves_controls"},
            {"seed": 0, "specificity_status": "no_positive_effect"},
        ]
    ).to_csv(run / "head_specific_patching_by_head.csv", index=False)
