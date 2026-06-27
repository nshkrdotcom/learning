from __future__ import annotations

import json
from pathlib import Path

from local_mi_lab.attention import ATTENTION_LIMITATION
from local_mi_lab.report import MANDATORY_LANGUAGE, generate_run_summary


def test_summary_generation_from_fixture_artifacts() -> None:
    fixture = Path(__file__).parent / "fixtures" / "small_run"
    summary = generate_run_summary(fixture)
    assert "# Run Summary" in summary
    assert "Examples: 2" in summary
    assert "## Baseline behavior by family" in summary
    assert "Selected activation cache present" in summary
    assert "## Logit lens by family" in summary
    assert "## Attention patterns" in summary
    assert "## Attention controls" in summary
    assert "Top induction-like attention pattern candidates" in summary
    assert "Patching results present" in summary


def test_missing_artifact_handling(tmp_path: Path) -> None:
    summary = generate_run_summary(tmp_path)
    assert "Missing: `baseline_metrics.json` was not found." in summary
    assert "Missing: `baseline_by_family.csv` was not found." in summary
    assert "Missing: activation manifest was not found." in summary
    assert "Missing: `attention_by_family.csv` was not found." in summary
    assert "Missing: `attention_summary.json` was not found." in summary
    assert "Missing: `patching_results.csv` was not found." in summary
    assert "Build induction_controls prompts." in summary


def test_summary_has_no_mechanism_overclaiming() -> None:
    fixture = Path(__file__).parent / "fixtures" / "small_run"
    summary = generate_run_summary(fixture)
    for sentence in MANDATORY_LANGUAGE:
        assert sentence in summary
    assert ATTENTION_LIMITATION in summary
    assert "raw positive attention alone" in summary
    assert "not a specific induction-head candidate" in summary
    assert "proves a mechanism" not in summary.lower()


def test_missing_candidate_selection_next_step(tmp_path: Path) -> None:
    _write_control_ready_artifacts(tmp_path)
    summary = generate_run_summary(tmp_path)
    assert "Run select_controlled_patching_candidates.py, then run_controlled_patching.py." in summary


def test_candidate_selection_without_patching_next_step(tmp_path: Path) -> None:
    _write_control_ready_artifacts(tmp_path)
    (tmp_path / "controlled_patching_candidates.csv").write_text(
        "candidate_id,source,layer,head,component\ncand_0,top_raw_positive_attention,0,1,attn_out\n",
        encoding="utf-8",
    )
    summary = generate_run_summary(tmp_path)
    assert "Run controlled patching on selected candidates." in summary


def test_report_describes_nonspecific_controlled_patching(tmp_path: Path) -> None:
    _write_control_ready_artifacts(tmp_path)
    _write_candidate_selection(tmp_path)
    _write_controlled_patching_summary(
        tmp_path,
        specificity_status_counts={"nonspecific_moves_controls": 2},
        positive_mean_effect_size=0.4,
        max_control_mean_effect_size=0.5,
        best_gap=-0.1,
    )
    summary = generate_run_summary(tmp_path)
    assert "A candidate that moves controls as much as positives is nonspecific." in summary
    assert "nonspecific causal pattern" in summary
    assert "Write a learning note explaining the false-positive pattern before adding new tasks." in summary
    assert "Layer-level attn_out patching is not head-specific unless the artifact explicitly says" in summary


def test_report_describes_positive_specific_replication_step(tmp_path: Path) -> None:
    _write_control_ready_artifacts(tmp_path)
    _write_candidate_selection(tmp_path)
    _write_controlled_patching_summary(
        tmp_path,
        specificity_status_counts={"positive_specific_candidate": 1},
        positive_mean_effect_size=0.6,
        max_control_mean_effect_size=0.2,
        best_gap=0.4,
    )
    summary = generate_run_summary(tmp_path)
    assert "positive-specific candidate" in summary
    assert "run a smaller replication with a new seed" in summary
    assert "not a mechanism claim" in summary


def _write_control_ready_artifacts(root: Path) -> None:
    (root / "prompts.csv").write_text("example_id,prompt\nexample_0,A B C\n", encoding="utf-8")
    (root / "baseline_by_family.csv").write_text("family,n_examples\npositive_repeat_sequence,1\n", encoding="utf-8")
    (root / "attention_by_family.csv").write_text("family,layer,head\npositive_repeat_sequence,0,1\n", encoding="utf-8")
    (root / "logit_lens_by_family.csv").write_text("family,layer\npositive_repeat_sequence,0\n", encoding="utf-8")


def _write_candidate_selection(root: Path) -> None:
    (root / "controlled_patching_candidates.csv").write_text(
        "candidate_id,source,layer,head,component\ncand_0,top_raw_positive_attention,0,1,attn_out\n",
        encoding="utf-8",
    )


def _write_controlled_patching_summary(
    root: Path,
    *,
    specificity_status_counts: dict[str, int],
    positive_mean_effect_size: float,
    max_control_mean_effect_size: float,
    best_gap: float,
) -> None:
    (root / "controlled_patching_summary.json").write_text(
        json.dumps(
            {
                "n_candidates": 1,
                "positive_mean_effect_size": positive_mean_effect_size,
                "max_control_mean_effect_size": max_control_mean_effect_size,
                "best_positive_minus_control_effect_gap": best_gap,
                "specificity_status_counts": specificity_status_counts,
            }
        ),
        encoding="utf-8",
    )
    (root / "controlled_patching_by_family.csv").write_text(
        "family,candidate_id,mean_effect_size\npositive_repeat_sequence,cand_0,0.4\n",
        encoding="utf-8",
    )
    (root / "controlled_patching_by_candidate.csv").write_text(
        "candidate_id,specificity_status\ncand_0,nonspecific_moves_controls\n",
        encoding="utf-8",
    )
