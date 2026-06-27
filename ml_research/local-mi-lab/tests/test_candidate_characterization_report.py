from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from local_mi_lab.candidate_characterization_report import (
    classify_final_characterization_status,
    compare_candidate_characterization_runs,
    render_characterization_markdown,
    summarize_multiseed_axes,
    summarize_multiseed_candidates,
)


def test_multiseed_merge_by_candidate() -> None:
    rows = pd.DataFrame(
        [
            _seed_row(20, "heldout_cand_000", 7, 7, "characterization_falsifies"),
            _seed_row(21, "heldout_cand_000", 7, 7, "characterization_downgrades"),
        ]
    )

    merged = summarize_multiseed_candidates(rows)

    row = merged.iloc[0]
    assert row["candidate_id"] == "heldout_cand_000"
    assert row["seeds_present"] == "20,21"
    assert row["n_downgrade_seeds"] == 1
    assert row["n_falsify_seeds"] == 1


def test_strengthened_classification() -> None:
    group = pd.DataFrame(
        [
            _seed_row(20, "c", 7, 7, "characterization_supports"),
            _seed_row(21, "c", 7, 7, "characterization_supports"),
            _seed_row(22, "c", 7, 7, "characterization_downgrades"),
        ]
    )

    assert classify_final_characterization_status(group) == "strengthened_local_candidate"


def test_downgraded_classification_when_diagnostics_missing() -> None:
    group = pd.DataFrame(
        [
            _seed_row(20, "c", 7, 7, "characterization_supports", ov="ov_weak"),
            _seed_row(21, "c", 7, 7, "characterization_supports", ov="ov_weak"),
            _seed_row(22, "c", 7, 7, "characterization_downgrades", ov="ov_weak"),
        ]
    )

    assert classify_final_characterization_status(group) == "downgraded_candidate"


def test_falsified_classification() -> None:
    group = pd.DataFrame(
        [
            _seed_row(20, "c", 7, 7, "characterization_falsifies"),
            _seed_row(21, "c", 7, 7, "characterization_falsifies"),
            _seed_row(22, "c", 7, 7, "characterization_downgrades"),
        ]
    )

    assert classify_final_characterization_status(group) == "falsified_candidate"


def test_negative_control_dominance_prevents_strengthened_status() -> None:
    group = pd.DataFrame(
        [
            _seed_row(20, "primary", 7, 7, "characterization_supports"),
            _seed_row(21, "primary", 7, 7, "characterization_supports"),
        ]
    )

    assert (
        classify_final_characterization_status(group, negative_control_dominates=True)
        == "falsified_candidate"
    )


def test_markdown_refuses_mechanism_claims() -> None:
    rows = pd.DataFrame(
        [
            _seed_row(20, "heldout_cand_000", 7, 7, "characterization_falsifies"),
            _seed_row(21, "heldout_cand_000", 7, 7, "characterization_falsifies"),
        ]
    )
    by_candidate = summarize_multiseed_candidates(rows)
    by_axis = summarize_multiseed_axes(rows)
    summary = {
        "executive_summary": "No prior candidate survived characterization.",
        "status_counts": {"falsified_candidate": 1},
        "strengthened_candidates": [],
    }

    markdown = render_characterization_markdown(summary, by_candidate, by_axis)

    assert "does not show an induction head" in markdown
    assert "circuit" in markdown
    assert "discovered an induction head" not in markdown


def test_report_includes_all_primary_heads(tmp_path: Path) -> None:
    run0 = _write_run(
        tmp_path,
        20,
        [
            _seed_row(20, "heldout_cand_000", 7, 7, "characterization_falsifies"),
            _seed_row(20, "heldout_cand_001", 9, 11, "characterization_falsifies"),
            _seed_row(20, "heldout_cand_002", 7, 11, "characterization_falsifies"),
            _seed_row(20, "heldout_cand_003", 7, 0, "characterization_falsifies"),
            _seed_row(20, "heldout_cand_004", 0, 8, "characterization_falsifies"),
        ],
    )
    run1 = _write_run(
        tmp_path,
        21,
        [
            _seed_row(21, "heldout_cand_000", 7, 7, "characterization_falsifies"),
            _seed_row(21, "heldout_cand_001", 9, 11, "characterization_falsifies"),
            _seed_row(21, "heldout_cand_002", 7, 11, "characterization_falsifies"),
            _seed_row(21, "heldout_cand_003", 7, 0, "characterization_falsifies"),
            _seed_row(21, "heldout_cand_004", 0, 8, "characterization_falsifies"),
        ],
    )

    paths = compare_candidate_characterization_runs(
        [run0, run1],
        tmp_path / "report",
        tracked_summary_path=tmp_path / "docs" / "summary.md",
    )

    markdown = paths["markdown"].read_text(encoding="utf-8")
    for label in ["L7H7", "L9H11", "L7H11", "L7H0", "L0H8"]:
        assert label in markdown
    assert paths["tracked_summary"].exists()
    assert paths["summary"].exists()


def _seed_row(
    seed: int,
    candidate_id: str,
    layer: int,
    head: int,
    status: str,
    *,
    group: str = "replicated_candidate",
    gap: float = 0.2,
    corr: float = 0.4,
    ov: str = "ov_supports_copy",
    qk: str = "qk_weak",
) -> dict[str, object]:
    return {
        "seed": seed,
        "candidate_id": candidate_id,
        "candidate_group": group,
        "layer": layer,
        "head": head,
        "mean_true_vs_control_effect": 0.3,
        "max_control_effect": 0.1,
        "positive_minus_control_gap": gap,
        "spearman_attention_effect_corr": corr,
        "mean_source_attention_margin": 0.1,
        "position_specificity_status": "destination_specific",
        "ov_copy_margin": 0.2 if ov == "ov_supports_copy" else 0.0,
        "ov_status": ov,
        "qk_source_margin": 0.0,
        "qk_status": qk,
        "characterization_seed_status": status,
    }


def _write_run(tmp_path: Path, seed: int, rows: list[dict[str, object]]) -> Path:
    root = tmp_path / f"seed{seed}"
    root.mkdir()
    pd.DataFrame(rows).to_csv(root / "candidate_characterization_by_candidate.csv", index=False)
    (root / "candidate_characterization_summary.json").write_text(
        json.dumps({"seed": seed}) + "\n",
        encoding="utf-8",
    )
    return root
