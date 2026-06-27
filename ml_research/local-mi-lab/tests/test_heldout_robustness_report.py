from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from local_mi_lab.heldout_robustness_report import (
    classify_heldout_replication_status,
    compare_heldout_runs,
    render_heldout_markdown,
    summarize_heldout_candidates,
    summarize_heldout_families,
)


def test_multiseed_heldout_merge() -> None:
    candidates = pd.DataFrame(
        [
            _candidate_row(10, "heldout_cand_000", 7, 7, "heldout_survives_seed"),
            _candidate_row(11, "heldout_cand_000", 7, 7, "heldout_survives_seed"),
        ]
    )
    families = pd.DataFrame(
        [
            _family_row(10, "heldout_cand_000", "heldout_symbolic_longer"),
            _family_row(11, "heldout_cand_000", "heldout_word_sequences"),
        ]
    )

    merged = summarize_heldout_candidates(candidates, families)

    row = merged.iloc[0]
    assert row["layer"] == 7
    assert row["head"] == 7
    assert row["seeds_present"] == "10,11"


def test_heldout_replicated_classification() -> None:
    group = pd.DataFrame(
        [
            _candidate_row(10, "c", 7, 7, "heldout_survives_seed"),
            _candidate_row(
                10,
                "c",
                7,
                7,
                "heldout_survives_seed",
                intervention="head_zero_ablation",
                position="previous_occurrence",
            ),
            _candidate_row(11, "c", 7, 7, "heldout_survives_seed"),
            _candidate_row(
                11,
                "c",
                7,
                7,
                "heldout_survives_seed",
                intervention="head_zero_ablation",
                position="previous_occurrence",
            ),
            _candidate_row(12, "c", 7, 7, "falsified_no_positive_effect"),
        ]
    )
    assert (
        classify_heldout_replication_status(
            group,
            {"heldout_symbolic_longer", "heldout_word_sequences"},
        )
        == "heldout_replicated"
    )


def test_heldout_downgraded_classification() -> None:
    group = pd.DataFrame(
        [
            _candidate_row(10, "c", 7, 7, "heldout_survives_seed"),
            _candidate_row(11, "c", 7, 7, "heldout_survives_seed"),
            _candidate_row(12, "c", 7, 7, "falsified_no_positive_effect"),
        ]
    )
    assert (
        classify_heldout_replication_status(
            group,
            {"heldout_symbolic_longer", "heldout_word_sequences"},
        )
        == "heldout_downgraded"
    )


def test_heldout_falsified_classification() -> None:
    group = pd.DataFrame(
        [
            _candidate_row(10, "c", 7, 7, "falsified_no_positive_effect"),
            _candidate_row(11, "c", 7, 7, "falsified_no_positive_effect"),
            _candidate_row(12, "c", 7, 7, "falsified_sign_flip"),
        ]
    )
    assert classify_heldout_replication_status(group, set()) == "heldout_falsified"


def test_controls_moving_dominates_survival() -> None:
    group = pd.DataFrame(
        [
            _candidate_row(10, "c", 7, 7, "heldout_survives_seed"),
            _candidate_row(11, "c", 7, 7, "heldout_survives_seed"),
            _candidate_row(12, "c", 7, 7, "falsified_controls_move"),
        ]
    )
    assert (
        classify_heldout_replication_status(
            group,
            {"heldout_symbolic_longer", "heldout_word_sequences"},
        )
        == "heldout_falsified"
    )


def test_report_refuses_mechanism_claims() -> None:
    candidates = pd.DataFrame(
        [
            _candidate_row(10, "heldout_cand_000", 7, 7, "heldout_survives_seed"),
            _candidate_row(
                10,
                "heldout_cand_000",
                7,
                7,
                "heldout_survives_seed",
                intervention="head_zero_ablation",
                position="previous_occurrence",
            ),
            _candidate_row(11, "heldout_cand_000", 7, 7, "heldout_survives_seed"),
            _candidate_row(
                11,
                "heldout_cand_000",
                7,
                7,
                "heldout_survives_seed",
                intervention="head_zero_ablation",
                position="previous_occurrence",
            ),
        ]
    )
    families = pd.DataFrame(
        [
            _family_row(10, "heldout_cand_000", "heldout_symbolic_longer"),
            _family_row(11, "heldout_cand_000", "heldout_word_sequences"),
        ]
    )
    by_candidate = summarize_heldout_candidates(candidates, families)
    by_family = summarize_heldout_families(families)
    summary = {
        "seeds": [10, 11],
        "status_counts": {"heldout_replicated": 1},
        "replicated_candidates": by_candidate.to_dict("records"),
        "primary_candidate_outcomes": by_candidate.to_dict("records"),
        "prior_raw_attention_outcomes": [],
        "negative_control_outcomes": [],
        "executive_summary": "A narrow local candidate survived.",
    }

    markdown = render_heldout_markdown(summary, by_candidate, by_family)

    assert "not a mechanism claim" in markdown
    assert "does not discover an induction head" in markdown
    assert "discovered an induction head" not in markdown


def test_negative_controls_reported_separately(tmp_path: Path) -> None:
    run0 = _write_run(
        tmp_path,
        seed=10,
        rows=[
            _candidate_row(10, "primary", 7, 7, "heldout_survives_seed"),
            _candidate_row(
                10,
                "negative",
                11,
                0,
                "heldout_survives_seed",
                group="negative_control_no_effect",
            ),
        ],
    )
    run1 = _write_run(
        tmp_path,
        seed=11,
        rows=[
            _candidate_row(11, "primary", 7, 7, "heldout_survives_seed"),
            _candidate_row(
                11,
                "negative",
                11,
                0,
                "falsified_no_positive_effect",
                group="negative_control_no_effect",
            ),
        ],
    )

    paths = compare_heldout_runs([run0, run1], tmp_path / "report")

    payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert payload["negative_control_outcomes"]
    assert paths["markdown"].exists()


def _candidate_row(
    seed: int,
    candidate_id: str,
    layer: int,
    head: int,
    status: str,
    *,
    group: str = "replicated_candidate",
    intervention: str = "head_clean_to_corrupt_patch",
    position: str = "final",
    head_specific: bool = True,
) -> dict[str, object]:
    return {
        "seed": seed,
        "candidate_id": candidate_id,
        "candidate_group": group,
        "layer": layer,
        "head": head,
        "intervention": intervention,
        "position_label": position,
        "head_specific_patch": head_specific,
        "actual_patch_scope": "single_head_z" if head_specific else "full_attn_out_layer",
        "metric": "true_vs_control_logit_diff",
        "n_families": 6,
        "n_examples": 72,
        "positive_family_mean_effect": 0.1 if status != "falsified_no_positive_effect" else -0.1,
        "max_control_family_mean_effect": -0.1,
        "positive_minus_control_gap": 0.2,
        "n_positive_families_with_gap_gt_0": 2,
        "n_control_families_moving": 0,
        "survival_status": status,
    }


def _family_row(seed: int, candidate_id: str, family: str) -> dict[str, object]:
    return {
        "seed": seed,
        "candidate_id": candidate_id,
        "candidate_group": "replicated_candidate",
        "layer": 7,
        "head": 7,
        "family": family,
        "heldout_family_type": "positive",
        "intervention": "head_clean_to_corrupt_patch",
        "position_label": "final",
        "head_specific_patch": True,
        "actual_patch_scope": "single_head_z",
        "metric": "true_vs_control_logit_diff",
        "n_examples": 12,
        "n_valid_examples": 12,
        "mean_effect_size": 0.1,
        "median_effect_size": 0.1,
        "n_position_unavailable": 0,
        "n_denominator_zero": 0,
    }


def _write_run(tmp_path: Path, *, seed: int, rows: list[dict[str, object]]) -> Path:
    run = tmp_path / f"run_{seed}"
    run.mkdir()
    pd.DataFrame(rows).to_csv(run / "heldout_robustness_by_candidate.csv", index=False)
    pd.DataFrame(
        [_family_row(seed, str(row["candidate_id"]), "heldout_symbolic_longer") for row in rows]
    ).to_csv(run / "heldout_robustness_by_family.csv", index=False)
    (run / "heldout_robustness_summary.json").write_text(
        json.dumps({"seed": seed}),
        encoding="utf-8",
    )
    return run
