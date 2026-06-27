from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from local_mi_lab.head_specific_report import (
    classify_replication_status,
    compare_head_specific_runs,
    load_head_specific_run,
    render_multiseed_markdown,
    summarize_multiseed_heads,
)


def test_multiseed_merge_by_layer_head() -> None:
    frames = [
        pd.DataFrame([_head_row(0, 7, 7, "head_specific_positive_candidate", 0.08)]),
        pd.DataFrame([_head_row(1, 7, 7, "head_specific_positive_candidate", 0.06)]),
    ]
    merged = summarize_multiseed_heads(frames)
    row = merged.iloc[0]
    assert row["layer"] == 7
    assert row["head"] == 7
    assert row["n_seeds"] == 2
    assert row["seeds_present"] == "0,1"


def test_replication_status_classification() -> None:
    group = pd.DataFrame(
        [
            _head_row(0, 7, 7, "head_specific_positive_candidate", 0.08),
            _head_row(1, 7, 7, "head_specific_positive_candidate", 0.06),
            _head_row(2, 7, 7, "no_positive_effect", -0.01),
        ]
    )
    assert classify_replication_status(group) == "replicated_head_specific_candidate"


def test_nonspecific_blocks_replication() -> None:
    group = pd.DataFrame(
        [
            _head_row(0, 7, 7, "head_specific_positive_candidate", 0.08),
            _head_row(1, 7, 7, "head_specific_positive_candidate", 0.06),
            _head_row(2, 7, 7, "nonspecific_moves_controls", -0.02),
        ]
    )
    assert classify_replication_status(group) == "nonspecific"


def test_random_comparison_flagged_separately(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame(
        [
            {"source": "top_raw_positive_attention", "layer": 0, "head": 1},
            {"source": "random_comparison", "layer": 7, "head": 7},
        ]
    ).to_csv(source / "controlled_patching_candidates.csv", index=False)
    run = _write_run(tmp_path, seed=0, rows=[_head_row(0, 7, 7, "head_specific_positive_candidate", 0.08)])
    (run / "source_run.txt").write_text(str(source), encoding="utf-8")

    frame, _manifest = load_head_specific_run(run)

    assert bool(frame["random_comparison_candidate_in_seed"].iloc[0]) is True
    assert bool(frame["raw_attention_candidate_in_seed"].iloc[0]) is False


def test_not_head_specific_status_dominates() -> None:
    group = pd.DataFrame(
        [
            _head_row(
                0,
                7,
                7,
                "head_specific_positive_candidate",
                0.08,
                head_specific=False,
            ),
            _head_row(1, 7, 7, "head_specific_positive_candidate", 0.06),
        ]
    )
    assert classify_replication_status(group) == "not_head_specific"


def test_markdown_report_contains_no_mechanism_overclaim() -> None:
    by_head = summarize_multiseed_heads(
        [
            pd.DataFrame([_head_row(0, 7, 7, "head_specific_positive_candidate", 0.08)]),
            pd.DataFrame([_head_row(1, 7, 7, "head_specific_positive_candidate", 0.06)]),
        ]
    )
    summary = {
        "replicated_candidates": by_head.to_dict("records"),
        "top_heads_by_mean_gap": by_head.to_dict("records"),
        "raw_attention_candidate_outcomes": [],
        "random_comparison_head_outcomes": [],
        "n_heads": 1,
        "n_replicated_candidates": 1,
        "status_counts": {"replicated_head_specific_candidate": 1},
        "executive_summary": "This experiment identified at least one narrow candidate.",
    }
    markdown = render_multiseed_markdown(summary, by_head, [{"seed": 0, "run_dir": "r0", "n_heads": 1}])
    assert "not a mechanism or circuit claim" in markdown
    assert "discovered an induction head" not in markdown


def test_compare_writes_expected_artifacts(tmp_path: Path) -> None:
    run0 = _write_run(
        tmp_path,
        seed=0,
        rows=[_head_row(0, 7, 7, "head_specific_positive_candidate", 0.08)],
    )
    run1 = _write_run(
        tmp_path,
        seed=1,
        rows=[_head_row(1, 7, 7, "head_specific_positive_candidate", 0.06)],
    )
    paths = compare_head_specific_runs([run0, run1], tmp_path / "report")
    assert paths["by_head"].exists()
    assert paths["summary"].exists()
    assert paths["markdown"].exists()
    payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert payload["n_replicated_candidates"] == 1


def _head_row(
    seed: int,
    layer: int,
    head: int,
    status: str,
    gap: float,
    *,
    head_specific: bool = True,
) -> dict[str, object]:
    positive = 0.05 if status != "no_positive_effect" else -0.01
    control = positive - gap
    return {
        "seed": seed,
        "layer": layer,
        "head": head,
        "hook_name": f"blocks.{layer}.attn.hook_z",
        "head_specific_patch": head_specific,
        "actual_patch_scope": "single_head_z" if head_specific else "full_attn_out_layer",
        "intervention": "head_clean_to_corrupt_patch",
        "metric": "true_vs_control_logit_diff",
        "positive_mean_effect_size": positive,
        "max_control_mean_effect_size": control,
        "positive_minus_control_effect_gap": gap,
        "hardest_control_family": "same_token_frequency_control",
        "n_positive_examples": 8,
        "n_control_examples": 24,
        "specificity_status": status,
        "raw_attention_candidate_in_seed": False,
        "random_comparison_candidate_in_seed": False,
    }


def _write_run(tmp_path: Path, *, seed: int, rows: list[dict[str, object]]) -> Path:
    run = tmp_path / f"run_{seed}"
    run.mkdir()
    pd.DataFrame(rows).to_csv(run / "head_specific_patching_by_head.csv", index=False)
    (run / "head_specific_induction_summary.json").write_text(
        json.dumps({"seed": seed}),
        encoding="utf-8",
    )
    return run
