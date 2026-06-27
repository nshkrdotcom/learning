from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from local_mi_lab.robustness_candidates import (
    PRIOR_RAW_ATTENTION_HEADS,
    REQUIRED_REPLICATED_HEADS,
    select_heldout_candidate_set,
    write_heldout_candidate_artifacts,
)


def test_replicated_candidates_included(tmp_path: Path) -> None:
    path = _write_multiseed(tmp_path)
    candidates = select_heldout_candidate_set(path)
    heads = {(candidate["layer"], candidate["head"]) for candidate in candidates}
    assert set(REQUIRED_REPLICATED_HEADS).issubset(heads)


def test_prior_raw_attention_heads_included(tmp_path: Path) -> None:
    path = _write_multiseed(tmp_path)
    candidates = select_heldout_candidate_set(path)
    raw = {
        (candidate["layer"], candidate["head"])
        for candidate in candidates
        if candidate["candidate_group"] == "prior_raw_attention_failed"
    }
    assert set(PRIOR_RAW_ATTENTION_HEADS).issubset(raw)


def test_negative_controls_selected_deterministically(tmp_path: Path) -> None:
    path = _write_multiseed(tmp_path)
    first = select_heldout_candidate_set(path)
    second = select_heldout_candidate_set(path)
    assert first == second
    negatives = [
        candidate
        for candidate in first
        if candidate["candidate_group"].startswith("negative_control")
    ]
    assert len(negatives) >= 5


def test_l7h7_flagged_as_prior_random_comparison(tmp_path: Path) -> None:
    path = _write_multiseed(tmp_path)
    candidates = select_heldout_candidate_set(path)
    l7h7 = next(candidate for candidate in candidates if (candidate["layer"], candidate["head"]) == (7, 7))
    assert l7h7["candidate_group"] == "random_comparison_replicated"
    assert l7h7["prior_random_comparison_candidate"] is True


def test_writes_candidate_artifacts(tmp_path: Path) -> None:
    path = _write_multiseed(tmp_path)
    candidates = select_heldout_candidate_set(path)
    paths = write_heldout_candidate_artifacts(
        candidates,
        tmp_path / "heldout_candidate_set.csv",
        source_multiseed=path,
    )
    assert paths["csv"].exists()
    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert "L7H7 remains flagged" in paths["markdown"].read_text()


def test_fails_if_required_replicated_head_absent(tmp_path: Path) -> None:
    path = _write_multiseed(tmp_path, drop=(7, 7))
    with pytest.raises(ValueError):
        select_heldout_candidate_set(path)


def test_allow_missing_required_bypasses_required_failure(tmp_path: Path) -> None:
    path = _write_multiseed(tmp_path, drop=(7, 7))
    candidates = select_heldout_candidate_set(path, allow_missing_required=True)
    assert (7, 7) not in {(candidate["layer"], candidate["head"]) for candidate in candidates}


def _write_multiseed(tmp_path: Path, drop: tuple[int, int] | None = None) -> Path:
    rows = []
    for layer, head in REQUIRED_REPLICATED_HEADS:
        rows.append(
            _row(
                layer,
                head,
                "replicated_head_specific_candidate",
                0.1 if (layer, head) == (7, 7) else 0.03,
                random=(layer, head) == (7, 7),
            )
        )
    for layer, head in PRIOR_RAW_ATTENTION_HEADS:
        rows.append(_row(layer, head, "no_effect" if layer != 11 else "nonspecific", -0.05, raw=True))
    for index in range(12):
        rows.append(_row(2 + index % 4, index, "no_effect", 0.001 * index))
    for index in range(12):
        rows.append(_row(8 + index % 4, index, "nonspecific", -0.001 * index))
    if drop is not None:
        rows = [row for row in rows if (row["layer"], row["head"]) != drop]
    path = tmp_path / "multiseed.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _row(
    layer: int,
    head: int,
    status: str,
    gap: float,
    *,
    raw: bool = False,
    random: bool = False,
) -> dict[str, object]:
    return {
        "layer": layer,
        "head": head,
        "mean_positive_minus_control_gap": gap,
        "replication_status": status,
        "raw_attention_candidate_in_any_seed": raw,
        "random_comparison_candidate_in_any_seed": random,
    }
