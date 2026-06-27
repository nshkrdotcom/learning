from __future__ import annotations

import pandas as pd
import pytest

from local_mi_lab.head_sweep import (
    completed_row_keys,
    expand_heads,
    fail_if_existing_outputs,
    layers_from_spec,
    make_head_sweep_run_dir,
    summary_status_counts,
)


def test_head_list_expansion_from_config() -> None:
    assert layers_from_spec([0, 2, 4], n_layers=12) == [0, 2, 4]
    assert layers_from_spec("0,2,4", n_layers=12) == [0, 2, 4]


def test_selected_layer_head_list_generation() -> None:
    assert expand_heads([0, 2], n_heads=3) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]


def test_auto_even_layers() -> None:
    assert layers_from_spec("auto_even_6", n_layers=12) == [0, 2, 4, 7, 9, 11]


def test_run_directory_naming(tmp_path) -> None:
    config = {
        "experiment": {"name": "gpt2_small_head_specific_induction"},
        "outputs": {"run_root": tmp_path},
    }
    run_dir = make_head_sweep_run_dir(config)
    assert run_dir.name.endswith("_gpt2_small_head_specific_induction")


def test_resume_completed_key_logic() -> None:
    rows = pd.DataFrame(
        [
            {
                "layer": 0,
                "head": 1,
                "example_id": "p0",
                "family": "positive_repeat_sequence",
                "metric": "true_vs_control_logit_diff",
                "intervention": "head_clean_to_corrupt_patch",
            }
        ]
    )
    assert completed_row_keys(rows) == {
        (
            0,
            1,
            "p0",
            "positive_repeat_sequence",
            "true_vs_control_logit_diff",
            "head_clean_to_corrupt_patch",
        )
    }


def test_summary_status_counts() -> None:
    by_head = pd.DataFrame(
        {
            "specificity_status": [
                "head_specific_positive_candidate",
                "no_positive_effect",
                "no_positive_effect",
            ]
        }
    )
    assert summary_status_counts(by_head) == {
        "no_positive_effect": 2,
        "head_specific_positive_candidate": 1,
    }


def test_fail_on_existing_output_without_overwrite_or_resume(tmp_path) -> None:
    output = tmp_path / "head_specific_patching_results.csv"
    output.write_text("x\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        fail_if_existing_outputs(tmp_path, resume=False, overwrite=False)
    fail_if_existing_outputs(tmp_path, resume=True, overwrite=False)
    fail_if_existing_outputs(tmp_path, resume=False, overwrite=True)
