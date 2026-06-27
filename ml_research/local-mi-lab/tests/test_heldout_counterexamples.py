from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inspect_heldout_counterexamples.py"
SPEC = importlib.util.spec_from_file_location("inspect_heldout_counterexamples", SCRIPT_PATH)
assert SPEC is not None
counterexamples = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(counterexamples)


def test_parse_candidate() -> None:
    assert counterexamples.parse_candidate("L7H11") == (7, 11)


def test_inspect_candidate_buckets_rows() -> None:
    rows = pd.DataFrame(
        [
            _result_row("heldout_symbolic_longer", "positive", 0.4),
            _result_row("heldout_word_sequences", "positive", -0.2),
            _result_row("heldout_no_structure_same_tokens", "control", 0.3),
            _result_row("heldout_wrong_target_same_prompt", "control", 0.1),
        ]
    )

    selected, markdown = counterexamples.inspect_candidate(rows, "L7H7")

    assert set(selected["inspection_bucket"]) == {
        "strongest_positive_success",
        "strongest_positive_failure",
        "controls_that_moved",
        "wrong_target_failure",
    }
    assert "Controls that moved" in markdown
    assert "Do not claim induction-head discovery" in markdown


def test_write_counterexample_artifacts(tmp_path: Path) -> None:
    rows = pd.DataFrame([_result_row("heldout_symbolic_longer", "positive", 0.4)])
    paths = counterexamples.write_counterexample_artifacts(
        "L7H7",
        rows,
        "# Held-Out Counterexamples: L7H7\n",
        tmp_path,
    )

    assert paths["csv"].exists()
    assert paths["markdown"].exists()


def _result_row(family: str, family_type: str, effect: float) -> dict[str, object]:
    return {
        "seed": 10,
        "layer": 7,
        "head": 7,
        "family": family,
        "example_id": f"{family}_000",
        "heldout_family_type": family_type,
        "intervention": "head_clean_to_corrupt_patch",
        "position_label": "final",
        "effect_size": effect,
        "true_expected_next_token": " D",
        "wrong_or_control_token": " X",
        "clean_prompt": "A B C D A B C",
        "corrupt_prompt": "A B C D X Y Z",
    }
