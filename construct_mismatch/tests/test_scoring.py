from __future__ import annotations

from pathlib import Path

import pandas as pd

from construct_mismatch.scoring import OBJECT_LABELS, STATUS_LABELS


def test_scoring_emits_only_allowed_labels() -> None:
    matrix_path = Path("artifacts/scoring/construct_mismatch_matrix.csv")
    classifications_path = Path("artifacts/scoring/object_classifications.csv")
    assert matrix_path.exists(), "Run the pipeline to generate the scoring matrix."
    assert classifications_path.exists(), "Run the pipeline to generate object classifications."
    matrix = pd.read_csv(matrix_path)
    classifications = pd.read_csv(classifications_path)
    assert set(matrix["status"]) <= set(STATUS_LABELS)
    assert set(classifications["object_classification"]) <= set(OBJECT_LABELS)
