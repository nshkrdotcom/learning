from __future__ import annotations

from pathlib import Path

from construct_mismatch.datasets import (
    CONSTRUCTS,
    DECOUPLING_AXES,
    REQUIRED_FIELDS,
    dataset_file,
    read_jsonl,
)


def test_dataset_schema_rows_have_required_fields() -> None:
    root = Path.cwd()
    for construct in CONSTRUCTS:
        for split in ("train", "heldout", "decoupling"):
            path = dataset_file(construct, split, root)
            assert path.exists(), f"Missing dataset file: {path}"
            for record in read_jsonl(path):
                assert REQUIRED_FIELDS <= set(record)


def test_each_construct_has_required_decoupling_axes() -> None:
    root = Path.cwd()
    for construct in CONSTRUCTS:
        records = read_jsonl(dataset_file(construct, "decoupling", root))
        axes = {record["decoupling_axis"] for record in records}
        assert set(DECOUPLING_AXES) <= axes
