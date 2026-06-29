from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

CONSTRUCTS = ("certainty", "sentiment")
ORDINARY_AXIS = "ordinary"
DECOUPLING_AXES = (
    "lexical_reversal",
    "negation",
    "quotation",
    "contrast",
    "format_shift",
)
ALL_AXES = (ORDINARY_AXIS, *DECOUPLING_AXES)
SPLITS = ("train", "heldout", "decoupling")
LABELS = ("class_a", "class_b")

REQUIRED_FIELDS = {
    "id",
    "construct",
    "split",
    "decoupling_axis",
    "prompt",
    "class_a_label",
    "class_b_label",
    "class_a_target",
    "class_b_target",
    "label",
    "template_id",
    "notes",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "data" / "processed"


def artifact_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "artifacts"


def validate_record(record: dict[str, Any]) -> None:
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        raise ValueError(f"Record {record.get('id', '<unknown>')} is missing fields: {sorted(missing)}")
    if record["construct"] not in CONSTRUCTS:
        raise ValueError(f"Invalid construct in {record['id']}: {record['construct']}")
    if record["split"] not in SPLITS:
        raise ValueError(f"Invalid split in {record['id']}: {record['split']}")
    if record["decoupling_axis"] not in ALL_AXES:
        raise ValueError(f"Invalid decoupling axis in {record['id']}: {record['decoupling_axis']}")
    if record["label"] not in LABELS:
        raise ValueError(f"Invalid label in {record['id']}: {record['label']}")
    if not record["prompt"] or not isinstance(record["prompt"], str):
        raise ValueError(f"Empty prompt in {record['id']}")
    for field in ("class_a_target", "class_b_target"):
        if not record[field].startswith(" "):
            raise ValueError(f"{field} must include the GPT-2 leading space in {record['id']}")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            validate_record(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                validate_record(record)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            records.append(record)
    return records


def dataset_file(construct: str, split: str, root: Path | None = None) -> Path:
    if construct not in CONSTRUCTS:
        raise ValueError(f"Unknown construct: {construct}")
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return data_path(root) / f"{construct}_{split}.jsonl"


def load_dataset(construct: str, split: str, root: Path | None = None) -> list[dict[str, Any]]:
    return read_jsonl(dataset_file(construct, split, root))


def load_construct_records(construct: str, root: Path | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        path = dataset_file(construct, split, root)
        if path.exists():
            records.extend(read_jsonl(path))
    return records


def group_by_axis(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["decoupling_axis"]].append(record)
    return dict(grouped)


def paired_records(records: list[dict[str, Any]], axis: str | None = None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if axis is not None and record["decoupling_axis"] != axis:
            continue
        pair_id = record.get("pair_id")
        role = record.get("pair_role")
        if pair_id and role in LABELS:
            by_pair[pair_id][role] = record
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for pair_id in sorted(by_pair):
        pair = by_pair[pair_id]
        if "class_a" in pair and "class_b" in pair:
            pairs.append((pair["class_a"], pair["class_b"]))
    return pairs
