from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from self_ground.activations import RankedFeature
from self_ground.data import ExperimentResult, MinimalPair

SUMMARY_COLUMNS = [
    "feature_id",
    "n_pairs",
    "necessity_mean",
    "sufficiency_mean",
    "specificity_mean",
    "collateral_mean",
    "cleanliness_mean",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def write_jsonl(rows: list[Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_minimal_pairs(path: str | Path) -> list[MinimalPair]:
    return [MinimalPair.model_validate(row) for row in read_jsonl(path)]


def write_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_feature_rankings_csv(rankings: list[RankedFeature], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature_id", "score", "mean_pos", "mean_neg", "mean_para", "mean_decoy"],
        )
        writer.writeheader()
        for row in rankings:
            writer.writerow(
                {
                    "feature_id": row.feature_id,
                    "score": row.score,
                    "mean_pos": row.mean_pos,
                    "mean_neg": row.mean_neg,
                    "mean_para": row.mean_para,
                    "mean_decoy": row.mean_decoy,
                }
            )


def write_summary_csv(results: list[ExperimentResult], path: str | Path) -> None:
    grouped: dict[str, list[ExperimentResult]] = defaultdict(list)
    for result in results:
        grouped[result.feature_id].append(result)

    rows: list[dict[str, Any]] = []
    for feature_id, feature_results in grouped.items():
        n = len(feature_results)
        rows.append(
            {
                "feature_id": feature_id,
                "n_pairs": n,
                "necessity_mean": sum(r.metrics.necessity for r in feature_results) / n,
                "sufficiency_mean": sum(r.metrics.sufficiency for r in feature_results) / n,
                "specificity_mean": sum(r.metrics.specificity for r in feature_results) / n,
                "collateral_mean": sum(r.metrics.collateral for r in feature_results) / n,
                "cleanliness_mean": sum(r.metrics.cleanliness for r in feature_results) / n,
            }
        )
    rows.sort(key=lambda row: (-float(row["cleanliness_mean"]), str(row["feature_id"])))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
