from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

SupportedTest = Literal["sign", "wilcoxon", "permutation", "custom_registered"]


def compute_paired_test(
    path: str | Path,
    *,
    paired_by: str,
    metric: str,
    test: SupportedTest = "sign",
    run_id: str | None = None,
) -> dict[str, Any]:
    if test != "sign":
        raise NotImplementedError(
            "The lightweight MechLedger SDK only computes `sign` tests. "
            "It does not vendor numerical stacks for wilcoxon, permutation, "
            "or custom_registered tests; compute those externally and register "
            "the resulting JSON."
        )
    rows = _read_jsonl(Path(path))
    values: list[float] = []
    seen_pairs: set[str] = set()
    for row in rows:
        pair_id = row.get(paired_by)
        if pair_id in (None, ""):
            raise ValueError(f"Missing paired_by field `{paired_by}` in input row.")
        pair_key = str(pair_id)
        if pair_key in seen_pairs:
            raise ValueError(f"Duplicate paired_by value `{pair_key}`.")
        seen_pairs.add(pair_key)
        if metric not in row or row[metric] in (None, ""):
            raise ValueError(f"Missing metric field `{metric}` in input row.")
        value = float(row[metric])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite metric value for `{metric}`.")
        values.append(value)
    if not values:
        raise ValueError("No paired rows were found.")
    positive = sum(1 for value in values if value > 0.0)
    negative = sum(1 for value in values if value < 0.0)
    zero = len(values) - positive - negative
    if positive > negative:
        effect_direction = "positive"
    elif negative > positive:
        effect_direction = "negative"
    elif zero == len(values):
        effect_direction = "unknown"
    else:
        effect_direction = "mixed"
    majority = max(positive, negative)
    sign_consistency = majority / len(values)
    return {
        "run_id": run_id,
        "paired_by": paired_by,
        "metric": metric,
        "test": "sign",
        "n_pairs": len(values),
        "p_value": _two_sided_sign_p_value(positive, negative),
        "effect_direction": effect_direction,
        "sign_consistency": sign_consistency,
        "threshold_source": "tool_default",
        "threshold_justification": None,
        "threshold_decision_id": None,
        "input_artifact_path": str(path),
        "output_artifact_path": "",
    }


def write_paired_test_result(result: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} is not an object.")
        rows.append(row)
    return rows


def _two_sided_sign_p_value(positive: int, negative: int) -> float | None:
    n = positive + negative
    if n == 0:
        return None
    tail = min(positive, negative)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * probability)
