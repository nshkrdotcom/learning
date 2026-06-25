from __future__ import annotations

import hashlib
import json
from typing import Any

ORDER_INSENSITIVE_CLAIM_FIELDS = {
    "allowed",
    "forbidden",
    "required_caveats",
    "debt_flags",
    "linked_experiments",
    "linked_runs",
    "linked_decisions",
    "tags",
}


def _strip_empty(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            stripped = _strip_empty(item)
            if stripped in (None, "", [], {}):
                continue
            cleaned[key] = stripped
        return cleaned
    if isinstance(value, list):
        return [
            _strip_empty(item) for item in value if _strip_empty(item) not in (None, "", [], {})
        ]
    return value


def canonicalize_claim_yaml(data: dict[str, Any]) -> dict[str, Any]:
    stripped = _strip_empty(data)
    canonical: dict[str, Any] = {}
    for key in sorted(stripped):
        value = stripped[key]
        if key in ORDER_INSENSITIVE_CLAIM_FIELDS and isinstance(value, list):
            scalar_values = {str(item) for item in value}
            canonical[key] = sorted(scalar_values)
        elif isinstance(value, dict):
            canonical[key] = canonicalize_claim_yaml(value)
        else:
            canonical[key] = value
    return canonical


def canonical_claim_json(data: dict[str, Any]) -> str:
    return json.dumps(
        canonicalize_claim_yaml(data),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_claim_hash(data: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_claim_json(data).encode("utf-8")).hexdigest()
