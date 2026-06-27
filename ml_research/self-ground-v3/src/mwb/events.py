from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mwb.time import utc_now


def append_event(events_path: Path, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event_type": event_type,
        "created_at": utc_now(),
        "payload": payload,
    }
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def read_events(events_path: Path) -> list[dict[str, Any]]:
    if not events_path.exists():
        return []
    events = []
    for line_number, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {events_path}:{line_number}: {exc}") from exc
    return events

