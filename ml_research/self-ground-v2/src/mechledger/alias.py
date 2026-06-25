from __future__ import annotations

import fcntl
import random
import time
from dataclasses import dataclass
from pathlib import Path

from mechledger.io import utc_now


@dataclass(slots=True)
class AliasRecord:
    run_id: str
    timestamp: str
    experiment_id: str
    slug: str


def append_alias_record(
    cache_path: str | Path, run_id: str, experiment_id: str | None, slug: str
) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    record = f"{run_id}\t{utc_now()}\t{experiment_id or 'noexp'}\t{slug}\n"
    for attempt in range(5):
        with cache_path.open("a", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle.write(record)
                handle.flush()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                return
            except BlockingIOError:
                time.sleep((0.025 * (2**attempt)) + random.random() * 0.01)
    with cache_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(record)
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_alias_cache(cache_path: str | Path) -> list[AliasRecord]:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return []
    records: list[AliasRecord] = []
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        run_id, timestamp, experiment_id, slug = parts
        if not run_id or not timestamp:
            continue
        records.append(
            AliasRecord(run_id=run_id, timestamp=timestamp, experiment_id=experiment_id, slug=slug)
        )
    return records


def resolve_run_alias(cache_path: str | Path, alias: str) -> AliasRecord:
    records = _dedup_latest(read_alias_cache(cache_path))
    if not records:
        raise ValueError("alias cache is empty; run `mechledger index` to rebuild it")
    if alias == "latest":
        return records[-1]
    if alias.startswith("latest:"):
        count = int(alias.split(":", 1)[1])
        if count < 1 or count > len(records):
            raise ValueError(f"alias {alias!r} is out of range")
        return records[-count]
    if alias.startswith("#"):
        number = int(alias[1:])
        if number < 1 or number > len(records):
            raise ValueError(f"alias {alias!r} is out of range")
        return records[number - 1]
    lowered = alias.lower()
    matches = [
        record
        for record in records
        if record.run_id.lower().startswith(lowered)
        or lowered in "_".join(record.run_id.split("_")[1:]).lower()
        or lowered == record.slug.lower()
    ]
    if not matches:
        raise ValueError(f"Alias {alias!r} did not match any known run.")
    if len(matches) > 1:
        candidate_lines = "\n  ".join(record.run_id for record in matches)
        raise ValueError(f"Alias `{alias}` is ambiguous. Matches:\n  {candidate_lines}")
    return matches[0]


def _dedup_latest(records: list[AliasRecord]) -> list[AliasRecord]:
    latest: dict[str, AliasRecord] = {}
    for record in records:
        latest[record.run_id] = record
    return sorted(latest.values(), key=lambda record: record.timestamp)
