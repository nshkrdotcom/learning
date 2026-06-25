from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

from mechledger.project import Project, now_utc

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


@dataclass(frozen=True)
class AliasRecord:
    run_id: str
    timestamp: str
    experiment_id: str
    slug: str


class AliasResolutionError(ValueError):
    pass


def append_alias(project: Project, run_id: str, experiment_id: str | None, slug: str) -> None:
    path = project.mechledger_dir / "alias_cache.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{run_id}\t{now_utc()}\t{experiment_id or 'noexp'}\t{slug}\n"
    with path.open("a+", encoding="utf-8") as handle:
        _lock(handle)
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
        _unlock(handle)


def read_aliases(project: Project) -> tuple[list[AliasRecord], bool]:
    path = project.mechledger_dir / "alias_cache.txt"
    if not path.exists():
        return [], False
    records: dict[str, AliasRecord] = {}
    malformed = False
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 4 or not parts[0]:
            malformed = True
            continue
        record = AliasRecord(parts[0], parts[1], parts[2], parts[3])
        records[record.run_id] = record
    return sorted(records.values(), key=lambda item: item.timestamp), malformed


def rebuild_alias_cache(project: Project) -> None:
    path = project.mechledger_dir / "alias_cache.txt"
    lines = []
    for run_json in sorted(project.runs_dir.glob("*/run.json")):
        import json

        data = json.loads(run_json.read_text(encoding="utf-8"))
        lines.append(
            f"{data['run_id']}\t{data.get('started_at') or now_utc()}\t"
            f"{data.get('experiment_id') or 'noexp'}\t{_slug_from_run_id(data['run_id'])}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def resolve_run_id(project: Project, alias: str) -> str:
    records, malformed = read_aliases(project)
    if malformed:
        (project.mechledger_dir / "cache").mkdir(parents=True, exist_ok=True)
        (project.mechledger_dir / "cache/alias_rebuild_required").write_text(
            "1\n", encoding="utf-8"
        )
    if not records and project.runs_dir.exists():
        rebuild_alias_cache(project)
        records, _ = read_aliases(project)
    if not records:
        raise AliasResolutionError(f"No runs are recorded; cannot resolve alias `{alias}`.")
    if alias == "latest":
        return records[-1].run_id
    if alias.startswith("latest:"):
        try:
            index = int(alias.split(":", 1)[1])
        except ValueError as exc:
            raise AliasResolutionError(f"Invalid latest:N alias: {alias}") from exc
        if index <= 0 or index > len(records):
            raise AliasResolutionError(f"Alias `{alias}` is out of range.")
        return records[-index].run_id
    if alias.startswith("#"):
        try:
            number = int(alias[1:])
        except ValueError as exc:
            raise AliasResolutionError(f"Invalid #N alias: {alias}") from exc
        if number <= 0 or number > len(records):
            raise AliasResolutionError(f"Alias `{alias}` is out of range.")
        return records[number - 1].run_id
    by_prefix = [record.run_id for record in records if record.run_id.startswith(alias)]
    by_component = [
        record.run_id
        for record in records
        if f"{record.experiment_id}_{record.slug}" in record.run_id
        and f"{record.experiment_id}_{record.slug}".startswith(alias)
    ]
    matches = sorted(set(by_prefix + by_component))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise AliasResolutionError(f"Alias `{alias}` did not match any run.")
    candidates = "\n".join(f"  {match}" for match in matches[:10])
    raise AliasResolutionError(
        f"Alias `{alias}` is ambiguous. Matches:\n{candidates}\nUse a longer prefix or full run ID."
    )


def _lock(handle: object) -> None:
    if fcntl is None:
        return
    delay = 0.02
    for _ in range(8):
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            time.sleep(delay + random.random() * delay)
            delay *= 2
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock(handle: object) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _slug_from_run_id(run_id: str) -> str:
    parts = run_id.split("_")
    return "_".join(parts[2:-1]) if len(parts) > 3 else run_id
