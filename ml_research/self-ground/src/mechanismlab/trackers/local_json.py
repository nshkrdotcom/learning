from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mechanismlab.core.runs import RunManifest


class LocalJsonTracker:
    name = "local_json"

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "tracker_events.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _write_event(self, event: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            **event,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def start_run(self, run: RunManifest) -> None:
        self._write_event({"event": "start_run", "run": run.model_dump(mode="json")})

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._write_event({"event": "log_metrics", "metrics": metrics, "step": step})

    def log_artifact(self, path: Path, name: str | None = None) -> None:
        self._write_event(
            {
                "event": "log_artifact",
                "path": str(Path(path)),
                "name": name or Path(path).name,
            }
        )

    def finish(self, status: str | None = None) -> None:
        self._write_event({"event": "finish", "status": status})
