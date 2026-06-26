from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mechledger.project import now_utc
from mechledger.sdk import stats

__all__ = [
    "ActiveRun",
    "current_run",
    "log_artifact",
    "log_event",
    "log_intervention_metadata",
    "log_metric",
    "stats",
]


class ActiveRun:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id

    def log_metric(
        self,
        metric_name: str,
        value: object,
        *,
        step: int | None = None,
        family: str | None = None,
        split: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
            self.log_event("metric_value_nonfinite", f"{metric_name} serialized as null")
            value = None
        _append_jsonl(
            self.run_dir / "metrics.jsonl",
            {
                "metric_name": metric_name,
                "value": value,
                "step": step,
                "family": family,
                "split": split,
                "metadata": metadata or {},
            },
        )
        self.log_event("metric_logged", metric_name)

    def log_event(
        self, event_type: str, message: str, metadata: dict[str, Any] | None = None
    ) -> None:
        metadata = metadata or {}
        if event_type == "external_call":
            missing = [
                key
                for key in ("external", "service", "reproducibility_scope")
                if key not in metadata
            ]
            if metadata.get("external") is not True or missing:
                raise ValueError(
                    "external_call events require metadata with external=True, "
                    "service, and reproducibility_scope."
                )
        _append_jsonl(
            self.run_dir / "events.jsonl",
            {
                "timestamp": now_utc(),
                "event_type": event_type,
                "message": message,
                "metadata": metadata,
            },
        )

    def log_artifact(
        self,
        path: str,
        *,
        artifact_type: str | None = None,
        claim_relevance: str = "none",
        description: str | None = None,
    ) -> None:
        _append_jsonl(
            self.run_dir / "artifacts.jsonl",
            {
                "timestamp": now_utc(),
                "event_type": "artifact_registered",
                "path": path,
                "artifact_type": artifact_type,
                "claim_relevance": claim_relevance,
                "review_status": "annotated" if claim_relevance != "none" else "unannotated",
                "description": description,
            },
        )

    def log_intervention_metadata(self, **metadata: Any) -> None:
        self.log_event("intervention_logged", "intervention metadata logged", metadata)


def current_run() -> ActiveRun:
    run_dir = os.environ.get("MECHLEDGER_RUN_DIR")
    run_id = os.environ.get("MECHLEDGER_RUN_ID")
    if not run_dir or not run_id:
        raise RuntimeError("No active MechLedger run. Use `mechledger run -- ...`.")
    return ActiveRun(Path(run_dir), run_id)


def log_metric(metric_name: str, value: object, **kwargs: Any) -> None:
    current_run().log_metric(metric_name, value, **kwargs)


def log_event(event_type: str, message: str, metadata: dict[str, Any] | None = None) -> None:
    current_run().log_event(event_type, message, metadata)


def log_artifact(path: str, **kwargs: Any) -> None:
    current_run().log_artifact(path, **kwargs)


def log_intervention_metadata(**metadata: Any) -> None:
    current_run().log_intervention_metadata(**metadata)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
