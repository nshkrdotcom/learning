from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from mechledger.alias import append_alias_record
from mechledger.artifacts import attach_artifact, auto_collect_artifacts
from mechledger.git_state import captured_environment, git_metadata
from mechledger.io import append_jsonl, sanitize_metric_value, utc_now, write_json
from mechledger.paths import alias_cache_path, find_project_root, runs_dir
from mechledger.run_capture import generate_run_id, slugify


def run(**kwargs):
    return start(**kwargs)


def start(
    *,
    experiment: str | None = None,
    run_class: str = "scratch",
    purpose: str | None = None,
    hypothesis: str | None = None,
    project_root: str | Path | None = None,
) -> RunSession:
    root = find_project_root(project_root)
    return RunSession(root, experiment, run_class, purpose, hypothesis)


class RunSession:
    def __init__(
        self,
        project_root: Path,
        experiment: str | None,
        run_class: str,
        purpose: str | None,
        hypothesis: str | None,
    ) -> None:
        self.project_root = project_root
        self.experiment = experiment
        self.run_class = run_class
        self.purpose = purpose
        self.hypothesis = hypothesis
        self.run_id = generate_run_id(
            experiment_id=experiment, purpose=purpose, run_class=run_class
        )
        self.run_dir = runs_dir(project_root) / self.run_id
        self._finished = False

    def __enter__(self) -> RunSession:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.finish(status="interrupted", blocker=str(exc) if exc else None)
            return False
        self.finish(status="completed")
        return False

    def start(self) -> RunSession:
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "artifacts").mkdir()
        (self.run_dir / "metrics.jsonl").touch()
        (self.run_dir / "events.jsonl").touch()
        (self.run_dir / "artifacts.jsonl").touch()
        write_json(self.run_dir / "artifact_manifest.json", {"artifacts": []})
        git = git_metadata(self.project_root)
        write_json(self.run_dir / "git.json", git)
        write_json(self.run_dir / "environment.json", captured_environment())
        run_json = {
            "run_id": self.run_id,
            "parent_run_id": None,
            "experiment_id": self.experiment,
            "run_class": self.run_class,
            "status": "running",
            "purpose": self.purpose,
            "hypothesis": self.hypothesis,
            "command": None,
            "started_at": utc_now(),
            "finished_at": None,
            "exit_code": None,
            "git_commit": git.get("git_commit"),
            "git_diff_hash": git.get("git_diff_hash"),
            "cwd": str(self.project_root),
            "model": None,
            "tokenizer": None,
            "hook_point": None,
            "sae_release": None,
            "sae_id": None,
            "seed": None,
            "blocker": None,
            "pinned": False,
        }
        write_json(self.run_dir / "run.json", run_json)
        write_json(
            self.run_dir / "heartbeat.json", {"last_heartbeat_at": utc_now(), "pid": os.getpid()}
        )
        self._event("run_created", "SDK run created.")
        self._event("run_started", "SDK run started.")
        append_alias_record(
            alias_cache_path(self.project_root),
            self.run_id,
            self.experiment,
            slugify(self.purpose or self.run_class),
        )
        return self

    def log_metric(
        self,
        metric_name: str,
        value: Any,
        *,
        step: int | None = None,
        family: str | None = None,
        split: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        clean_value = sanitize_metric_value(value)
        if isinstance(value, float) and not math.isfinite(value):
            self._event(
                "metric_logged",
                "Non-finite metric serialized as null.",
                {"metric_name": metric_name},
            )
        append_jsonl(
            self.run_dir / "metrics.jsonl",
            {
                "metric_name": metric_name,
                "value": clean_value,
                "step": step,
                "family": family,
                "split": split,
                "metadata": metadata or {},
            },
        )
        self._event("metric_logged", f"Metric logged: {metric_name}", {"metric_name": metric_name})

    def log_artifact(
        self,
        path: str | Path,
        *,
        artifact_type: str | None = None,
        claim_relevance: str = "none",
        description: str | None = None,
    ):
        return attach_artifact(
            self.run_dir,
            path,
            artifact_type=artifact_type,
            claim_relevance=claim_relevance,
            description=description,
        )

    def log_intervention_metadata(self, **metadata: Any) -> None:
        self._event("intervention_logged", "Intervention metadata logged.", metadata)

    def finish(self, *, status: str = "completed", blocker: str | None = None) -> None:
        if self._finished:
            return
        auto_collect_artifacts(self.run_dir)
        data = __import__("json").loads((self.run_dir / "run.json").read_text(encoding="utf-8"))
        data.update(
            {
                "status": status,
                "finished_at": utc_now(),
                "exit_code": 0 if status == "completed" else None,
                "blocker": blocker,
            }
        )
        write_json(self.run_dir / "run.json", data)
        write_json(self.run_dir / "summary.json", {"run_id": self.run_id, "status": status})
        if status == "completed":
            (self.run_dir / "heartbeat.json").unlink(missing_ok=True)
            self._event("run_completed", "SDK run completed.")
        else:
            self._event("run_interrupted", "SDK run interrupted.", {"blocker": blocker})
        self._finished = True

    def _event(self, event_type: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        append_jsonl(
            self.run_dir / "events.jsonl",
            {
                "timestamp": utc_now(),
                "event_type": event_type,
                "message": message,
                "metadata": metadata or {},
            },
        )
