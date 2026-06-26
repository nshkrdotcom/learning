from __future__ import annotations

import json
import os
import resource
import threading
import time
from pathlib import Path
from typing import Any

from mechledger.project import now_utc
from mechledger.sdk import stats

__all__ = [
    "ActiveRun",
    "ManagedRun",
    "artifacts_dir",
    "current_run",
    "log_artifact",
    "log_event",
    "log_intervention_metadata",
    "log_metric",
    "run",
    "start",
    "stats",
]

CLAIM_RELEVANCE_VALUES = {"none", "diagnostic", "supporting", "contradicting", "required"}


class ActiveRun:
    def __init__(self, run_dir: Path, run_id: str, project: Any | None = None) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self._project = project

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
        if not metric_name:
            raise ValueError("metric_name is required.")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metric metadata must be a JSON object.")
        if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
            self.log_event("metric_value_nonfinite", f"{metric_name} serialized as null")
            value = None
        if value is not None and not isinstance(value, str | int | float | bool):
            raise ValueError("metric value must be a JSON scalar or null.")
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
        if not event_type:
            raise ValueError("event_type is required.")
        metadata = metadata or {}
        if not isinstance(metadata, dict):
            raise ValueError("event metadata must be a JSON object.")
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
        if not path:
            raise ValueError("artifact path is required.")
        if claim_relevance not in CLAIM_RELEVANCE_VALUES:
            raise ValueError(
                "claim_relevance must be one of: "
                + ", ".join(sorted(CLAIM_RELEVANCE_VALUES))
            )
        project = self._project or _find_project_for_run(self.run_dir)
        if project is None:
            _append_jsonl(
                self.run_dir / "artifacts.jsonl",
                {
                    "timestamp": now_utc(),
                    "event_type": "artifact_registered",
                    "path": path,
                    "artifact_type": artifact_type,
                    "claim_relevance": claim_relevance,
                    "review_status": "annotated"
                    if claim_relevance != "none"
                    else "unannotated",
                    "description": description,
                },
            )
            return
        from mechledger.artifacts import register_artifact

        artifact = register_artifact(
            project,
            self.run_id,
            Path(path),
            artifact_type=artifact_type,
            claim_relevance=claim_relevance,
            description=description,
        )
        self.log_event(
            "artifact_registered",
            artifact["artifact_id"],
            {"path": artifact["original_path"], "claim_relevance": claim_relevance},
        )

    def log_intervention_metadata(self, **metadata: Any) -> None:
        self.log_event("intervention_logged", "intervention metadata logged", metadata)

    def artifacts_dir(self) -> Path:
        path = self.run_dir / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path


class ManagedRun(ActiveRun):
    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        project: Any,
        run_payload: dict[str, Any],
        started_monotonic: float,
    ) -> None:
        super().__init__(run_dir, run_id, project)
        self._run_payload = run_payload
        self._started_monotonic = started_monotonic
        self._closed = False
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def __enter__(self) -> ManagedRun:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if exc_type is None:
            self.finish()
        else:
            self.finish(status="interrupted", exit_code=1)
        return False

    def finish(self, *, status: str = "completed", exit_code: int = 0) -> None:
        if self._closed:
            return
        self._closed = True
        self._heartbeat_stop.set()
        self._heartbeat_thread.join(timeout=2)
        wall = time.monotonic() - self._started_monotonic
        if status == "completed":
            (self.run_dir / "heartbeat.json").unlink(missing_ok=True)

        from mechledger.artifacts import auto_collect_artifacts
        from mechledger.debt_report import generate_scientific_debt_report
        from mechledger.run_auditor import (
            append_event,
            write_claim_proposal,
            write_run_json,
            write_run_ledger_row,
        )

        auto_collected = auto_collect_artifacts(self._project, self.run_id)
        finished = now_utc()
        self._run_payload.update(
            {"status": status, "finished_at": finished, "exit_code": exit_code}
        )
        write_run_json(self.run_dir, self._run_payload)
        append_event(self.run_dir, _event_for_status(status), status)
        resource_usage = {
            "wall_time_seconds": wall,
            "cpu_time_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
            "gpu_time_seconds": None,
            "peak_gpu_memory_bytes": None,
            "gpu_model": None,
            "disk_bytes_written": _directory_size(self.run_dir),
            "artifact_bytes_written": sum(item.get("byte_size") or 0 for item in auto_collected),
            "tensor_bytes_written": 0,
            "api_calls": None,
            "input_tokens": None,
            "output_tokens": None,
            "estimated_cost_usd": None,
            "energy_estimate_kwh": None,
        }
        _write_json(self.run_dir / "resource_usage.json", resource_usage)
        _write_json(
            self.run_dir / "summary.json",
            {
                "run_id": self.run_id,
                "status": status,
                "exit_code": exit_code,
                "artifact_count": len(auto_collected),
                "stdout_bytes": 0,
                "stderr_bytes": 0,
            },
        )
        write_run_ledger_row(self.run_dir, self._run_payload, status)
        write_claim_proposal(self._project, self.run_id)
        generate_scientific_debt_report(self._project, self.run_id)

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.is_set():
            _write_json(
                self.run_dir / "heartbeat.json",
                {"last_heartbeat_at": now_utc(), "pid": os.getpid()},
            )
            self._heartbeat_stop.wait(30)


def start(
    *,
    experiment: str | None = None,
    run_class: str = "notebook_exploration",
    purpose: str | None = None,
    hypothesis: str | None = None,
    run_id: str | None = None,
    **metadata: Any,
) -> ManagedRun:
    from mechledger.alias import append_alias
    from mechledger.run_auditor import (
        ALLOWED_RUN_CLASSES,
        append_event,
        captured_environment,
        generate_run_id,
        git_state,
        slugify,
        write_run_json,
    )

    project = _find_project_for_run(Path.cwd())
    if project is None:
        raise RuntimeError("MechLedger project not initialized. Run `mechledger init` first.")
    if run_class not in ALLOWED_RUN_CLASSES:
        raise ValueError(f"Unknown run class: {run_class}")
    slug_source = purpose or run_class
    generated_run_id = generate_run_id(
        project,
        experiment_id=experiment,
        slug_source=slug_source,
        run_id=run_id,
    )
    run_dir = project.runs_dir / generated_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "artifacts").mkdir()
    for name in ["events.jsonl", "metrics.jsonl", "artifacts.jsonl"]:
        (run_dir / name).write_text("", encoding="utf-8")
    _write_json(run_dir / "artifact_manifest.json", {"artifacts": []})
    (run_dir / "run_class_transition.json").write_text("[]\n", encoding="utf-8")
    (run_dir / "stdout.txt").write_text("", encoding="utf-8")
    (run_dir / "stderr.txt").write_text("", encoding="utf-8")
    (run_dir / "command.txt").write_text("SDK active run\n", encoding="utf-8")
    git = git_state(project.root)
    _write_json(run_dir / "git.json", git)
    _write_json(run_dir / "environment.json", captured_environment())
    started = now_utc()
    run_payload = {
        "run_id": generated_run_id,
        "parent_run_id": None,
        "experiment_id": experiment,
        "run_class": run_class,
        "status": "running",
        "purpose": purpose,
        "hypothesis": hypothesis,
        "command": None,
        "argv": [],
        "started_at": started,
        "finished_at": None,
        "exit_code": None,
        "git_commit": git.get("commit"),
        "git_diff_hash": git.get("diff_hash"),
        "cwd": str(project.root),
        "model": metadata.get("model"),
        "tokenizer": metadata.get("tokenizer"),
        "hook_point": metadata.get("hook_point"),
        "sae_release": metadata.get("sae_release"),
        "sae_id": metadata.get("sae_id"),
        "seed": metadata.get("seed"),
        "blocker": None,
        "pinned": False,
    }
    write_run_json(run_dir, run_payload)
    append_event(run_dir, "run_created", "SDK run directory created")
    append_event(run_dir, "run_started", "SDK active run started")
    append_alias(project, generated_run_id, experiment, slugify(slug_source)[:40] or "run")
    return ManagedRun(run_dir, generated_run_id, project, run_payload, time.monotonic())


def run(**kwargs: Any) -> ManagedRun:
    return start(**kwargs)


def current_run() -> ActiveRun:
    run_dir = os.environ.get("MECHLEDGER_RUN_DIR")
    run_id = os.environ.get("MECHLEDGER_RUN_ID")
    if not run_dir or not run_id:
        raise RuntimeError("No active MechLedger run. Use `mechledger run -- ...`.")
    return ActiveRun(Path(run_dir), run_id, _find_project_for_run(Path(run_dir)))


def log_metric(metric_name: str, value: object, **kwargs: Any) -> None:
    current_run().log_metric(metric_name, value, **kwargs)


def log_event(event_type: str, message: str, metadata: dict[str, Any] | None = None) -> None:
    current_run().log_event(event_type, message, metadata)


def log_artifact(path: str, **kwargs: Any) -> None:
    current_run().log_artifact(path, **kwargs)


def log_intervention_metadata(**metadata: Any) -> None:
    current_run().log_intervention_metadata(**metadata)


def artifacts_dir() -> Path:
    return current_run().artifacts_dir()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _find_project_for_run(start_path: Path) -> Any | None:
    from mechledger.project import find_project

    env_root = os.environ.get("MECHLEDGER_PROJECT_ROOT")
    candidates = [Path(env_root)] if env_root else []
    candidates.append(start_path)
    for candidate in candidates:
        try:
            return find_project(candidate)
        except FileNotFoundError:
            continue
    return None


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _event_for_status(status: str) -> str:
    if status == "completed":
        return "run_completed"
    if status == "interrupted":
        return "run_interrupted"
    if status == "cancelled":
        return "run_cancelled"
    return "run_failed"
