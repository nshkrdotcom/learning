from __future__ import annotations

import csv
import os
import re
import secrets
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mechledger.alias import append_alias_record
from mechledger.artifacts import auto_collect_artifacts
from mechledger.git_state import captured_environment, git_metadata
from mechledger.io import append_jsonl, utc_now, write_json
from mechledger.models import RUN_LEDGER_COLUMNS
from mechledger.paths import alias_cache_path, runs_dir


@dataclass(slots=True)
class CapturedRun:
    run_id: str
    run_dir: Path
    exit_code: int


def generate_run_id(
    *,
    experiment_id: str | None = None,
    purpose: str | None = None,
    command: list[str] | None = None,
    run_class: str = "scratch",
) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    experiment = (experiment_id or "noexp").lower()
    source = purpose or (Path(command[0]).stem if command else run_class)
    slug = slugify(source)[:40] or run_class
    nonce = secrets.token_hex(3)
    run_id = f"{timestamp}_{experiment}_{slug}_{nonce}"
    return run_id[:120]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_-")
    return slug or "run"


def capture_command(
    command: list[str],
    *,
    project_root: str | Path,
    experiment_id: str | None = None,
    run_class: str = "scratch",
    purpose: str | None = None,
    hypothesis: str | None = None,
    run_id: str | None = None,
) -> CapturedRun:
    if not command:
        raise ValueError("command must not be empty")
    project_root = Path(project_root).resolve()
    run_id = run_id or generate_run_id(
        experiment_id=experiment_id, purpose=purpose, command=command, run_class=run_class
    )
    run_dir = runs_dir(project_root) / run_id
    if run_dir.exists():
        raise FileExistsError(str(run_dir))
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").touch()
    (run_dir / "events.jsonl").touch()
    (run_dir / "artifacts.jsonl").touch()
    write_json(run_dir / "artifact_manifest.json", {"artifacts": []})
    command_text = " ".join(command)
    (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")
    git = git_metadata(project_root)
    write_json(run_dir / "git.json", git)
    env_capture = captured_environment()
    write_json(run_dir / "environment.json", env_capture)
    started_at = utc_now()
    run_json = _run_json(
        run_id=run_id,
        project_root=project_root,
        experiment_id=experiment_id,
        run_class=run_class,
        purpose=purpose,
        hypothesis=hypothesis,
        command=command_text,
        started_at=started_at,
        git=git,
        status="running",
    )
    write_json(run_dir / "run.json", run_json)
    _event(run_dir, "run_created", "Run directory created.")
    _event(run_dir, "run_started", "Command started.")
    write_json(run_dir / "heartbeat.json", {"last_heartbeat_at": utc_now(), "pid": os.getpid()})
    env = dict(os.environ)
    env["MECHLEDGER_RUN_ID"] = run_id
    env["MECHLEDGER_RUN_DIR"] = str(run_dir)
    env["MECHLEDGER_ARTIFACTS_DIR"] = str(artifacts_dir)
    wall_start = time.monotonic()
    cpu_start = time.process_time()
    result = subprocess.run(
        command,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    wall_time = time.monotonic() - wall_start
    cpu_time = time.process_time() - cpu_start
    (run_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    auto_collect_artifacts(run_dir)
    status = "completed" if result.returncode == 0 else "failed"
    finished_at = utc_now()
    run_json.update({"status": status, "finished_at": finished_at, "exit_code": result.returncode})
    write_json(run_dir / "run.json", run_json)
    write_json(
        run_dir / "resource_usage.json",
        {
            "wall_time_seconds": wall_time,
            "cpu_time_seconds": cpu_time,
            "gpu_time_seconds": None,
            "peak_gpu_memory_bytes": None,
            "gpu_model": None,
            "disk_bytes_written": _directory_size(run_dir),
            "artifact_bytes_written": _directory_size(artifacts_dir),
            "tensor_bytes_written": 0,
            "api_calls": None,
            "input_tokens": None,
            "output_tokens": None,
            "estimated_cost_usd": None,
            "energy_estimate_kwh": None,
        },
    )
    write_json(
        run_dir / "summary.json",
        {"run_id": run_id, "status": status, "exit_code": result.returncode, "artifact_count": 0},
    )
    _write_run_ledger_row(run_dir, run_json)
    _event(run_dir, "run_completed" if result.returncode == 0 else "run_failed", f"Run {status}.")
    (run_dir / "heartbeat.json").unlink(missing_ok=True)
    append_alias_record(
        alias_cache_path(project_root), run_id, experiment_id, slugify(purpose or command[0])
    )
    return CapturedRun(run_id=run_id, run_dir=run_dir, exit_code=result.returncode)


def _run_json(
    *,
    run_id: str,
    project_root: Path,
    experiment_id: str | None,
    run_class: str,
    purpose: str | None,
    hypothesis: str | None,
    command: str | None,
    started_at: str,
    git: dict,
    status: str,
) -> dict:
    return {
        "run_id": run_id,
        "parent_run_id": None,
        "experiment_id": experiment_id,
        "run_class": run_class,
        "status": status,
        "purpose": purpose,
        "hypothesis": hypothesis,
        "command": command,
        "started_at": started_at,
        "finished_at": None,
        "exit_code": None,
        "git_commit": git.get("git_commit"),
        "git_diff_hash": git.get("git_diff_hash"),
        "cwd": str(project_root),
        "model": None,
        "tokenizer": None,
        "hook_point": None,
        "sae_release": None,
        "sae_id": None,
        "seed": None,
        "blocker": None,
        "pinned": False,
    }


def _event(run_dir: Path, event_type: str, message: str, metadata: dict | None = None) -> None:
    append_jsonl(
        run_dir / "events.jsonl",
        {
            "timestamp": utc_now(),
            "event_type": event_type,
            "message": message,
            "metadata": metadata or {},
        },
    )


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _write_run_ledger_row(run_dir: Path, run_json: dict) -> None:
    row = {column: "" for column in RUN_LEDGER_COLUMNS}
    row.update(
        {
            "date": run_json["started_at"][:10],
            "run_id": run_json["run_id"],
            "git_commit": run_json.get("git_commit") or "",
            "phase": run_json.get("experiment_id") or "",
            "purpose": run_json.get("purpose") or "",
            "hypothesis": run_json.get("hypothesis") or "",
            "command": run_json.get("command") or "",
            "status": run_json.get("status") or "",
            "blocker": run_json.get("blocker") or "",
        }
    )
    with (run_dir / "run_ledger_row.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
