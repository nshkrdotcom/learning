from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import resource
import secrets
import shutil
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mechledger.alias import append_alias
from mechledger.artifacts import auto_collect_artifacts
from mechledger.debt_report import generate_scientific_debt_report
from mechledger.project import Project, now_utc, run_ledger_header
from mechledger.redaction_policy import redact_environment

RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,120}$")
ALLOWED_RUN_CLASSES = {
    "scratch",
    "notebook_exploration",
    "path_validation",
    "smoke_test",
    "diagnostic",
    "calibration",
    "benchmark",
    "serious_evidence_run",
    "paper_candidate",
    "replication",
    "published_result",
}
def generate_run_id(
    project: Project,
    *,
    experiment_id: str | None,
    slug_source: str,
    run_id: str | None = None,
) -> str:
    if run_id:
        validate_run_id(run_id)
        if (project.runs_dir / run_id).exists():
            raise FileExistsError(
                f"Run directory already exists for user-provided run ID: {run_id}"
            )
        return run_id
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    experiment_component = (experiment_id or "noexp").lower()
    slug = slugify(slug_source)[:40] or "run"
    for _ in range(20):
        candidate = f"{timestamp}_{experiment_component}_{slug}_{secrets.token_hex(4)}"
        validate_run_id(candidate)
        if not (project.runs_dir / candidate).exists():
            return candidate
    raise RuntimeError("could not generate a collision-free run ID")


def validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "Invalid run ID. Use ASCII letters, digits, underscores, or hyphens; max 120 chars."
        )
    if any(char in run_id for char in ":/\\ !$`'\";&|<>*?()[]{}"):
        raise ValueError("Invalid run ID contains shell or filesystem metacharacters.")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "_", text.lower()).strip("_")
    return re.sub(r"_+", "_", slug)


def capture_run(
    project: Project,
    argv: list[str],
    *,
    experiment_id: str | None = None,
    run_class: str = "scratch",
    purpose: str | None = None,
    hypothesis: str | None = None,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, int]:
    if run_class not in ALLOWED_RUN_CLASSES:
        raise ValueError(f"Unknown run class: {run_class}")
    if not argv:
        raise ValueError("No command supplied after `mechledger run --`.")
    slug_source = purpose or Path(argv[0]).name or run_class
    run_id = generate_run_id(
        project, experiment_id=experiment_id, slug_source=slug_source, run_id=run_id
    )
    run_dir = project.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "artifacts").mkdir()
    for name in ["events.jsonl", "metrics.jsonl", "artifacts.jsonl"]:
        (run_dir / name).write_text("", encoding="utf-8")
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps({"artifacts": []}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run_class_transition.json").write_text("[]\n", encoding="utf-8")
    command_text = " ".join(argv)
    (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")
    started = now_utc()
    git = git_state(project.root)
    env = captured_environment()
    (run_dir / "git.json").write_text(
        json.dumps(git, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_payload = {
        "run_id": run_id,
        "parent_run_id": None,
        "experiment_id": experiment_id,
        "run_class": run_class,
        "status": "running",
        "purpose": purpose,
        "hypothesis": hypothesis,
        "command": command_text,
        "argv": argv,
        "started_at": started,
        "finished_at": None,
        "exit_code": None,
        "git_commit": git.get("commit"),
        "git_diff_hash": git.get("diff_hash"),
        "cwd": str(project.root),
        "model": (metadata or {}).get("model"),
        "tokenizer": (metadata or {}).get("tokenizer"),
        "hook_point": (metadata or {}).get("hook_point"),
        "sae_release": (metadata or {}).get("sae_release"),
        "sae_id": (metadata or {}).get("sae_id"),
        "seed": (metadata or {}).get("seed"),
        "blocker": None,
        "pinned": False,
    }
    write_run_json(run_dir, run_payload)
    append_event(run_dir, "run_created", "run directory created")
    append_event(run_dir, "run_started", command_text)
    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop, args=(run_dir, heartbeat_stop), daemon=True
    )
    heartbeat.start()
    process_env = os.environ.copy()
    package_src = str(Path(__file__).resolve().parents[1])
    process_env["PYTHONPATH"] = (
        package_src + os.pathsep + process_env.get("PYTHONPATH", "")
        if process_env.get("PYTHONPATH")
        else package_src
    )
    process_env["MECHLEDGER_RUN_ID"] = run_id
    process_env["MECHLEDGER_RUN_DIR"] = str(run_dir)
    process_env["MECHLEDGER_PROJECT_ROOT"] = str(project.root)
    started_monotonic = time.monotonic()
    interrupted_by_keyboard = False
    try:
        try:
            result = subprocess.run(
                argv,
                cwd=project.root,
                env=process_env,
                capture_output=True,
                text=True,
                check=False,
            )
        except KeyboardInterrupt:
            interrupted_by_keyboard = True
            result = subprocess.CompletedProcess(
                argv,
                130,
                stdout="",
                stderr="Interrupted by user.\n",
            )
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=2)
    wall = time.monotonic() - started_monotonic
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    (run_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    status = _status_from_returncode(result.returncode, interrupted_by_keyboard)
    public_exit_code = _public_exit_code(result.returncode, interrupted_by_keyboard)
    finished = now_utc()
    run_payload.update({"status": status, "finished_at": finished, "exit_code": result.returncode})
    write_run_json(run_dir, run_payload)
    (run_dir / "heartbeat.json").unlink(missing_ok=True)
    auto_collected = auto_collect_artifacts(project, run_id)
    append_event(run_dir, _event_for_status(status), status)
    resource_usage = {
        "wall_time_seconds": wall,
        "cpu_time_seconds": resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,
        "gpu_time_seconds": None,
        "peak_gpu_memory_bytes": None,
        "gpu_model": None,
        "disk_bytes_written": _directory_size(run_dir),
        "artifact_bytes_written": sum(item.get("byte_size") or 0 for item in auto_collected),
        "tensor_bytes_written": 0,
        "api_calls": None,
        "input_tokens": None,
        "output_tokens": None,
        "estimated_cost_usd": None,
        "energy_estimate_kwh": None,
    }
    (run_dir / "resource_usage.json").write_text(
        json.dumps(resource_usage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "run_id": run_id,
        "status": status,
        "exit_code": result.returncode,
        "artifact_count": len(auto_collected),
        "stdout_bytes": len(stdout.encode()),
        "stderr_bytes": len(stderr.encode()),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_run_ledger_row(run_dir, run_payload, status)
    write_claim_proposal(project, run_id)
    generate_scientific_debt_report(project, run_id)
    append_alias(project, run_id, experiment_id, slugify(slug_source)[:40] or "run")
    return run_id, public_exit_code


def write_run_json(run_dir: Path, payload: dict[str, Any]) -> None:
    (run_dir / "run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def append_event(
    run_dir: Path, event_type: str, message: str, metadata: dict[str, Any] | None = None
) -> None:
    payload = {
        "timestamp": now_utc(),
        "event_type": event_type,
        "message": message,
        "metadata": metadata or {},
    }
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_run_ledger_row(run_dir: Path, run_payload: dict[str, Any], status: str) -> None:
    header = run_ledger_header().split(",")
    row = {
        "date": run_payload["started_at"][:10],
        "run_id": run_payload["run_id"],
        "git_commit": run_payload.get("git_commit") or "",
        "phase": run_payload.get("run_class") or "",
        "purpose": run_payload.get("purpose") or "",
        "hypothesis": run_payload.get("hypothesis") or "",
        "command": run_payload.get("command") or "",
        "model": run_payload.get("model") or "",
        "hook_point": run_payload.get("hook_point") or "",
        "sae_release": run_payload.get("sae_release") or "",
        "sae_id": run_payload.get("sae_id") or "",
        "ranking_dir": "",
        "out_dir": str(run_dir),
        "seed": str(run_payload.get("seed") or ""),
        "per_family": "",
        "top_k_features": "",
        "baseline_mode": "",
        "operations": "",
        "status": status,
        "blocker": run_payload.get("blocker") or "",
        "key_metric_1": "",
        "key_metric_2": "",
        "artifact_paths": "artifact_manifest.json",
        "decision": "",
    }
    with (run_dir / "run_ledger_row.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerow(row)


def write_claim_proposal(project: Project, run_id: str) -> None:
    run_dir = project.runs_dir / run_id
    proposal = {
        "proposal_id": f"CP-{run_id}",
        "run_id": run_id,
        "generated_at": now_utc(),
        "target_claim_id": None,
        "current_claim_status_at_generation": None,
        "proposed_status": None,
        "proposed_direction": "neutral",
        "expected_claim_ledger_hash": _file_hash(
            project.root / project.config.default_claim_ledger
        ),
        "expected_claim_block_hash": None,
        "supporting_metric_names": [],
        "contradicting_metric_names": [],
        "supporting_artifact_paths": [],
        "contradicting_artifact_paths": [],
        "scientific_debt_ids": [],
        "blocking_issues": [],
        "required_human_checks": ["Review run artifacts before applying any claim update."],
        "proposed_markdown_patch_path": str(run_dir / "claim_update_proposal.md"),
        "review_status": "pending",
        "reviewed_at": None,
        "reviewed_by": None,
        "force_applied": False,
    }
    (run_dir / "claim_update_proposal.json").write_text(
        json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "claim_update_proposal.md").write_text(
        f"# Claim Update Proposal for {run_id}\n\nNo automatic claim promotion proposed.\n",
        encoding="utf-8",
    )


def git_state(root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    diff = run_git(["diff", "--binary"]) or ""
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(run_git(["status", "--porcelain"])),
        "diff_hash": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def captured_environment() -> dict[str, str]:
    return redact_environment(os.environ)


def _heartbeat_loop(run_dir: Path, stop: threading.Event) -> None:
    while not stop.is_set():
        (run_dir / "heartbeat.json").write_text(
            json.dumps({"last_heartbeat_at": now_utc(), "pid": os.getpid()}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stop.wait(30)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status_from_returncode(returncode: int, interrupted_by_keyboard: bool = False) -> str:
    if interrupted_by_keyboard:
        return "cancelled"
    if returncode == 0:
        return "completed"
    if returncode < 0:
        return "interrupted"
    return "failed"


def _public_exit_code(returncode: int, interrupted_by_keyboard: bool = False) -> int:
    if interrupted_by_keyboard:
        return 130
    if returncode < 0:
        return 128 + abs(returncode)
    return returncode


def _event_for_status(status: str) -> str:
    if status == "completed":
        return "run_completed"
    if status == "interrupted":
        return "run_interrupted"
    if status == "cancelled":
        return "run_cancelled"
    return "run_failed"


def copy_if_small(src: Path, dst: Path, *, max_bytes: int = 10_000_000) -> bool:
    if src.stat().st_size > max_bytes:
        return False
    shutil.copy2(src, dst)
    return True
