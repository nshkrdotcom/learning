from __future__ import annotations

import json
import math
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from attention_lab.queue.ledger import QueueLedger
from attention_lab.queue.runner import CommandRunner, CommandResult, default_command_runner
from attention_lab.queue.state_files import clear_active, copy_to_active, finalize_config
from attention_lab.training.config import load_config, save_config


@dataclass(frozen=True)
class ScreenVerdict:
    passed: bool
    failure_class: str | None
    mechanism_active: bool | None
    step_reached: int | None


def screen_config_with_overrides(config: dict, screen_run_dir: str | Path) -> dict:
    screened = deepcopy(config)
    screened["run"]["out_dir"] = str(screen_run_dir)
    screened["train"]["max_steps"] = 150
    screened["train"]["val_every"] = 50
    screened["train"]["save_every"] = 150
    return screened


def load_metrics(metrics_path: str | Path) -> list[dict]:
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return []
    rows = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def step_reached(metrics: list[dict]) -> int | None:
    steps = [int(row["step"]) for row in metrics if row.get("step") is not None]
    return max(steps) if steps else None


def mechanism_active_from_diagnostics(attention_type: str, diagnostics_path: str | Path) -> bool | None:
    if attention_type == "standard":
        return None
    diagnostics_path = Path(diagnostics_path)
    if not diagnostics_path.exists():
        return None
    saw_row = False
    with diagnostics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            saw_row = True
            row = json.loads(line)
            grad_norm = row.get("cp_gradient_norm")
            if grad_norm is not None and float(grad_norm) > 1e-6:
                return True
    return False if saw_row else None


def _val_losses(metrics: list[dict]) -> list[tuple[int, float]]:
    losses = []
    for row in metrics:
        if row.get("event") != "val" or row.get("val_loss") is None:
            continue
        losses.append((int(row.get("step", 0)), float(row["val_loss"])))
    return losses


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def classify_screen_result(
    *,
    returncode: int,
    stderr: str,
    metrics: list[dict],
    attention_type: str,
    diagnostics_path: str | Path | None = None,
    baseline_tokens_per_sec: float | None = None,
) -> ScreenVerdict:
    reached = step_reached(metrics)
    stderr_lower = stderr.lower()
    if "out of memory" in stderr_lower or "cuda oom" in stderr_lower:
        return ScreenVerdict(False, "OOM", None, reached)
    if returncode != 0 and (reached is None or reached < 10):
        return ScreenVerdict(False, "COMPILE_ERROR", None, reached)

    losses = _val_losses(metrics)
    if any(not math.isfinite(loss) for _, loss in losses):
        return ScreenVerdict(False, "NAN", None, reached)

    if len(losses) >= 2:
        early_loss = next((loss for step, loss in losses if step >= 10), losses[0][1])
        final_loss = losses[-1][1]
        if final_loss > early_loss * 0.97:
            return ScreenVerdict(False, "FLAT_LOSS", None, reached)

    mechanism_active = (
        mechanism_active_from_diagnostics(attention_type, diagnostics_path) if diagnostics_path is not None else None
    )
    if mechanism_active is False:
        return ScreenVerdict(False, "DEAD_GRAD", mechanism_active, reached)

    tokens_per_sec = [
        float(row["tokens_per_sec"])
        for row in metrics
        if row.get("event") == "train" and row.get("tokens_per_sec") is not None
    ]
    median_tokens_per_sec = _median(tokens_per_sec)
    if (
        baseline_tokens_per_sec is not None
        and median_tokens_per_sec is not None
        and median_tokens_per_sec < baseline_tokens_per_sec * 0.30
    ):
        return ScreenVerdict(False, "SLOW", mechanism_active, reached)

    return ScreenVerdict(True, None, mechanism_active, reached)


def run_screen(
    row: dict,
    ledger: QueueLedger,
    *,
    command_runner: CommandRunner = default_command_runner,
    keep_screens: bool = False,
) -> dict:
    run_id = row["id"]
    config_path = Path(row["config_path"])
    config = load_config(config_path)
    screen_run_dir = Path("runs") / "screen" / f"{row['config_name']}_{run_id}"
    screen_config = screen_config_with_overrides(config, screen_run_dir)
    screen_config_path = screen_run_dir / "screen_config.yaml"
    screen_run_dir.mkdir(parents=True, exist_ok=True)
    save_config(screen_config, screen_config_path)

    ledger.mark_started(run_id)
    copy_to_active(config_path)
    log_path = screen_run_dir / "queue_screen.log"
    cmd = ["uv", "run", "scripts/train.py", "--config", str(screen_config_path), "--overwrite"]
    result: CommandResult = command_runner(cmd, log_path)
    metrics = load_metrics(screen_run_dir / "metrics.jsonl")
    verdict = classify_screen_result(
        returncode=result.returncode,
        stderr=result.stderr,
        metrics=metrics,
        attention_type=config["model"].get("attention_type", "standard"),
        diagnostics_path=screen_run_dir / "evals" / "attention_diagnostics.jsonl",
        baseline_tokens_per_sec=ledger.get_baseline_screen_tokens_per_sec(),
    )

    if verdict.passed:
        if config["model"].get("attention_type", "standard") == "standard":
            train_speeds = [
                float(row["tokens_per_sec"])
                for row in metrics
                if row.get("event") == "train" and row.get("tokens_per_sec") is not None
            ]
            median_speed = _median(train_speeds)
            if median_speed is not None:
                ledger.update_baseline_screen_tokens_per_sec(median_speed)
        ledger.promote_to_full(run_id)
        clear_active(config_path)
    else:
        ledger.mark_failed(
            run_id,
            failure_class=verdict.failure_class or "UNKNOWN",
            killed=True,
            step_reached=verdict.step_reached,
            mechanism_active=verdict.mechanism_active,
        )
        finalize_config(config_path, "done" if verdict.failure_class != "COMPILE_ERROR" else "failed")

    if not keep_screens:
        shutil.rmtree(screen_run_dir, ignore_errors=True)
    return {
        "ok": verdict.passed,
        "failure_class": verdict.failure_class,
        "mechanism_active": verdict.mechanism_active,
        "step_reached": verdict.step_reached,
    }
