from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from attention_lab.queue.ledger import QueueLedger
from attention_lab.training.experiments import get_experiment


RUN_INDEX_FIELDS = (
    "id",
    "config_name",
    "config_path",
    "run_dir",
    "attention_type",
    "stage",
    "status",
    "failure_class",
    "step_reached",
    "final_val_loss",
    "best_val_loss",
    "final_ppl",
    "median_tokens_per_sec",
    "peak_vram_allocated_mb",
    "hellaswag_acc",
    "mechanism_active",
    "notes",
)


def export_queue_report(
    *,
    experiment_id: str,
    ledger: QueueLedger,
    repo_root: str | Path = ".",
) -> dict[str, Path | int]:
    repo_root = Path(repo_root)
    experiment = get_experiment(experiment_id)
    report_dir = repo_root / experiment["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [_select_fields(row) for row in ledger.list_runs() if _belongs_to_experiment(row, experiment)]
    if not rows:
        rows = _config_rows(repo_root / experiment["config_dir"])

    json_path = report_dir / "run_index.json"
    md_path = report_dir / "run_index.md"
    import json

    json_path.write_text(json.dumps({"experiment_id": experiment_id, "runs": rows}, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_run_index_markdown(experiment_id, rows), encoding="utf-8")
    return {"json_path": json_path, "markdown_path": md_path, "row_count": len(rows)}


def append_decision_log(
    *,
    experiment_id: str,
    shows: str,
    not_shows: str,
    next_step: str,
    repo_root: str | Path = ".",
) -> Path:
    if not shows.strip() or not not_shows.strip() or not next_step.strip():
        raise ValueError("--shows, --not-shows, and --next must be nonempty")
    repo_root = Path(repo_root)
    experiment = get_experiment(experiment_id)
    report_dir = repo_root / experiment["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "decision_log.md"
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    entry = (
        f"## {timestamp}\n\n"
        f"SHOWS:\n{shows.strip()}\n\n"
        f"NOT_SHOWS:\n{not_shows.strip()}\n\n"
        f"NEXT:\n{next_step.strip()}\n\n"
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(entry)
    return path


def _belongs_to_experiment(row: dict[str, Any], experiment: dict[str, Any]) -> bool:
    run_dir = str(row.get("run_dir") or "")
    config_path = str(row.get("config_path") or "")
    return run_dir.startswith(experiment["run_dir"]) or config_path.startswith(experiment["config_dir"])


def _select_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in RUN_INDEX_FIELDS}


def _config_rows(config_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for config_path in sorted(config_dir.glob("*.yaml")):
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        rows.append(
            {
                "id": config_path.stem,
                "config_name": config_path.stem,
                "config_path": str(config_path),
                "run_dir": config.get("run", {}).get("out_dir"),
                "attention_type": config.get("model", {}).get("attention_type"),
                "stage": None,
                "status": "NOT_QUEUED",
                "failure_class": None,
                "step_reached": None,
                "final_val_loss": None,
                "best_val_loss": None,
                "final_ppl": None,
                "median_tokens_per_sec": None,
                "peak_vram_allocated_mb": None,
                "hellaswag_acc": None,
                "mechanism_active": None,
                "notes": "config present; no queue ledger row",
            }
        )
    return rows


def _render_run_index_markdown(experiment_id: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Queue Run Index: {experiment_id}",
        "",
        "This file is exported from the queue ledger. It is an operational index, not a scientific interpretation.",
        "",
        "| run | attention | stage | status | failure | step | final loss | best loss | ppl | tok/s | vram MB | hs | active | notes |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row.get("config_name")),
                    _md(row.get("attention_type")),
                    _md(row.get("stage")),
                    _md(row.get("status")),
                    _md(row.get("failure_class")),
                    _md(row.get("step_reached")),
                    _md(row.get("final_val_loss")),
                    _md(row.get("best_val_loss")),
                    _md(row.get("final_ppl")),
                    _md(row.get("median_tokens_per_sec")),
                    _md(row.get("peak_vram_allocated_mb")),
                    _md(row.get("hellaswag_acc")),
                    _md(row.get("mechanism_active")),
                    _md(row.get("notes")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _md(value: Any) -> str:
    if value is None or value == "":
        return "---"
    return str(value).replace("|", "\\|").replace("\n", "<br>")
