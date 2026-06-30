from __future__ import annotations

from datetime import datetime
from typing import Any


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "---"
    return f"{float(value):.{digits}f}"


def _fmt_speed(value: Any) -> str:
    if value is None or value == "":
        return "---"
    value = float(value)
    if value >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{value:.0f}"


def _fmt_vram(value: Any) -> str:
    if value is None or value == "":
        return "---"
    return f"{float(value) / 1024:.1f}G"


def _fmt_hot(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return "1" if int(value) else "0"


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(0, width - 1)] + "~"


def render_leaderboard(rows: list[dict[str, Any]], *, now: str | None = None) -> str:
    now = now or datetime.now().strftime("%Y-%m-%d %H:%M")
    running = next((row for row in rows if row.get("status") == "RUNNING"), None)
    running_text = "none"
    if running is not None:
        step = running.get("step_reached") or "---"
        running_text = f"{running.get('config_name')} step {step}"

    lines = [
        f"QUEUE STATUS  {now}  |  running: {running_text}",
        "=" * 96,
        "RUNNING / RECENT",
        "  run_name                             attn_type       stage  status    loss    ppl    tok/s  vram  hs  notes",
    ]
    recent = [row for row in rows if row.get("status") in {"RUNNING", "PASSED"}]
    if not recent:
        lines.append("  ---")
    for row in recent[-12:]:
        lines.append(_format_row(row))

    pending = [row for row in rows if row.get("status") == "PENDING"]
    lines.extend(["", f"PENDING ({len(pending)} in queue)"])
    if not pending:
        lines.append("  ---")
    for row in pending[:20]:
        lines.append(f"  {_truncate(row.get('config_name', ''), 36):36} {row.get('attention_type') or '---':14} {row.get('stage'):6} PENDING")

    failed = [row for row in rows if row.get("status") in {"FAILED", "KILLED"}]
    lines.extend(["", "KILLED / FAILED (recent)"])
    if not failed:
        lines.append("  ---")
    for row in failed[-12:]:
        lines.append(_format_row(row, include_failure=True))
    lines.extend(["", "=" * 96, "hs = mechanism_active  |  to add: attn-queue add configs/experiments/<ID>/..."])
    return "\n".join(lines) + "\n"


def _format_row(row: dict[str, Any], *, include_failure: bool = False) -> str:
    notes = row.get("failure_class") if include_failure and row.get("failure_class") else row.get("notes") or ""
    return (
        f"  {_truncate(row.get('config_name', ''), 36):36} "
        f"{_truncate(row.get('attention_type') or '---', 14):14} "
        f"{row.get('stage') or '---':6} "
        f"{row.get('status') or '---':8} "
        f"{_fmt_float(row.get('final_val_loss')):7} "
        f"{_fmt_float(row.get('final_ppl'), digits=1):6} "
        f"{_fmt_speed(row.get('median_tokens_per_sec')):6} "
        f"{_fmt_vram(row.get('peak_vram_allocated_mb')):5} "
        f"{_fmt_hot(row.get('mechanism_active')):2} "
        f"{_truncate(notes, 40)}"
    )
