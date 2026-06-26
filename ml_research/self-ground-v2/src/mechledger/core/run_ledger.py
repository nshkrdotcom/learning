from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mechledger.core.diagnostics import raise_diagnostic
from mechledger.project import run_ledger_header

DEFAULT_RUN_LEDGER_COLUMNS = run_ledger_header().split(",")
RUN_LEDGER_STATUSES = {
    "created",
    "running",
    "completed",
    "interrupted",
    "interrupted_indexed",
    "blocked",
    "failed",
    "cancelled",
    "draft",
    "planned",
    "candidate_claim",
    "single_run_evidence",
    "multi_run_evidence",
    "insufficient_evidence",
    "candidate_evidence",
    "strong_candidate_evidence",
    "failed_or_weakened",
    "contradicted",
    "retired",
}


class RunLedger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    rows: list[dict[str, str]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def parse_run_ledger(path: str | Path) -> RunLedger:
    path = Path(path)
    if not path.exists():
        return RunLedger(path=str(path), rows=[])
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != DEFAULT_RUN_LEDGER_COLUMNS:
            raise_diagnostic(
                file=str(path),
                line=1,
                code="run_ledger.header.invalid",
                message="Run ledger CSV header does not match the MechLedger default columns.",
                suggested_fix="replace the header with the documented default run ledger columns.",
            )
        rows = list(reader)
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        run_id = row.get("run_id") or ""
        if not run_id:
            raise_diagnostic(
                file=str(path),
                line=line_number,
                code="run_ledger.run_id.missing",
                message="Run ledger row is missing run_id.",
                suggested_fix="write the canonical run_id in the run_id column.",
            )
        if run_id in seen:
            raise_diagnostic(
                file=str(path),
                line=line_number,
                object_id=run_id,
                code="run_ledger.run_id.duplicate",
                message="Run ledger has duplicate canonical run_id.",
                suggested_fix="remove the duplicate row or correct the run_id.",
            )
        seen.add(run_id)
        status = row.get("status") or ""
        if status and status not in RUN_LEDGER_STATUSES:
            raise_diagnostic(
                file=str(path),
                line=line_number,
                object_id=run_id,
                code="run_ledger.status.invalid",
                message=f"Run ledger row has invalid status: {status}",
                suggested_fix=(
                    "use one of "
                    + ", ".join(sorted(RUN_LEDGER_STATUSES))
                    + " in the status column."
                ),
            )
    return RunLedger(path=str(path), rows=rows)
