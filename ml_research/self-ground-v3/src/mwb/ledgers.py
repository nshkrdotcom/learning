from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from mwb.git_state import capture_git_state
from mwb.refs import slugify
from mwb.sqlite_index import initialize_schema, insert_payload
from mwb.time import utc_now
from mwb.workflows.runs import resolve_run_path

RUN_LEDGER_COLUMNS = [
    "date",
    "run_id",
    "git_commit",
    "phase",
    "purpose",
    "hypothesis",
    "command",
    "model",
    "hook_point",
    "sae_release",
    "sae_id",
    "ranking_dir",
    "out_dir",
    "seed",
    "per_family",
    "top_k_features",
    "baseline_mode",
    "operations",
    "status",
    "blocker",
    "key_metric_1",
    "key_metric_2",
    "artifact_paths",
    "decision",
]

CLAIM_STATUSES = {
    "no_real_evidence",
    "single_run_evidence",
    "candidate_evidence",
    "causal_evidence",
    "replicated_evidence",
    "weakened",
    "contradicted",
    "insufficient_evidence",
}
DECISION_STATUSES = {"proposed", "accepted", "superseded", "rejected"}

CLAIM_HEADING_RE = re.compile(r"^###\s+(C[0-9]+[A-Za-z0-9_-]*)\s+[-\u2014]\s+(.+?)\s*$")
DECISION_HEADING_RE = re.compile(r"^##\s+(D[0-9]+[A-Za-z0-9_-]*)\s+[-\u2014]\s+(.+?)\s*$")
RESEARCH_HEADING_RE = re.compile(r"^##\s+([0-9]{4}-[0-9]{2}-[0-9]{2})\s*$")


def ensure_research_scaffold(project: Any) -> None:
    research = project.root / "research"
    for relative in [
        "experiments",
        "logs",
        "literature",
        "paper",
        "bundles",
        "reference_tasks",
    ]:
        (research / relative).mkdir(parents=True, exist_ok=True)
    logs = research / "logs"
    _write_if_missing(logs / "claim_ledger.md", "# Claim Ledger\n")
    _write_if_missing(logs / "decision_log.md", "# Decision Log\n")
    _write_if_missing(logs / "research_log.md", "# Research Log\n")
    _write_if_missing(logs / "run_ledger.csv", ",".join(RUN_LEDGER_COLUMNS) + "\n")


def validate_ledgers(
    project: Any,
    *,
    sqlite_path: Path | None = None,
    index: bool = True,
) -> dict[str, Any]:
    ensure_research_scaffold(project)
    parsed = parse_research_ledgers(project.root / "research" / "logs")
    if parsed["errors"]:
        return {"status": "fail", "counts": _counts(parsed), "errors": parsed["errors"]}
    if index:
        target_sqlite = sqlite_path or project.sqlite_path
        initialize_schema(target_sqlite)
        index_ledgers(target_sqlite, parsed)
    return {"status": "ok", "counts": _counts(parsed), "errors": []}


def parse_research_ledgers(logs_dir: Path) -> dict[str, Any]:
    claims, claim_errors = parse_claim_ledger(logs_dir / "claim_ledger.md")
    decisions, decision_errors = parse_decision_log(logs_dir / "decision_log.md")
    research_entries, research_errors = parse_research_log(logs_dir / "research_log.md")
    run_rows, run_errors = parse_run_ledger(logs_dir / "run_ledger.csv")
    return {
        "claims": claims,
        "decisions": decisions,
        "research_log_entries": research_entries,
        "run_ledger_rows": run_rows,
        "errors": [*claim_errors, *decision_errors, *research_errors, *run_errors],
    }


def index_ledgers(sqlite_path: Path, parsed: dict[str, Any]) -> dict[str, int]:
    counts = _counts(parsed)
    for claim in parsed["claims"]:
        insert_payload(sqlite_path, "claims", claim["claim_id"], claim)
    for decision in parsed["decisions"]:
        insert_payload(sqlite_path, "decisions", decision["decision_id"], decision)
    for entry in parsed["research_log_entries"]:
        insert_payload(sqlite_path, "research_log_entries", entry["entry_id"], entry)
    for row in parsed["run_ledger_rows"]:
        insert_payload(sqlite_path, "runs", row["run_id"], row)
    return counts


def parse_claim_ledger(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries, errors = _parse_markdown_yaml_entries(
        path,
        heading_re=CLAIM_HEADING_RE,
        id_field="claim_id",
        title_field="title",
    )
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        claim_id = str(entry["claim_id"])
        payload = entry["payload"]
        _require_field(path, entry, payload, "status", errors)
        _require_field(path, entry, payload, "allowed", errors)
        _require_field(path, entry, payload, "forbidden", errors)
        if payload.get("status") not in CLAIM_STATUSES:
            _error(
                path,
                entry["line"],
                claim_id,
                f"unknown claim status: {payload.get('status')}",
                errors,
            )
        _require_string_list(path, entry, payload, "allowed", errors)
        _require_string_list(path, entry, payload, "forbidden", errors)
        for optional in [
            "required_caveats",
            "debt_flags",
            "linked_experiments",
            "linked_runs",
            "linked_decisions",
            "tags",
        ]:
            if optional in payload:
                _require_string_list(path, entry, payload, optional, errors)
        if claim_id in seen:
            _error(path, entry["line"], claim_id, "duplicate claim ID", errors)
        seen.add(claim_id)
        valid.append(
            {
                **payload,
                "claim_id": claim_id,
                "title": str(payload.get("title") or entry["title"]),
                "ledger_kind": "claim_ledger",
                "source_path": str(path),
                "source_line": entry["line"],
            }
        )
    return valid, errors


def parse_decision_log(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries, errors = _parse_markdown_yaml_entries(
        path,
        heading_re=DECISION_HEADING_RE,
        id_field="decision_id",
        title_field="title",
    )
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        decision_id = str(entry["decision_id"])
        payload = entry["payload"]
        _require_field(path, entry, payload, "status", errors)
        if payload.get("status") not in DECISION_STATUSES:
            _error(
                path,
                entry["line"],
                decision_id,
                f"unknown decision status: {payload.get('status')}",
                errors,
            )
        for optional in ["affected_experiments", "affected_claims"]:
            if optional in payload:
                _require_string_list(path, entry, payload, optional, errors)
        if decision_id in seen:
            _error(path, entry["line"], decision_id, "duplicate decision ID", errors)
        seen.add(decision_id)
        valid.append(
            {
                **payload,
                "decision_id": decision_id,
                "title": str(payload.get("title") or entry["title"]),
                "ledger_kind": "decision_log",
                "source_path": str(path),
                "source_line": entry["line"],
            }
        )
    return valid, errors


def parse_research_log(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries, errors = _parse_markdown_yaml_entries(
        path,
        heading_re=RESEARCH_HEADING_RE,
        id_field="entry_id",
        title_field="date",
        heading_id_is_payload=False,
    )
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        payload = entry["payload"]
        _require_field(path, entry, payload, "entry_id", errors)
        entry_id = str(payload.get("entry_id", ""))
        for optional in ["linked_runs", "linked_claims", "linked_decisions", "open_questions"]:
            if optional in payload:
                _require_string_list(path, entry, payload, optional, errors)
        if entry_id in seen:
            _error(path, entry["line"], entry_id, "duplicate research log entry ID", errors)
        seen.add(entry_id)
        valid.append(
            {
                **payload,
                "date": entry["date"],
                "ledger_kind": "research_log",
                "source_path": str(path),
                "source_line": entry["line"],
            }
        )
    return valid, errors


def parse_run_ledger(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    if not path.exists():
        return [], [_plain_error(path, None, None, "missing run ledger")]
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RUN_LEDGER_COLUMNS:
            return [], [
                _plain_error(
                    path,
                    1,
                    None,
                    "run ledger header must match the canonical column order",
                )
            ]
        rows = list(reader)
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        run_id = str(row.get("run_id") or "")
        if not run_id:
            _error(path, index, None, "missing run_id", errors)
            continue
        if run_id in seen:
            _error(path, index, run_id, "duplicate run_id", errors)
        seen.add(run_id)
        valid.append(
            {
                **{column: row.get(column, "") for column in RUN_LEDGER_COLUMNS},
                "ledger_kind": "run_ledger",
                "source_path": str(path),
                "source_line": index,
            }
        )
    return valid, errors


def propose_run_ledger_row(project: Any, run_ref_or_path: str) -> dict[str, Any]:
    run_dir = resolve_run_path(Path(run_ref_or_path), project=project)
    manifest = _read_json(run_dir / "run_manifest.json")
    if not manifest:
        raise FileNotFoundError(f"run manifest not found for {run_ref_or_path}")
    run_ref = str(manifest.get("run_ref") or run_dir.name)
    blocker_report = _read_json(run_dir / "blocker_report.json")
    metrics = _read_json(run_dir / "control_metrics.json")
    git_state = capture_git_state(project.root)
    row = {column: "" for column in RUN_LEDGER_COLUMNS}
    row.update(
        {
            "date": _date_from_timestamp(str(manifest.get("created_at") or utc_now())),
            "run_id": run_ref,
            "git_commit": str(git_state.get("commit") or ""),
            "phase": str(manifest.get("source_kind") or ""),
            "purpose": str(manifest.get("evidence_posture") or manifest.get("status") or ""),
            "hypothesis": str(manifest.get("source_hypothesis_ref") or ""),
            "command": str(manifest.get("command") or ""),
            "model": str(manifest.get("model") or ""),
            "hook_point": str(manifest.get("hook_point") or ""),
            "sae_release": str(manifest.get("sae_release") or ""),
            "sae_id": str(manifest.get("sae_id") or ""),
            "out_dir": str(run_dir.relative_to(project.root)),
            "operations": _operations_from_manifest(manifest),
            "status": str(manifest.get("status") or ""),
            "blocker": str(blocker_report.get("primary_blocker") or ""),
            "key_metric_1": _metric_pair(metrics, "target_delta"),
            "key_metric_2": _metric_pair(metrics, "matched_control_delta"),
            "artifact_paths": str(run_dir.relative_to(project.root)),
        }
    )
    path = run_dir / "run_ledger_row.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    return {"status": "ok", "run_ref": run_ref, "proposal_path": str(path), "row": row}


def propose_claim_update(project: Any, card_ref_or_path: str) -> dict[str, Any]:
    card_path = _resolve_card_path(project, card_ref_or_path)
    card = _read_json(card_path)
    if not card:
        raise FileNotFoundError(f"MechanismCard not found for {card_ref_or_path}")
    claim_id = str(card.get("claim_ref") or card.get("metadata", {}).get("claim_ref") or "")
    if not claim_id:
        claim_id = f"C000_{slugify(str(card.get('wb_ref') or card_path.stem))}"
    title = str(card.get("title") or f"Claim for {claim_id}")
    blockers = _string_list(card.get("blockers") or card.get("metadata", {}).get("blockers"))
    payload = {
        "claim_id": claim_id,
        "title": title,
        "status": _claim_status_from_card(card),
        "scope": str(card.get("run_ref") or ""),
        "allowed": _string_list(card.get("allowed_language")),
        "forbidden": _string_list(card.get("blocked_language")),
        "required_caveats": blockers,
        "debt_flags": blockers,
        "linked_runs": [str(card["run_ref"])] if card.get("run_ref") else [],
        "linked_decisions": [],
        "source_mechanism_card_ref": str(card.get("wb_ref") or card_path.stem),
        "source_card_path": str(card_path),
    }
    proposals_dir = project.mechanism_dir / "proposals" / "claims"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    json_path = proposals_dir / f"{claim_id}.json"
    md_path = proposals_dir / f"{claim_id}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_claim_proposal(payload, card), encoding="utf-8")
    return {
        "status": "ok",
        "claim_id": claim_id,
        "proposal_json": str(json_path),
        "proposal_markdown": str(md_path),
    }


def _parse_markdown_yaml_entries(
    path: Path,
    *,
    heading_re: re.Pattern[str],
    id_field: str,
    title_field: str,
    heading_id_is_payload: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    if not path.exists():
        return [], [_plain_error(path, None, None, f"missing {path.name}")]
    lines = path.read_text(encoding="utf-8").splitlines()
    entries: list[dict[str, Any]] = []
    index = 0
    in_fence = False
    yaml = YAML(typ="safe")
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            index += 1
            continue
        if in_fence:
            index += 1
            continue
        match = heading_re.match(lines[index])
        if match is None:
            index += 1
            continue
        heading_id = match.group(1)
        heading_title = match.group(2) if len(match.groups()) > 1 else match.group(1)
        yaml_start = index + 1
        while yaml_start < len(lines) and not lines[yaml_start].strip():
            yaml_start += 1
        if yaml_start >= len(lines) or lines[yaml_start].strip() != "```yaml":
            _error(path, index + 1, heading_id, "missing YAML block", errors)
            index += 1
            continue
        yaml_end = yaml_start + 1
        while yaml_end < len(lines) and lines[yaml_end].strip() != "```":
            yaml_end += 1
        if yaml_end >= len(lines):
            _error(path, index + 1, heading_id, "unterminated YAML block", errors)
            index = yaml_start + 1
            continue
        raw_yaml = "\n".join(lines[yaml_start + 1 : yaml_end])
        try:
            payload = yaml.load(raw_yaml) or {}
        except Exception as exc:  # noqa: BLE001
            _error(path, yaml_start + 1, heading_id, f"invalid YAML: {exc}", errors)
            index = yaml_end + 1
            continue
        if not isinstance(payload, dict):
            _error(path, yaml_start + 1, heading_id, "YAML block must be a mapping", errors)
            index = yaml_end + 1
            continue
        if heading_id_is_payload:
            if payload.get(id_field) != heading_id:
                _error(path, yaml_start + 1, heading_id, f"{id_field} mismatch", errors)
            entry_id = heading_id
        else:
            entry_id = str(payload.get(id_field) or "")
        if id_field not in payload:
            _error(path, yaml_start + 1, heading_id, f"missing {id_field}", errors)
        entries.append(
            {
                id_field: entry_id,
                title_field: heading_title,
                "payload": payload,
                "line": index + 1,
            }
        )
        index = yaml_end + 1
    return entries, errors


def _resolve_card_path(project: Any, value: str) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate
    direct = project.mechanism_dir / "cards" / f"{value}.json"
    if direct.exists():
        return direct
    runs_dir = project.mechanism_dir / "runs"
    for path in sorted(runs_dir.glob("*/mechanism_card.json")) if runs_dir.exists() else []:
        payload = _read_json(path)
        if value in {str(payload.get("wb_ref")), str(payload.get("claim_ref"))}:
            return path
    raise FileNotFoundError(f"MechanismCard not found: {value}")


def _render_claim_proposal(payload: dict[str, Any], card: dict[str, Any]) -> str:
    yaml = YAML()
    from io import StringIO

    buffer = StringIO()
    yaml.dump(payload, buffer)
    return "\n".join(
        [
            f"### {payload['claim_id']} - {payload['title']}",
            "",
            "```yaml",
            buffer.getvalue().strip(),
            "```",
            "",
            "Evidence:",
            f"- Source MechanismCard: {card.get('wb_ref')}",
            f"- Evidence tier: {card.get('evidence_tier')}",
            "",
            "Contradicting evidence:",
            *[f"- {item}" for item in payload.get("debt_flags", [])],
            "",
            "Required next evidence:",
            *[f"- Resolve {item}" for item in payload.get("debt_flags", [])],
            "",
        ]
    )


def _claim_status_from_card(card: dict[str, Any]) -> str:
    status = str(card.get("status") or "")
    tier = str(card.get("evidence_tier") or "")
    if status in CLAIM_STATUSES:
        return status
    if tier in {"association", "projection"}:
        return "single_run_evidence"
    return "no_real_evidence"


def _counts(parsed: dict[str, Any]) -> dict[str, int]:
    return {
        "claims": len(parsed["claims"]),
        "decisions": len(parsed["decisions"]),
        "research_log_entries": len(parsed["research_log_entries"]),
        "run_ledger_rows": len(parsed["run_ledger_rows"]),
    }


def _require_field(
    path: Path,
    entry: dict[str, Any],
    payload: dict[str, Any],
    field: str,
    errors: list[dict[str, Any]],
) -> None:
    if field not in payload:
        _error(path, entry["line"], _entry_id(entry), f"missing {field}", errors)


def _require_string_list(
    path: Path,
    entry: dict[str, Any],
    payload: dict[str, Any],
    field: str,
    errors: list[dict[str, Any]],
) -> None:
    value = payload.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _error(path, entry["line"], _entry_id(entry), f"{field} must be a list of strings", errors)


def _entry_id(entry: dict[str, Any]) -> str | None:
    for key in ["claim_id", "decision_id", "entry_id"]:
        if entry.get(key):
            return str(entry[key])
    return None


def _error(
    path: Path,
    line: int | None,
    object_id: str | None,
    message: str,
    errors: list[dict[str, Any]],
) -> None:
    errors.append(_plain_error(path, line, object_id, message))


def _plain_error(
    path: Path,
    line: int | None,
    object_id: str | None,
    message: str,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "line": line,
        "object_id": object_id,
        "message": message,
    }


def _write_if_missing(path: Path, text: str) -> None:
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _date_from_timestamp(value: str) -> str:
    return value.split("T", 1)[0] if "T" in value else value[:10]


def _operations_from_manifest(manifest: dict[str, Any]) -> str:
    operations = manifest.get("operations")
    if isinstance(operations, list):
        return ",".join(str(item) for item in operations)
    if operations:
        return str(operations)
    tried = manifest.get("tried_axes")
    if isinstance(tried, dict) and tried.get("operations"):
        return ",".join(_string_list(tried["operations"]))
    return ""


def _metric_pair(metrics: dict[str, Any], key: str) -> str:
    return f"{key}={metrics[key]}" if key in metrics else ""


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
