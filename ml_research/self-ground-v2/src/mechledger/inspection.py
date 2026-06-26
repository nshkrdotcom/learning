from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mechledger.core.claim_ledger import ClaimRecord, parse_claim_ledger
from mechledger.core.decision_log import DecisionRecord, parse_decision_log
from mechledger.core.experiment_spec import ExperimentSpec, parse_experiment_spec
from mechledger.core.research_log import ResearchLogEntry, parse_research_log
from mechledger.core.run_ledger import parse_run_ledger
from mechledger.project import Project


@dataclass
class ProjectSnapshot:
    project: Project
    warnings: list[str] = field(default_factory=list)
    claims: dict[str, ClaimRecord] = field(default_factory=dict)
    decisions: dict[str, DecisionRecord] = field(default_factory=dict)
    research_entries: list[ResearchLogEntry] = field(default_factory=list)
    run_ledger_rows: list[dict[str, str]] = field(default_factory=list)
    experiments: dict[str, ExperimentSpec] = field(default_factory=dict)
    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifact_metadata: list[dict[str, Any]] = field(default_factory=list)
    debts: list[dict[str, Any]] = field(default_factory=list)
    external_labels: list[dict[str, Any]] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)


def collect_project(project: Project) -> ProjectSnapshot:
    snapshot = ProjectSnapshot(project=project)
    root = project.root

    claim_path = project.resolve(project.config.default_claim_ledger)
    snapshot.claims = parse_claim_ledger(claim_path).claims

    decision_path = project.resolve(project.config.default_decision_log)
    if decision_path.exists():
        snapshot.decisions = parse_decision_log(decision_path).decisions
    else:
        snapshot.warnings.append(
            f"optional file missing: {relative_to_root(project, decision_path)}"
        )

    research_path = project.resolve(project.config.default_research_log)
    if research_path.exists():
        snapshot.research_entries = parse_research_log(research_path).entries
    else:
        snapshot.warnings.append(
            f"optional file missing: {relative_to_root(project, research_path)}"
        )

    run_ledger_path = project.resolve(project.config.default_run_ledger)
    if run_ledger_path.exists():
        snapshot.run_ledger_rows = parse_run_ledger(run_ledger_path).rows
    else:
        snapshot.warnings.append(
            f"optional file missing: {relative_to_root(project, run_ledger_path)}"
        )

    experiments_root = root / "research/experiments"
    if experiments_root.exists():
        for path in sorted(experiments_root.glob("**/*.md")):
            if path.name.startswith("TEMPLATE_"):
                continue
            spec = parse_experiment_spec(path)
            snapshot.experiments[spec.experiment_id] = spec

    if project.runs_dir.exists():
        for run_dir in sorted(item for item in project.runs_dir.iterdir() if item.is_dir()):
            run_json = run_dir / "run.json"
            if not run_json.exists():
                snapshot.warnings.append(
                    f"run directory missing run.json: {relative_to_root(project, run_dir)}"
                )
                continue
            run_payload = read_json_object(run_json)
            run_id = str(run_payload.get("run_id") or run_dir.name)
            snapshot.runs[run_id] = {
                "run_id": run_id,
                "path": relative_to_root(project, run_dir) + "/",
                "run_dir": run_dir,
                "run": run_payload,
                "artifacts": [],
                "debt_report": None,
            }
            manifest_path = run_dir / "artifact_manifest.json"
            if manifest_path.exists():
                manifest = read_json_object(manifest_path)
                artifacts = manifest.get("artifacts") or []
                if not isinstance(artifacts, list):
                    raise ValueError(f"{manifest_path}: `artifacts` must be a list.")
                for artifact in artifacts:
                    if isinstance(artifact, dict):
                        record = dict(artifact)
                        record["run_id"] = run_id
                        record["manifest_path"] = relative_to_root(project, manifest_path)
                        snapshot.artifact_metadata.append(record)
                        snapshot.runs[run_id]["artifacts"].append(record)
            debt_path = run_dir / "scientific_debt_report.json"
            if debt_path.exists():
                debt_report = read_json_object(debt_path)
                snapshot.runs[run_id]["debt_report"] = debt_report
                for debt in debt_report.get("debts") or []:
                    if isinstance(debt, dict):
                        record = dict(debt)
                        record.setdefault("run_id", run_id)
                        record["report_path"] = relative_to_root(project, debt_path)
                        snapshot.debts.append(record)

    labels_path = root / "research/literature/external_labels.jsonl"
    if labels_path.exists():
        snapshot.external_labels = read_jsonl(labels_path)

    for record_path in record_paths(root):
        payload = read_structured_record(record_path)
        payload["file"] = relative_to_root(project, record_path)
        snapshot.records.append(payload)
    snapshot.records.sort(key=lambda item: str(item.get("record_id") or item.get("file")))
    return snapshot


def record_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for base in (root / "research/records", root / "research/portfolio"):
        if base.exists():
            paths.extend(sorted(base.glob("**/*.json")))
            paths.extend(sorted(base.glob("**/*.yaml")))
            paths.extend(sorted(base.glob("**/*.yml")))
    return sorted({path.resolve(): path for path in paths}.values())


def read_structured_record(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        return read_json_object(path)
    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: record must be a mapping.")
    return dict(payload)


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: malformed JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object.")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: malformed JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: JSONL row must be an object.")
        rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_root(project: Project, path: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(project.root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
