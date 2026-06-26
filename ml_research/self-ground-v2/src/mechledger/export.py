from __future__ import annotations

import gzip
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from mechledger.alias import resolve_run_id
from mechledger.inspection import (
    ProjectSnapshot,
    collect_project,
    relative_to_root,
    sha256_file,
    write_json,
)
from mechledger.project import SCHEMA_VERSION, Project

CANONICAL_CANDIDATES = [
    ".mechledger/project.json",
    "research/logs/claim_ledger.md",
    "research/logs/decision_log.md",
    "research/logs/research_log.md",
    "research/logs/run_ledger.csv",
    "research/literature/external_labels.jsonl",
]

RUN_FILES = [
    "run.json",
    "metrics.jsonl",
    "events.jsonl",
    "artifacts.jsonl",
    "artifact_manifest.json",
    "evidence_assessment.json",
    "scientific_debt_report.json",
    "scientific_debt_report.md",
    "calibration_check.json",
    "calibration_check.md",
    "telemetry_check.json",
    "telemetry_check.md",
    "null_check.json",
    "null_check.md",
    "paired_test.json",
    "paired_test.md",
]


def write_ro_crate(project: Project, out_dir: Path) -> tuple[Path, list[str]]:
    snapshot = collect_project(project)
    graph = _rocrate_graph(snapshot)
    payload = {
        "@context": {
            "@vocab": "https://schema.org/",
            "mechledger": "https://mechledger.local/schema#",
        },
        "@graph": graph,
        "mechledger:warnings": sorted(snapshot.warnings),
    }
    output = out_dir / "ro-crate-metadata.json"
    write_json(output, payload)
    return output, sorted(snapshot.warnings)


def _rocrate_graph(snapshot: ProjectSnapshot) -> list[dict[str, Any]]:
    project = snapshot.project
    graph: list[dict[str, Any]] = [
        {
            "@id": "./",
            "@type": "Dataset",
            "name": project.root.name,
            "projectId": project.config.project_id,
            "schemaVersion": project.config.schema_version,
        },
        {
            "@id": ".mechledger/project.json",
            "@type": "File",
            "encodingFormat": "application/json",
            "about": "./",
        },
    ]
    claim_ledger_id = project.config.default_claim_ledger
    graph.append({"@id": claim_ledger_id, "@type": "File", "about": "./"})
    for claim in sorted(snapshot.claims.values(), key=lambda item: item.claim_id):
        graph.append(
            {
                "@id": f"{claim_ledger_id}#{claim.claim_id}",
                "@type": "CreativeWork",
                "identifier": claim.claim_id,
                "name": claim.title or claim.heading_title,
                "claimStatus": claim.status.value,
                "scope": claim.scope,
                "linkedRuns": [
                    f".mechledger/runs/{run_id}/" for run_id in sorted(claim.linked_runs)
                ],
                "linkedDecisions": [
                    f"{project.config.default_decision_log}#{decision_id}"
                    for decision_id in sorted(claim.linked_decisions)
                ],
                "linkedExperiments": [
                    _experiment_entity_id(snapshot, experiment_id)
                    for experiment_id in sorted(claim.linked_experiments)
                    if _experiment_entity_id(snapshot, experiment_id)
                ],
                "debtFlags": sorted(claim.debt_flags),
            }
        )
    decision_log_id = project.config.default_decision_log
    if (project.root / decision_log_id).exists():
        graph.append({"@id": decision_log_id, "@type": "File", "about": "./"})
    for decision in sorted(snapshot.decisions.values(), key=lambda item: item.decision_id):
        graph.append(
            {
                "@id": f"{decision_log_id}#{decision.decision_id}",
                "@type": "CreativeWork",
                "identifier": decision.decision_id,
                "name": decision.title,
                "decisionStatus": decision.status,
                "decisionType": decision.decision_type,
                "affectedClaims": [
                    f"{claim_ledger_id}#{claim_id}"
                    for claim_id in sorted(decision.affected_claims)
                ],
                "affectedExperiments": [
                    _experiment_entity_id(snapshot, experiment_id)
                    for experiment_id in sorted(decision.affected_experiments)
                    if _experiment_entity_id(snapshot, experiment_id)
                ],
            }
        )
    research_log_id = project.config.default_research_log
    if (project.root / research_log_id).exists():
        graph.append({"@id": research_log_id, "@type": "File", "about": "./"})
    for entry in sorted(snapshot.research_entries, key=lambda item: item.entry_id):
        graph.append(
            {
                "@id": f"{research_log_id}#{entry.entry_id}",
                "@type": "CreativeWork",
                "identifier": entry.entry_id,
                "dateCreated": entry.date,
                "linkedRuns": [
                    f".mechledger/runs/{run_id}/" for run_id in sorted(entry.linked_runs)
                ],
                "linkedClaims": [
                    f"{claim_ledger_id}#{claim_id}" for claim_id in sorted(entry.linked_claims)
                ],
                "linkedDecisions": [
                    f"{decision_log_id}#{decision_id}"
                    for decision_id in sorted(entry.linked_decisions)
                ],
                "openQuestions": sorted(entry.open_questions),
            }
        )
    run_ledger_id = project.config.default_run_ledger
    if (project.root / run_ledger_id).exists():
        graph.append({"@id": run_ledger_id, "@type": "File", "about": "./"})
    for row in sorted(snapshot.run_ledger_rows, key=lambda item: item.get("run_id") or ""):
        run_id = row.get("run_id") or ""
        graph.append(
            {
                "@id": f"{run_ledger_id}#{run_id}",
                "@type": "Dataset",
                "identifier": run_id,
                "run": f".mechledger/runs/{run_id}/",
                "experiment": _experiment_entity_id(snapshot, row.get("experiment_id") or ""),
            }
        )
    for spec in sorted(snapshot.experiments.values(), key=lambda item: item.experiment_id):
        entity_id = _experiment_entity_id(snapshot, spec.experiment_id)
        graph.append(
            {
                "@id": entity_id,
                "@type": "CreativeWork",
                "identifier": spec.experiment_id,
                "name": spec.title,
                "experimentStatus": spec.status,
                "claimTargets": [
                    f"{claim_ledger_id}#{claim_id}" for claim_id in sorted(spec.claim_targets)
                ],
                "sourceRuns": [
                    f".mechledger/runs/{run_id}/" for run_id in sorted(spec.source_runs)
                ],
                "prerequisites": spec.prerequisites,
            }
        )
    for run_id, run_record in sorted(snapshot.runs.items()):
        run_data = run_record["run"]
        run_entity = f".mechledger/runs/{run_id}/"
        artifacts = [
            _artifact_entity_id(run_id, artifact)
            for artifact in sorted(
                run_record.get("artifacts") or [],
                key=lambda item: str(item.get("artifact_id") or ""),
            )
        ]
        graph.append(
            {
                "@id": run_entity,
                "@type": "Dataset",
                "identifier": run_id,
                "runStatus": run_data.get("status"),
                "runClass": run_data.get("run_class"),
                "experiment": _experiment_entity_id(snapshot, run_data.get("experiment_id") or ""),
                "artifacts": artifacts,
            }
        )
        run_dir = project.runs_dir / run_id
        for name in RUN_FILES:
            path = run_dir / name
            if path.exists():
                graph.append(
                    {
                        "@id": relative_to_root(project, path),
                        "@type": "File",
                        "about": run_entity,
                    }
                )
        for artifact in sorted(
            run_record.get("artifacts") or [],
            key=lambda item: str(item.get("artifact_id") or ""),
        ):
            graph.append(
                {
                    "@id": _artifact_entity_id(run_id, artifact),
                    "@type": "File",
                    "identifier": artifact.get("artifact_id"),
                    "run": run_entity,
                    "contentUrl": artifact.get("project_relative_path")
                    or artifact.get("original_path"),
                    "claimRelevance": artifact.get("claim_relevance"),
                    "reviewStatus": artifact.get("review_status"),
                }
            )
        debt_report = run_record.get("debt_report")
        if debt_report:
            graph.append(
                {
                    "@id": f".mechledger/runs/{run_id}/scientific_debt_report.json",
                    "@type": "CreativeWork",
                    "run": run_entity,
                    "cleanCandidateSupport": debt_report.get("clean_candidate_support"),
                }
            )
    for debt in sorted(snapshot.debts, key=lambda item: str(item.get("debt_id") or "")):
        run_id = str(debt.get("run_id") or "")
        graph.append(
            {
                "@id": f"{debt.get('report_path')}#{debt.get('debt_id')}",
                "@type": "CreativeWork",
                "identifier": debt.get("debt_id"),
                "debtType": debt.get("debt_type"),
                "severity": debt.get("severity"),
                "status": debt.get("status"),
                "run": f".mechledger/runs/{run_id}/" if run_id else None,
                "claim": f"{claim_ledger_id}#{debt.get('claim_id')}"
                if debt.get("claim_id")
                else None,
                "waiverDecision": f"{decision_log_id}#{debt.get('waiver_decision_id')}"
                if debt.get("waiver_decision_id")
                else None,
            }
        )
    for label in sorted(snapshot.external_labels, key=lambda item: str(item.get("label_id"))):
        graph.append(
            {
                "@id": f"research/literature/external_labels.jsonl#{label.get('label_id')}",
                "@type": "CreativeWork",
                "identifier": label.get("label_id"),
                "source": label.get("source"),
                "labelText": label.get("label_text"),
                "featureId": label.get("feature_id"),
                "linkedClaims": [
                    f"{claim_ledger_id}#{claim_id}"
                    for claim_id in sorted(label.get("linked_claims") or [])
                ],
                "evidenceRole": "external-label-metadata",
            }
        )
    for record in sorted(
        snapshot.records,
        key=lambda item: (
            str(item.get("file") or ""),
            str(item.get("record_specific_id") or item.get("record_id") or ""),
        ),
    ):
        file = str(record.get("file") or "")
        specific_id = str(record.get("record_specific_id") or record.get("record_id"))
        graph.append(
            {
                "@id": f"{file}#{specific_id}",
                "@type": "CreativeWork",
                "identifier": specific_id,
                "recordId": record.get("record_id"),
                "recordSpecificId": specific_id,
                "recordType": record.get("record_type"),
                "canonicalRecordType": record.get("canonical_record_type"),
                "schemaStatus": record.get("schema_status"),
                "linkedRuns": [
                    f".mechledger/runs/{run_id}/"
                    for run_id in sorted(record.get("linked_runs") or [])
                ],
                "linkedClaims": [
                    f"{claim_ledger_id}#{claim_id}"
                    for claim_id in sorted(record.get("linked_claims") or [])
                ],
                "linkedDecisions": [
                    f"{decision_log_id}#{decision_id}"
                    for decision_id in sorted(record.get("linked_decisions") or [])
                ],
                "artifactPaths": sorted(record.get("artifact_paths") or []),
                "evidenceRole": "platform-record-metadata",
            }
        )
    graph.sort(key=lambda item: str(item.get("@id") or ""))
    return graph


def write_bundle(
    project: Project,
    out: Path,
    *,
    run_aliases: list[str] | None = None,
    include_local_runs: bool = False,
    include_artifacts: bool = False,
    redact_env: bool = True,
    manifest_only: bool = False,
) -> tuple[Path, dict[str, Any]]:
    run_aliases = run_aliases or []
    snapshot = collect_project(project)
    selected_run_ids = _selected_run_ids(project, snapshot, run_aliases, include_local_runs)
    files = _bundle_files(project, selected_run_ids)
    artifact_metadata, artifact_bytes, omitted = _artifact_bundle_entries(
        project,
        snapshot,
        selected_run_ids,
        include_artifacts=include_artifacts,
    )
    files.extend(artifact_bytes)
    files = sorted({path: source for path, source in files}.items())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_by": "mechledger",
        "project_id": project.config.project_id,
        "included_canonical_files": [
            dest for dest, _ in files if not dest.startswith(".mechledger/runs/")
        ],
        "included_run_ids": selected_run_ids,
        "artifact_metadata": artifact_metadata,
        "platform_records": _platform_record_manifest_entries(snapshot),
        "omitted_artifact_reasons": omitted,
        "redaction_policy": {"redact_env": redact_env},
        "files": [
            {"path": dest, "sha256": sha256_file(source)} for dest, source in files
        ],
        "warnings": sorted(snapshot.warnings),
    }
    if manifest_only:
        write_json(out, manifest)
        return out, manifest
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.name.endswith(".tar.gz"):
        _write_tar_gz(out, files, manifest)
    elif out.name.endswith(".tar.zst"):
        _write_tar_zst(out, files, manifest)
    else:
        raise ValueError("Bundle --out must end in .tar.gz, .tar.zst, or use --manifest-only.")
    return out, manifest


def write_appendix(
    project: Project,
    out: Path,
    *,
    claims: list[str] | None = None,
    runs: list[str] | None = None,
    include_debt: bool = False,
    include_decisions: bool = False,
    include_artifacts: bool = False,
) -> Path:
    snapshot = collect_project(project)
    claim_filter = set(claims or [])
    run_filter = set(runs or [])
    selected_claims = [
        claim
        for claim in sorted(snapshot.claims.values(), key=lambda item: item.claim_id)
        if not claim_filter or claim.claim_id in claim_filter
    ]
    lines = [
        "# MechLedger Appendix",
        "",
        f"Project ID: `{project.config.project_id}`",
        "",
        "MechLedger does not prove scientific truth or verify citations; it reports "
        "registered evidence, claim language policy, and visible scientific debt.",
        "",
        "## Claims",
        "",
        "| Claim | Status | Scope | Linked runs | Linked decisions |",
        "| --- | --- | --- | --- | --- |",
    ]
    for claim in selected_claims:
        linked_runs = [
            run for run in sorted(claim.linked_runs) if not run_filter or run in run_filter
        ]
        linked_decisions = sorted(claim.linked_decisions)
        lines.append(
            "| `{}` | `{}` | {} | {} | {} |".format(
                claim.claim_id,
                claim.status.value,
                claim.scope or "",
                ", ".join(f"`{run_id}`" for run_id in linked_runs) or "none",
                ", ".join(f"`{decision_id}`" for decision_id in linked_decisions) or "none",
            )
        )
        if claim.status.value in {"failed_or_weakened", "contradicted", "retired"}:
            lines.append(
                f"\n`{claim.claim_id}` is `{claim.status.value}` and is not phrased as support."
            )
        lines.append(f"\nAllowed: {', '.join(claim.allowed) or 'none'}")
        lines.append(f"Forbidden: {', '.join(claim.forbidden) or 'none'}")
        lines.append(f"Required caveats: {', '.join(claim.required_caveats) or 'none'}")
        if claim.debt_flags:
            lines.append(f"Debt flags: {', '.join(claim.debt_flags)}")
    lines.extend(["", "## Runs", ""])
    for run_id, run_record in sorted(snapshot.runs.items()):
        if run_filter and run_id not in run_filter:
            continue
        run = run_record["run"]
        lines.append(
            f"- `{run_id}`: status `{run.get('status')}`, class `{run.get('run_class')}`, "
            f"experiment `{run.get('experiment_id')}`"
        )
    if include_decisions:
        lines.extend(["", "## Decisions", ""])
        for decision in sorted(snapshot.decisions.values(), key=lambda item: item.decision_id):
            lines.append(f"- `{decision.decision_id}`: `{decision.status}` {decision.title or ''}")
    if include_debt:
        lines.extend(["", "## Unresolved Scientific Debt", ""])
        debts = [
            debt
            for debt in sorted(snapshot.debts, key=lambda item: str(item.get("debt_id") or ""))
            if debt.get("status") == "open"
            and (not run_filter or str(debt.get("run_id")) in run_filter)
        ]
        if debts:
            for debt in debts:
                lines.append(
                    f"- `{debt.get('debt_id')}` `{debt.get('debt_type')}` "
                    f"severity `{debt.get('severity')}`: {debt.get('message')}"
                )
        else:
            lines.append("- none recorded")
    if include_artifacts:
        lines.extend(["", "## Artifact Manifest Summary", ""])
        for artifact in sorted(
            snapshot.artifact_metadata,
            key=lambda item: (str(item.get("run_id")), str(item.get("artifact_id"))),
        ):
            if run_filter and str(artifact.get("run_id")) not in run_filter:
                continue
            lines.append(
                f"- `{artifact.get('artifact_id')}` on `{artifact.get('run_id')}`: "
                f"{artifact.get('claim_relevance')} / {artifact.get('review_status')}"
            )
    if snapshot.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in sorted(snapshot.warnings))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out


def _selected_run_ids(
    project: Project,
    snapshot: ProjectSnapshot,
    aliases: list[str],
    include_local_runs: bool,
) -> list[str]:
    if aliases:
        return sorted({resolve_run_id(project, alias) for alias in aliases})
    if include_local_runs:
        return sorted(snapshot.runs)
    return []


def _bundle_files(project: Project, run_ids: list[str]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for rel in CANONICAL_CANDIDATES:
        path = project.root / rel
        if path.exists() and path.is_file():
            files.append((rel, path))
    for base in (
        project.root / "research/experiments",
        project.root / "research/records",
        project.root / "research/portfolio",
    ):
        if base.exists():
            for path in sorted(item for item in base.glob("**/*") if item.is_file()):
                files.append((relative_to_root(project, path), path))
    for run_id in run_ids:
        run_dir = project.runs_dir / run_id
        for name in RUN_FILES:
            path = run_dir / name
            if path.exists() and path.is_file():
                files.append((relative_to_root(project, path), path))
    return files


def _artifact_bundle_entries(
    project: Project,
    snapshot: ProjectSnapshot,
    run_ids: list[str],
    *,
    include_artifacts: bool,
) -> tuple[list[dict[str, Any]], list[tuple[str, Path]], list[dict[str, str]]]:
    metadata: list[dict[str, Any]] = []
    bytes_to_include: list[tuple[str, Path]] = []
    omitted: list[dict[str, str]] = []
    selected = set(run_ids)
    for artifact in sorted(
        snapshot.artifact_metadata,
        key=lambda item: (str(item.get("run_id")), str(item.get("artifact_id"))),
    ):
        if selected and str(artifact.get("run_id")) not in selected:
            continue
        metadata.append({key: value for key, value in artifact.items() if key != "resolved_path"})
        rel = artifact.get("project_relative_path") or artifact.get("original_path")
        source = Path(str(artifact.get("resolved_path") or project.root / str(rel)))
        if not include_artifacts:
            omitted.append(
                {
                    "artifact_id": str(artifact.get("artifact_id")),
                    "reason": "artifact bytes omitted unless --include-artifacts is set",
                }
            )
            continue
        if not source.exists() or not source.is_file():
            omitted.append(
                {"artifact_id": str(artifact.get("artifact_id")), "reason": "local path missing"}
            )
            continue
        try:
            dest = source.resolve().relative_to(project.root.resolve()).as_posix()
        except ValueError:
            omitted.append(
                {
                    "artifact_id": str(artifact.get("artifact_id")),
                    "reason": "artifact outside project root omitted",
                }
            )
            continue
        bytes_to_include.append((dest, source))
    return metadata, bytes_to_include, omitted


def _platform_record_manifest_entries(snapshot: ProjectSnapshot) -> list[dict[str, Any]]:
    return [
        {
            "artifact_paths": sorted(str(path) for path in record.get("artifact_paths") or []),
            "canonical_record_type": str(record.get("canonical_record_type") or ""),
            "evidence_role": "platform-record-metadata",
            "file": str(record.get("file") or ""),
            "linked_claims": sorted(str(item) for item in record.get("linked_claims") or []),
            "linked_decisions": sorted(
                str(item) for item in record.get("linked_decisions") or []
            ),
            "linked_runs": sorted(str(item) for item in record.get("linked_runs") or []),
            "record_id": str(record.get("record_id") or ""),
            "record_specific_id": str(
                record.get("record_specific_id") or record.get("record_id") or ""
            ),
            "record_type": str(record.get("record_type") or ""),
            "schema_status": str(record.get("schema_status") or ""),
        }
        for record in sorted(
            snapshot.records,
            key=lambda item: (
                str(item.get("file") or ""),
                str(item.get("record_id") or ""),
            ),
        )
    ]


def _write_tar_gz(out: Path, files: list[tuple[str, Path]], manifest: dict[str, Any]) -> None:
    with out.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                _add_bytes(archive, "manifest.json", _json_bytes(manifest))
                for dest, source in files:
                    _add_bytes(archive, dest, source.read_bytes())


def _write_tar_zst(out: Path, files: list[tuple[str, Path]], manifest: dict[str, Any]) -> None:
    zstd = shutil.which("zstd")
    if not zstd:
        raise ValueError(
            "Writing .tar.zst requires the `zstd` command-line tool; use .tar.gz or "
            "install zstd."
        )
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as handle:
        temp_tar = Path(handle.name)
    try:
        with tarfile.open(temp_tar, mode="w") as archive:
            _add_bytes(archive, "manifest.json", _json_bytes(manifest))
            for dest, source in files:
                _add_bytes(archive, dest, source.read_bytes())
        subprocess.run([zstd, "-q", "-f", str(temp_tar), "-o", str(out)], check=True)
    finally:
        temp_tar.unlink(missing_ok=True)


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _experiment_entity_id(snapshot: ProjectSnapshot, experiment_id: str) -> str | None:
    spec = snapshot.experiments.get(experiment_id)
    if spec is None:
        return None
    path = relative_to_root(snapshot.project, Path(spec.file))
    return f"{path}#{experiment_id}"


def _artifact_entity_id(run_id: str, artifact: dict[str, Any]) -> str:
    return f".mechledger/runs/{run_id}/artifact_manifest.json#{artifact.get('artifact_id')}"
