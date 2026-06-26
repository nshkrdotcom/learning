from __future__ import annotations

import getpass
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mechledger.inspection import relative_to_root, sha256_file, write_json
from mechledger.project import Project, now_utc


class CopilotOutputType(StrEnum):
    EXPERIMENT_SPEC_DRAFT = "experiment_spec_draft"
    DECISION_RECORD_DRAFT = "decision_record_draft"
    RESEARCH_LOG_DRAFT = "research_log_draft"
    CLAIM_UPDATE_PROPOSAL = "claim_update_proposal"
    DRAFT_VIOLATION_REPORT = "draft_violation_report"
    SCIENTIFIC_DEBT_SUMMARY = "scientific_debt_summary"
    NEXT_ACTION_PLAN = "next_action_plan"
    ARTIFACT_SUMMARY = "artifact_summary"
    PRIOR_ART_SUMMARY = "prior_art_summary"
    PAPER_WORDING_SUGGESTION = "paper_wording_suggestion"
    OTHER = "other"


class CopilotReviewOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    MODIFIED = "modified"
    PENDING = "pending"


class CopilotOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_id: str
    session_id: str
    output_type: CopilotOutputType
    generated_artifact_path: str
    source_artifact_paths: list[str] = Field(default_factory=list)
    prompt_artifact_path: str
    model: str
    human_reviewed: bool = False
    reviewed_by: str | None = None
    review_outcome: CopilotReviewOutcome | None = None
    accepted_artifact_path: str | None = None
    accepted_provenance_path: str | None = None


class CopilotSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    started_at: str
    ended_at: str | None = None
    purpose: str
    model: str
    source_artifacts: list[str] = Field(default_factory=list)
    outputs: list[CopilotOutput] = Field(default_factory=list)

    @model_validator(mode="after")
    def output_session_ids_match(self) -> CopilotSession:
        for output in self.outputs:
            if output.session_id != self.session_id:
                raise ValueError(
                    f"output {output.output_id} session_id {output.session_id} does not "
                    f"match metadata session_id {self.session_id}"
                )
        return self


class CopilotProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    copilot_session_id: str
    copilot_output_id: str
    source_prompt_hash: str
    generated_artifact_hash: str
    accepted_artifact_hash: str
    accepted_at: str
    review_outcome: CopilotReviewOutcome
    reviewed_by: str | None
    model: str
    output_type: CopilotOutputType
    source_artifact_paths: list[str]
    prompt_artifact_path: str
    generated_artifact_path: str
    accepted_artifact_path: str
    accepted_provenance_path: str


class DiscoveredCopilotOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata_path: Path
    session: CopilotSession
    output: CopilotOutput
    generated_artifact_exists: bool
    prompt_artifact_exists: bool

    def list_record(self, project: Project) -> dict[str, Any]:
        payload = self.output.model_dump(mode="json")
        payload.update(
            {
                "metadata_path": relative_to_root(project, self.metadata_path),
                "generated_artifact_exists": self.generated_artifact_exists,
                "prompt_artifact_exists": self.prompt_artifact_exists,
            }
        )
        return payload


def list_copilot_outputs(project: Project) -> list[DiscoveredCopilotOutput]:
    outputs: list[DiscoveredCopilotOutput] = []
    root = _copilot_root(project)
    if not root.exists():
        return outputs
    for metadata_path in sorted(root.glob("*/metadata.json")):
        session = load_copilot_metadata(metadata_path)
        for output in session.outputs:
            outputs.append(
                DiscoveredCopilotOutput(
                    metadata_path=metadata_path,
                    session=session,
                    output=output,
                    generated_artifact_exists=_project_path(
                        project, output.generated_artifact_path
                    ).is_file(),
                    prompt_artifact_exists=_project_path(
                        project, output.prompt_artifact_path
                    ).is_file(),
                )
            )
    _reject_duplicate_output_ids(outputs)
    outputs.sort(key=lambda item: (item.output.output_id, item.session.session_id))
    return outputs


def load_copilot_metadata(path: Path) -> CopilotSession:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: malformed copilot metadata JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: copilot metadata must be a JSON object.")
    try:
        return CopilotSession.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"{path}: invalid copilot metadata: {exc}") from exc


def find_copilot_output(project: Project, output_id: str) -> DiscoveredCopilotOutput:
    for discovered in list_copilot_outputs(project):
        if discovered.output.output_id == output_id:
            return discovered
    raise FileNotFoundError(f"Unknown copilot output: {output_id}")


def reject_copilot_output(project: Project, output_id: str) -> CopilotOutput:
    discovered = find_copilot_output(project, output_id)
    updated = discovered.output.model_copy(
        update={
            "human_reviewed": True,
            "reviewed_by": _reviewed_by(),
            "review_outcome": CopilotReviewOutcome.REJECTED,
            "accepted_artifact_path": None,
            "accepted_provenance_path": None,
        }
    )
    _replace_output(discovered, updated)
    return updated


def accept_copilot_output(
    project: Project,
    output_id: str,
    *,
    destination: Path,
    modified_path: Path | None = None,
) -> tuple[CopilotOutput, CopilotProvenance]:
    discovered = find_copilot_output(project, output_id)
    output = discovered.output
    dest = _resolve_destination(project, destination)
    generated = _project_path(project, output.generated_artifact_path)
    prompt = _project_path(project, output.prompt_artifact_path)
    if not generated.is_file():
        raise FileNotFoundError(
            f"{output.generated_artifact_path}: generated artifact does not exist."
        )
    if not prompt.is_file():
        raise FileNotFoundError(f"{output.prompt_artifact_path}: prompt artifact does not exist.")

    if modified_path is None:
        source = generated
        outcome = CopilotReviewOutcome.ACCEPTED
    else:
        source = _resolve_existing_project_file(project, modified_path, label="modified path")
        outcome = CopilotReviewOutcome.MODIFIED

    accepted_text = source.read_text(encoding="utf-8")
    accepted_text = insert_copilot_session_id_if_canonical(
        project,
        dest,
        accepted_text,
        output.session_id,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(accepted_text, encoding="utf-8")

    provenance_path = Path(f"{dest}.mechledger-provenance.json")
    provenance_rel = relative_to_root(project, provenance_path)
    provenance = CopilotProvenance(
        copilot_session_id=output.session_id,
        copilot_output_id=output.output_id,
        source_prompt_hash=_prefixed_sha256(prompt),
        generated_artifact_hash=_prefixed_sha256(generated),
        accepted_artifact_hash=_prefixed_sha256(dest),
        accepted_at=now_utc(),
        review_outcome=outcome,
        reviewed_by=_reviewed_by(),
        model=output.model,
        output_type=output.output_type,
        source_artifact_paths=output.source_artifact_paths,
        prompt_artifact_path=output.prompt_artifact_path,
        generated_artifact_path=output.generated_artifact_path,
        accepted_artifact_path=relative_to_root(project, dest),
        accepted_provenance_path=provenance_rel,
    )
    write_json(provenance_path, provenance.model_dump(mode="json"))

    updated = output.model_copy(
        update={
            "human_reviewed": True,
            "reviewed_by": provenance.reviewed_by,
            "review_outcome": outcome,
            "accepted_artifact_path": provenance.accepted_artifact_path,
            "accepted_provenance_path": provenance.accepted_provenance_path,
        }
    )
    _replace_output(discovered, updated)
    return updated, provenance


def insert_copilot_session_id_if_canonical(
    project: Project,
    destination: Path,
    content: str,
    session_id: str,
) -> str:
    rel = relative_to_root(project, destination)
    canonical = {
        project.config.default_claim_ledger,
        project.config.default_decision_log,
        project.config.default_research_log,
    }
    if rel not in canonical:
        return content
    return _insert_copilot_session_id(content, session_id)


def _insert_copilot_session_id(content: str, session_id: str) -> str:
    pattern = re.compile(r"(^```yaml[^\n]*\n)(.*?)(^```[ \t]*$)", re.MULTILINE | re.DOTALL)
    pieces: list[str] = []
    cursor = 0
    for match in pattern.finditer(content):
        pieces.append(content[cursor : match.start()])
        opener, body, closer = match.group(1), match.group(2), match.group(3)
        pieces.append(opener)
        pieces.append(_update_yaml_body(body, session_id))
        pieces.append(closer)
        cursor = match.end()
    pieces.append(content[cursor:])
    return "".join(pieces)


def _update_yaml_body(body: str, session_id: str) -> str:
    id_match = re.search(r"^(\s*)(claim_id|decision_id|entry_id)\s*:\s*.+$", body, re.MULTILINE)
    if id_match is None:
        return body
    session_match = re.search(r"^(\s*)copilot_session_id\s*:\s*(.*?)\s*$", body, re.MULTILINE)
    if session_match:
        raw_value = session_match.group(2).strip()
        normalized = raw_value.strip("\"'")
        if normalized in {"", "null", "Null", "NULL", "~"}:
            replacement = f"{session_match.group(1)}copilot_session_id: {session_id}"
            return body[: session_match.start()] + replacement + body[session_match.end() :]
        if normalized == session_id:
            return body
        raise ValueError(
            f"YAML block already has a different copilot_session_id: {raw_value}"
        )
    line_end = body.find("\n", id_match.end())
    insert = f"\n{id_match.group(1)}copilot_session_id: {session_id}"
    if line_end == -1:
        return body + insert
    return body[:line_end] + insert + body[line_end:]


def _replace_output(discovered: DiscoveredCopilotOutput, updated: CopilotOutput) -> None:
    session = discovered.session
    outputs = [
        updated if output.output_id == updated.output_id else output for output in session.outputs
    ]
    payload = session.model_copy(update={"outputs": outputs}).model_dump(mode="json")
    write_json(discovered.metadata_path, payload)


def _reject_duplicate_output_ids(outputs: list[DiscoveredCopilotOutput]) -> None:
    seen: dict[str, DiscoveredCopilotOutput] = {}
    for item in outputs:
        output_id = item.output.output_id
        if output_id in seen:
            first = seen[output_id]
            raise ValueError(
                f"Duplicate copilot output_id {output_id}: "
                f"{first.metadata_path} and {item.metadata_path}"
            )
        seen[output_id] = item


def _resolve_destination(project: Project, path: Path) -> Path:
    resolved = _project_path(project, path).resolve()
    try:
        rel = resolved.relative_to(project.root.resolve())
    except ValueError as exc:
        raise ValueError(f"Destination must be inside the project root: {path}") from exc
    if not rel.parts or rel.parts[0] != "research":
        raise ValueError(f"Destination must be inside research/: {path}")
    if ".mechledger" in rel.parts:
        raise ValueError(f"Destination must not be inside .mechledger/: {path}")
    return resolved


def _resolve_existing_project_file(project: Project, path: Path, *, label: str) -> Path:
    resolved = _project_path(project, path).resolve()
    try:
        resolved.relative_to(project.root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the project root: {path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return resolved


def _project_path(project: Project, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project.root / path


def _copilot_root(project: Project) -> Path:
    return project.mechledger_dir / "copilot"


def _prefixed_sha256(path: Path) -> str:
    return f"sha256:{sha256_file(path)}"


def _reviewed_by() -> str | None:
    try:
        username = getpass.getuser()
    except Exception:  # noqa: BLE001
        return None
    return username or None
