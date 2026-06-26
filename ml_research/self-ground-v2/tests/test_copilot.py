from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from helpers_project import init_project, runner

from mechledger.cli import app

SESSION_ID = "COP20260625T120000Z_abc123"


def _write_copilot_output(
    tmp_path: Path,
    *,
    session_id: str = SESSION_ID,
    output_id: str = "OUT001",
    output_type: str = "claim_update_proposal",
    output_text: str = "assistant draft\n",
    prompt_text: str = "prompt text\n",
    generated_artifact_path: str | None = None,
    prompt_artifact_path: str | None = None,
    metadata_overrides: dict[str, Any] | None = None,
    output_overrides: dict[str, Any] | None = None,
) -> Path:
    session_dir = tmp_path / ".mechledger/copilot" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    output_rel = generated_artifact_path or f".mechledger/copilot/{session_id}/output.md"
    prompt_rel = prompt_artifact_path or f".mechledger/copilot/{session_id}/prompt.md"
    output_path = tmp_path / output_rel
    prompt_path = tmp_path / prompt_rel
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")
    prompt_path.write_text(prompt_text, encoding="utf-8")
    output: dict[str, Any] = {
        "output_id": output_id,
        "session_id": session_id,
        "output_type": output_type,
        "generated_artifact_path": output_rel,
        "source_artifact_paths": ["research/logs/claim_ledger.md"],
        "prompt_artifact_path": prompt_rel,
        "model": "external-assistant",
        "human_reviewed": False,
        "reviewed_by": None,
        "review_outcome": None,
        "accepted_artifact_path": None,
        "accepted_provenance_path": None,
    }
    output.update(output_overrides or {})
    metadata: dict[str, Any] = {
        "session_id": session_id,
        "started_at": "2026-06-25T12:00:00Z",
        "ended_at": None,
        "purpose": "Draft claim update proposal",
        "model": "external-assistant",
        "source_artifacts": ["research/logs/claim_ledger.md"],
        "outputs": [output],
    }
    metadata.update(metadata_overrides or {})
    (session_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return session_dir / "metadata.json"


def _metadata(tmp_path: Path, session_id: str = SESSION_ID) -> dict[str, Any]:
    return json.loads(
        (tmp_path / ".mechledger/copilot" / session_id / "metadata.json").read_text(
            encoding="utf-8"
        )
    )


def _reviewed_by_is_valid(value: Any) -> bool:
    return value is None or isinstance(value, str)


def test_copilot_list_empty_when_directory_missing(tmp_path: Path) -> None:
    init_project(tmp_path)

    result = runner.invoke(
        app, ["copilot", "list"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    machine = runner.invoke(
        app,
        ["copilot", "list", "--json"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert "No copilot outputs found." in result.output
    assert machine.exit_code == 0, machine.output
    assert json.loads(machine.output) == []


def test_copilot_list_discovers_outputs_and_json_records(tmp_path: Path) -> None:
    init_project(tmp_path)
    _write_copilot_output(tmp_path)

    text = runner.invoke(
        app, ["copilot", "list"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    machine = runner.invoke(
        app,
        ["copilot", "list", "--json"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert text.exit_code == 0, text.output
    assert "OUT001" in text.output
    assert "generated_exists=True" in text.output
    records = json.loads(machine.output)
    assert records[0]["output_id"] == "OUT001"
    assert records[0]["session_id"] == SESSION_ID
    assert records[0]["generated_artifact_exists"] is True
    assert records[0]["prompt_artifact_exists"] is True


def test_copilot_duplicate_output_id_fails(tmp_path: Path) -> None:
    init_project(tmp_path)
    _write_copilot_output(tmp_path, session_id=SESSION_ID, output_id="OUT001")
    _write_copilot_output(
        tmp_path,
        session_id="COP20260625T120500Z_def456",
        output_id="OUT001",
    )

    result = runner.invoke(
        app, ["copilot", "list"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )

    assert result.exit_code == 2
    assert "Duplicate copilot output_id OUT001" in result.output


def test_copilot_show_text_and_json(tmp_path: Path) -> None:
    init_project(tmp_path)
    _write_copilot_output(tmp_path)

    text = runner.invoke(
        app,
        ["copilot", "show", "OUT001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    machine = runner.invoke(
        app,
        ["copilot", "show", "OUT001", "--json"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert text.exit_code == 0, text.output
    assert f"session_id: {SESSION_ID}" in text.output
    assert "generated_artifact_exists: True" in text.output
    payload = json.loads(machine.output)
    assert payload == _metadata(tmp_path)["outputs"][0]


def test_copilot_show_reports_missing_artifact_status(tmp_path: Path) -> None:
    init_project(tmp_path)
    _write_copilot_output(tmp_path)
    (tmp_path / f".mechledger/copilot/{SESSION_ID}/prompt.md").unlink()

    listed = runner.invoke(
        app, ["copilot", "list"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    shown = runner.invoke(
        app,
        ["copilot", "show", "OUT001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert listed.exit_code == 0, listed.output
    assert "prompt_exists=False" in listed.output
    assert shown.exit_code == 0, shown.output
    assert "prompt_artifact_exists: False" in shown.output


def test_copilot_review_unknown_output_fails(tmp_path: Path) -> None:
    init_project(tmp_path)

    result = runner.invoke(
        app,
        ["copilot", "review", "OUT_MISSING", "--reject"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 2
    assert "Unknown copilot output: OUT_MISSING" in result.output


def test_copilot_reject_updates_metadata_without_canonical_artifact(tmp_path: Path) -> None:
    init_project(tmp_path)
    _write_copilot_output(tmp_path)
    destination = tmp_path / "research/logs/copied.md"

    result = runner.invoke(
        app,
        ["copilot", "review", "OUT001", "--reject"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    output = _metadata(tmp_path)["outputs"][0]
    assert output["human_reviewed"] is True
    assert output["review_outcome"] == "rejected"
    assert _reviewed_by_is_valid(output["reviewed_by"])
    assert output["accepted_artifact_path"] is None
    assert output["accepted_provenance_path"] is None
    assert not destination.exists()
    assert not (tmp_path / "research/logs/copied.md.mechledger-provenance.json").exists()


def test_copilot_accept_requires_and_validates_destination(tmp_path: Path) -> None:
    init_project(tmp_path)
    _write_copilot_output(tmp_path)

    missing_to = runner.invoke(
        app,
        ["copilot", "review", "OUT001", "--accept"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    outside_project = runner.invoke(
        app,
        [
            "copilot",
            "review",
            "OUT001",
            "--accept",
            "--to",
            str(tmp_path.parent / "outside.md"),
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    outside_research = runner.invoke(
        app,
        ["copilot", "review", "OUT001", "--accept", "--to", "notes/out.md"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    under_mechledger = runner.invoke(
        app,
        [
            "copilot",
            "review",
            "OUT001",
            "--accept",
            "--to",
            ".mechledger/copilot/accepted.md",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert missing_to.exit_code == 2
    assert "--to" in missing_to.output
    assert outside_project.exit_code == 2
    assert "inside the project root" in outside_project.output
    assert outside_research.exit_code == 2
    assert "inside research/" in outside_research.output
    assert under_mechledger.exit_code == 2
    assert "inside research/" in under_mechledger.output or ".mechledger" in under_mechledger.output


def test_copilot_accept_copies_output_and_writes_provenance_and_metadata(
    tmp_path: Path,
) -> None:
    init_project(tmp_path)
    _write_copilot_output(tmp_path, output_text="accepted draft\n", prompt_text="source prompt\n")

    result = runner.invoke(
        app,
        [
            "copilot",
            "review",
            "OUT001",
            "--accept",
            "--to",
            "research/logs/accepted.md",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    accepted = tmp_path / "research/logs/accepted.md"
    sidecar = tmp_path / "research/logs/accepted.md.mechledger-provenance.json"
    assert accepted.read_text(encoding="utf-8") == "accepted draft\n"
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    assert provenance["copilot_session_id"] == SESSION_ID
    assert provenance["copilot_output_id"] == "OUT001"
    assert provenance["accepted_artifact_path"] == "research/logs/accepted.md"
    assert provenance["accepted_provenance_path"] == (
        "research/logs/accepted.md.mechledger-provenance.json"
    )
    assert provenance["source_prompt_hash"].startswith("sha256:")
    assert provenance["generated_artifact_hash"].startswith("sha256:")
    assert provenance["accepted_artifact_hash"].startswith("sha256:")
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", provenance["accepted_artifact_hash"])

    output = _metadata(tmp_path)["outputs"][0]
    assert output["human_reviewed"] is True
    assert output["review_outcome"] == "accepted"
    assert output["accepted_artifact_path"] == "research/logs/accepted.md"
    assert output["accepted_provenance_path"] == (
        "research/logs/accepted.md.mechledger-provenance.json"
    )


def test_copilot_modified_review_copies_modified_file_and_records_both_hashes(
    tmp_path: Path,
) -> None:
    init_project(tmp_path)
    _write_copilot_output(tmp_path, output_text="generated draft\n")
    modified = tmp_path / "research/session_drafts/edited_decision.md"
    modified.parent.mkdir(parents=True, exist_ok=True)
    modified.write_text("human modified draft\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "copilot",
            "review",
            "OUT001",
            "--modified",
            str(modified),
            "--to",
            "research/logs/decision_log.md",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "research/logs/decision_log.md").read_text(
        encoding="utf-8"
    ) == "human modified draft\n"
    provenance = json.loads(
        (
            tmp_path / "research/logs/decision_log.md.mechledger-provenance.json"
        ).read_text(encoding="utf-8")
    )
    assert provenance["review_outcome"] == "modified"
    assert provenance["generated_artifact_hash"] != provenance["accepted_artifact_hash"]
    assert _metadata(tmp_path)["outputs"][0]["review_outcome"] == "modified"


def test_copilot_yaml_insertion_for_canonical_ledgers(tmp_path: Path) -> None:
    init_project(tmp_path)
    cases = [
        ("claim_id: C001\nstatus: candidate_claim\n", "research/logs/claim_ledger.md"),
        ("decision_id: D001\nstatus: accepted\n", "research/logs/decision_log.md"),
        ("entry_id: R001\nlinked_claims: [C001]\n", "research/logs/research_log.md"),
    ]
    for index, (yaml_body, destination) in enumerate(cases, start=1):
        output_id = f"OUT{index:03d}"
        _write_copilot_output(
            tmp_path,
            session_id=f"{SESSION_ID}_{index}",
            output_id=output_id,
            output_text=f"# Ledger\n\n```yaml\n{yaml_body}```\n",
        )
        result = runner.invoke(
            app,
            ["copilot", "review", output_id, "--accept", "--to", destination],
            catch_exceptions=False,
            env={"PWD": str(tmp_path)},
        )
        assert result.exit_code == 0, result.output
        content = (tmp_path / destination).read_text(encoding="utf-8")
        assert f"copilot_session_id: {SESSION_ID}_{index}" in content


def test_copilot_yaml_insertion_null_idempotent_conflict_and_arbitrary_fences(
    tmp_path: Path,
) -> None:
    init_project(tmp_path)
    _write_copilot_output(
        tmp_path,
        output_text=(
            "# Claim Ledger\n\n"
            "```yaml\nclaim_id: C001\ncopilot_session_id: null\n```\n\n"
            "```yaml\nnot_a_claim: true\n```\n"
        ),
    )
    destination = "research/logs/claim_ledger.md"

    first = runner.invoke(
        app,
        ["copilot", "review", "OUT001", "--accept", "--to", destination],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    second = runner.invoke(
        app,
        ["copilot", "review", "OUT001", "--accept", "--to", destination],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    content = (tmp_path / destination).read_text(encoding="utf-8")
    assert content.count(f"copilot_session_id: {SESSION_ID}") == 1
    assert "not_a_claim: true\n```" in content

    _write_copilot_output(
        tmp_path,
        session_id="COP20260625T130000Z_conflict",
        output_id="OUT_CONFLICT",
        output_text="```yaml\nclaim_id: C002\ncopilot_session_id: OTHER_SESSION\n```\n",
    )
    conflict = runner.invoke(
        app,
        ["copilot", "review", "OUT_CONFLICT", "--accept", "--to", destination],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert conflict.exit_code == 2
    assert "different copilot_session_id" in conflict.output


def test_copilot_malformed_metadata_fails_clearly(tmp_path: Path) -> None:
    init_project(tmp_path)
    session_dir = tmp_path / ".mechledger/copilot/BAD"
    session_dir.mkdir(parents=True)
    (session_dir / "metadata.json").write_text('{"outputs": "not a list"}\n', encoding="utf-8")

    result = runner.invoke(
        app, ["copilot", "list"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )

    assert result.exit_code == 2
    assert "metadata.json" in result.output
    assert "outputs" in result.output


def test_copilot_review_missing_generated_artifact_fails(tmp_path: Path) -> None:
    init_project(tmp_path)
    _write_copilot_output(tmp_path)
    (tmp_path / f".mechledger/copilot/{SESSION_ID}/output.md").unlink()

    result = runner.invoke(
        app,
        ["copilot", "review", "OUT001", "--accept", "--to", "research/logs/accepted.md"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 2
    assert "generated artifact does not exist" in result.output


def test_copilot_prompt_hash_uses_prompt_artifact(tmp_path: Path) -> None:
    init_project(tmp_path)
    _write_copilot_output(tmp_path, prompt_text="first prompt\n")
    first = runner.invoke(
        app,
        ["copilot", "review", "OUT001", "--accept", "--to", "research/logs/first.md"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert first.exit_code == 0, first.output
    first_hash = json.loads(
        (tmp_path / "research/logs/first.md.mechledger-provenance.json").read_text(
            encoding="utf-8"
        )
    )["source_prompt_hash"]

    _write_copilot_output(
        tmp_path,
        session_id="COP20260625T130000Z_prompt",
        output_id="OUT002",
        prompt_text="second prompt\n",
    )
    second = runner.invoke(
        app,
        ["copilot", "review", "OUT002", "--accept", "--to", "research/logs/second.md"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert second.exit_code == 0, second.output
    second_hash = json.loads(
        (tmp_path / "research/logs/second.md.mechledger-provenance.json").read_text(
            encoding="utf-8"
        )
    )["source_prompt_hash"]

    assert first_hash != second_hash


def test_copilot_commands_are_wired_into_help(tmp_path: Path) -> None:
    root = runner.invoke(app, ["--help"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    copilot = runner.invoke(
        app, ["copilot", "--help"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    review = runner.invoke(
        app,
        ["copilot", "review", "--help"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert root.exit_code == 0
    assert "copilot" in root.output
    assert copilot.exit_code == 0
    assert "list" in copilot.output and "show" in copilot.output and "review" in copilot.output
    assert review.exit_code == 0
    assert "--accept" in review.output and "--reject" in review.output
