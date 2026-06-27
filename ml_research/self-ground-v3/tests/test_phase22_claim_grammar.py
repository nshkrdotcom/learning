import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mwb.claim_grammar import ClaimGrammarService
from mwb.cli import app
from mwb.project import ProjectManager
from mwb.sqlite_index import fetch_payload, rebuild_sqlite_index
from mwb.workflows.draft_guard import check_draft_text


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def card_payload(evidence_tier: str, *, blockers: list[str] | None = None) -> dict:
    return {
        "claim_ref": f"claim_{evidence_tier}",
        "mechanism_card_ref": f"mc_{evidence_tier}",
        "evidence_tier": evidence_tier,
        "status": evidence_tier,
        "blockers": blockers or [],
        "allowed_language": ["is associated with"],
        "blocked_language": [],
        "scientific_debt": [
            {
                "debt_ref": "debt_missing_heldout",
                "kind": "missing_heldout_generalization",
                "status": "unresolved",
                "required_to_upgrade_to": "generalization",
            }
        ]
        if blockers and "insufficient_heldout_generalization" in blockers
        else [],
    }


def test_observation_claim_allows_association_evidence() -> None:
    report = ClaimGrammarService().check_claim(
        {
            "claim_ref": "claim_assoc",
            "text": "Feature F is associated with negation examples.",
            "mechanism_card": card_payload("association"),
        }
    )

    assert report.status == "allowed"
    assert report.claim_type == "association"
    assert report.missing_requirements == []
    assert "associated with" in report.allowed_verbs


def test_static_projection_claim_requires_projection_evidence() -> None:
    service = ClaimGrammarService()

    blocked = service.check_claim(
        {
            "claim_ref": "claim_projection",
            "text": "Feature F projects toward the negation token contrast.",
            "mechanism_card": card_payload("association"),
        }
    )
    allowed = service.check_claim(
        {
            "claim_ref": "claim_projection",
            "text": "Feature F projects toward the negation token contrast.",
            "mechanism_card": card_payload("projection"),
        }
    )

    assert blocked.status == "blocked"
    assert "static_projection_or_path_algebra" in blocked.missing_requirements
    assert allowed.status == "allowed"
    assert allowed.claim_type == "projection"


def test_stronger_claims_require_causal_mediation_generalization_and_mechanism_evidence() -> None:
    card = card_payload("causal_necessity", blockers=["insufficient_heldout_generalization"])
    service = ClaimGrammarService()

    mechanism = service.check_claim(
        {
            "claim_ref": "claim_mech",
            "text": "Feature F implements negation handling.",
            "mechanism_card": card,
        }
    )
    scoped_necessity = service.check_claim(
        {
            "claim_ref": "claim_need",
            "text": "Feature F contributes causally to the tested negation contrast.",
            "mechanism_card": card,
        }
    )
    generalization = service.check_claim(
        {
            "claim_ref": "claim_generalizes",
            "text": "Feature F generalizes across held-out vocabularies and dictionaries.",
            "mechanism_card": card,
        }
    )

    assert mechanism.status == "blocked"
    assert "causal_sufficiency" in mechanism.missing_requirements
    assert "generalization_minimum" in mechanism.missing_requirements
    assert "implements" in mechanism.blocked_verbs
    assert scoped_necessity.status == "caveated"
    assert scoped_necessity.required_caveats == [
        "Do not generalize beyond the tested model, layer, dictionary, and prompt bundle."
    ]
    assert generalization.status == "blocked"
    assert generalization.blocking_debt[0]["kind"] == "missing_heldout_generalization"


def test_inline_override_is_visible_but_does_not_upgrade_blocked_claim() -> None:
    report = ClaimGrammarService().check_claim(
        {
            "claim_ref": "claim_override",
            "text": "Feature F implements negation handling.",
            "mechanism_card": card_payload("association", blockers=["control_leaky"]),
            "override": {
                "reason": "author wants stronger language",
                "requested_by": "tester",
                "visibility": "inline",
            },
        }
    )

    assert report.status == "blocked"
    assert report.override["visible"] is True
    assert report.override["applied"] is False
    assert "control_leaky" in report.blockers


def test_claim_check_cli_writes_report_and_restores_sqlite(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    fixture = tmp_path / "claim.json"
    fixture.write_text(
        json.dumps(
            {
                "claim_ref": "claim_cli_assoc",
                "text": "Feature F is associated with negation examples.",
                "mechanism_card": card_payload("association"),
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["claim", "check", str(fixture)])

    assert result.exit_code == 0, result.output
    output = json.loads(result.output)
    assert output["wb_type"] == "ClaimGrammarReport"
    assert output["status"] == "allowed"
    assert (project.mechanism_dir / "claims" / "claim_cli_assoc_grammar_report.json").exists()
    restored = rebuild_sqlite_index(project, output_path=tmp_path / "rebuilt.sqlite")
    assert restored["counts"]["claim_grammar_reports"] == 1
    indexed = fetch_payload(tmp_path / "rebuilt.sqlite", "claim_grammar_reports", output["wb_ref"])
    assert indexed["claim_type"] == "association"


def test_draft_check_uses_claim_grammar_before_phrase_fallback() -> None:
    card = card_payload("causal_necessity", blockers=["insufficient_heldout_generalization"])
    card["claim_ref"] = "claim_need"

    report = check_draft_text(
        "Feature F generalizes across held-out vocabularies. [CLAIM:claim_need]",
        {"claim_need": card},
    )

    assert report["status"] == "blocked"
    assert report["findings"][0]["claim_type"] == "generalization"
    assert "generalization_minimum" in report["findings"][0]["missing_requirements"]
