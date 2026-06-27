import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mwb.claim_grammar import ClaimGrammarService
from mwb.cli import app
from mwb.policy_profiles import PolicyProfileService
from mwb.project import ProjectManager
from mwb.sqlite_index import fetch_payload, rebuild_sqlite_index
from mwb.workflows.cards import card_from_run


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def static_compiler_payload() -> dict:
    return {
        "tensor_space_ref": "space_resid_post",
        "unembedding_space_ref": "space_resid_post",
        "target_token_ids": [10],
        "foil_token_ids": [20],
        "decoder_vector": [1.0, 0.0],
        "unembedding": {"10": [1.0, 0.0], "20": [0.0, 1.0]},
        "dictionary": {
            "feature_id": "unit_sae_123",
            "decoder_vectors": {
                "unit_sae_123": [1.0, 0.0],
                "unit_clean_neighbor": [0.0, 1.0],
            },
        },
        "activation_density": {"target": 0.1, "control": 0.1, "max_ratio": 1.5},
    }


def base_hypothesis(operations: list[dict]) -> dict:
    return {
        "wb_ref": "hyp_policy",
        "title": "policy verification hypothesis",
        "units": ["unit_sae_123"],
        "example_bundle_ref": "bundle_1",
        "control_bundle_ref": "ctrl_1",
        "expected_effect": "target_delta > controls",
        "required_controls": ["negation_removed"],
        "metadata": {"tensor_space_compatible": True, "model_sae_hook_match": True},
        "static_compiler": static_compiler_payload(),
        "verification": {
            "baseline": {
                "target_logit": 2.0,
                "control_logit": 1.0,
                "logits": [2.0, 1.0],
                "activation_norm": 1.0,
            },
            "operations": operations,
            "telemetry_thresholds": {"kl_drift": 0.25, "norm_drift": 0.5},
        },
    }


def operation(operation_name: str) -> dict:
    return {
        "operation": operation_name,
        "unit_ref": "unit_sae_123",
        "patch_mode": operation_name,
        "patch_source": "matched_control",
        "patch_target": "target_prompt",
        "coefficient": 1.0,
        "intervened": {
            "target_logit": 1.2,
            "control_logit": 0.9,
            "logits": [1.2, 0.9],
            "activation_norm": 0.95,
        },
    }


def prediction_lock() -> dict:
    return {
        "wb_ref": "lock_policy",
        "hypothesis_ref": "hyp_policy",
        "hypothesis_spec_hash": "sha256:test",
        "expected_direction": "target_delta_positive",
        "expected_controls": {"negation_removed": "low_delta"},
        "git_state": {"commit": "abc123"},
        "environment": {"python": "3.12"},
    }


def mechanism_card_payload(*, profile: str = "strict") -> dict:
    return {
        "claim_ref": "claim_policy_mechanism",
        "mechanism_card_ref": "mc_policy_mechanism",
        "evidence_tier": "mechanism",
        "status": "mechanism_supported",
        "blockers": [],
        "evidence": [
            "association",
            "static_projection_or_path_algebra",
            "causal_necessity",
            "causal_sufficiency",
            "specificity_controls",
            "telemetry_clean",
            "alternative_explanations_resolved",
        ],
        "policy_profile": profile,
        "scientific_debt": [],
    }


def test_policy_profiles_change_claim_ceiling() -> None:
    service = ClaimGrammarService()

    strict = service.check_claim(
        {
            "claim_ref": "claim_strict_mech",
            "text": "Feature F implements negation handling.",
            "mechanism_card": mechanism_card_payload(profile="strict"),
        }
    )
    exploratory = service.check_claim(
        {
            "claim_ref": "claim_exploratory_mech",
            "text": "Feature F implements negation handling.",
            "mechanism_card": mechanism_card_payload(profile="exploratory"),
        }
    )

    assert strict.status == "blocked"
    assert strict.supported_claim_type == "causal_sufficiency"
    assert "generalization_minimum" in strict.missing_requirements
    assert strict.policy_profile == "strict"
    assert exploratory.status == "allowed"
    assert exploratory.policy_profile == "exploratory"


def test_default_strict_profile_adds_zero_ablation_claim_ceiling(tmp_path: Path) -> None:
    from mwb.causal_verification import CausalVerificationService

    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")

    run = CausalVerificationService(project).verify_payload(
        base_hypothesis([operation("zero_ablate")]),
        prediction_lock=prediction_lock(),
        diagnostic_only=False,
        dry_run=False,
    )

    assert run.evidence_posture == "diagnostic_only"
    assert run.metadata["policy_profile"] == "strict"
    assert run.metadata["claim_ceiling"] == "diagnostic_only"
    assert "zero_ablation_claim_ceiling" in run.metadata["blockers"]


def test_strict_profile_requires_noising_and_denoising_for_candidate_claims(tmp_path: Path) -> None:
    from mwb.causal_verification import CausalVerificationService

    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    service = CausalVerificationService(project)

    missing_denoising = service.verify_payload(
        base_hypothesis([operation("resample_ablate"), operation("noising")]),
        prediction_lock=prediction_lock(),
        diagnostic_only=False,
        dry_run=False,
    )
    complete = service.verify_payload(
        base_hypothesis(
            [operation("resample_ablate"), operation("noising"), operation("denoising")]
        ),
        prediction_lock=prediction_lock(),
        diagnostic_only=False,
        dry_run=False,
    )

    assert "missing_denoising" in missing_denoising.metadata["blockers"]
    assert missing_denoising.status == "insufficient_evidence"
    assert "missing_denoising" not in complete.metadata["blockers"]
    assert complete.status == "candidate_evidence"


def test_policy_applies_to_cards_and_draft_guard(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_ref": "run_policy_card",
                "status": "mechanism_supported",
                "policy_report": {
                    "policy_profile": "strict",
                    "claim_ceiling": "causal_sufficiency",
                    "blockers": ["policy_generalization_required"],
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "control_metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "blocker_report.json").write_text(
        json.dumps(
            {
                "run_ref": "run_policy_card",
                "blockers": ["policy_generalization_required"],
                "primary_blocker": "policy_generalization_required",
            }
        ),
        encoding="utf-8",
    )

    card = card_from_run(run_dir)
    report = ClaimGrammarService().check_claim(
        {
            "claim_ref": card.metadata["claim_ref"],
            "text": "Feature F implements negation handling.",
            "mechanism_card": {
                **card.model_dump(mode="json"),
                "claim_ref": card.metadata["claim_ref"],
                "blockers": card.metadata["blockers"],
                "policy_profile": card.metadata["policy_profile"],
            },
        }
    )

    assert card.evidence_tier == "causal_sufficiency"
    assert "implements" in card.blocked_language
    assert report.status == "blocked"
    assert "policy_generalization_required" in report.blockers


def test_policy_profile_cli_writes_report_and_restores_sqlite(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    runner = CliRunner()

    result = runner.invoke(app, ["policy", "check", "--profile", "strict"])

    assert result.exit_code == 0, result.output
    output = json.loads(result.output)
    assert output["wb_type"] == "PolicyEvaluationReport"
    assert output["policy_profile"] == "strict"
    assert output["status"] == "pass"
    assert PolicyProfileService(project).load_project_profile().name == "strict"

    restored = rebuild_sqlite_index(project, output_path=tmp_path / "rebuilt.sqlite")

    assert restored["counts"]["policy_evaluations"] == 1
    indexed = fetch_payload(tmp_path / "rebuilt.sqlite", "policy_evaluations", output["wb_ref"])
    assert indexed["policy_profile"] == "strict"
