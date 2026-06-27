import json
import math
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mwb.cli import app
from mwb.domain.objects import PredictionLock
from mwb.project import ProjectManager
from mwb.sqlite_index import rebuild_sqlite_index
from mwb.workflows.verify import run_verify


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def base_hypothesis(static_compiler: dict) -> dict:
    return {
        "wb_ref": "hyp_static",
        "title": "static compiler hypothesis",
        "units": ["unit_sae_123"],
        "example_bundle_ref": "bundle_1",
        "control_bundle_ref": "ctrl_1",
        "expected_effect": "target_delta > controls",
        "required_controls": ["negation_removed"],
        "metadata": {
            "tensor_space_compatible": True,
            "model_sae_hook_match": True,
            "decoder_unembed_projection_score": -1.0,
        },
        "static_compiler": static_compiler,
    }


def static_payload(
    *,
    decoder_vector: list[float] | None = None,
    dictionary_vectors: dict[str, list[float]] | None = None,
    activation_density: dict[str, float] | None = None,
) -> dict:
    return {
        "tensor_space_ref": "space_resid_post",
        "unembedding_space_ref": "space_resid_post",
        "target_token_ids": [10],
        "foil_token_ids": [20],
        "decoder_vector": decoder_vector or [1.0, 0.0, 0.0],
        "unembedding": {
            "10": [1.0, 0.0, 0.0],
            "20": [0.0, 1.0, 0.0],
        },
        "dictionary": {
            "feature_id": "unit_sae_123",
            "decoder_vectors": dictionary_vectors
            or {
                "unit_sae_123": [1.0, 0.0, 0.0],
                "unit_clean_neighbor": [0.0, 0.0, 1.0],
            },
        },
        "activation_density": activation_density
        or {"target": 0.1, "control": 0.1, "max_ratio": 1.5},
    }


def test_decoder_unembed_projection_uses_l2_cosine_not_metadata() -> None:
    from mwb.static_compiler import StaticCompiler

    report = StaticCompiler().compile_payload(base_hypothesis(static_payload()))
    projection = next(
        check for check in report.checks if check["name"] == "decoder_unembed_projection"
    )

    assert report.status == "pass"
    assert projection["status"] == "pass"
    assert math.isclose(projection["score"], 1 / math.sqrt(2), rel_tol=1e-9)
    assert projection["normalization"] == "l2_cosine"


def test_dictionary_neighbor_interference_fails_on_high_cosine() -> None:
    from mwb.static_compiler import StaticCompiler

    report = StaticCompiler().compile_payload(
        base_hypothesis(
            static_payload(
                dictionary_vectors={
                    "unit_sae_123": [1.0, 0.0, 0.0],
                    "unit_split_neighbor": [0.99, 0.01, 0.0],
                }
            )
        )
    )
    neighbor = next(check for check in report.checks if check["name"] == "neighbor_interference")

    assert report.status == "fail"
    assert report.plausibility_gate == "FAIL"
    assert "dictionary_interference" in report.blockers
    assert neighbor["nearest_neighbor_ref"] == "unit_split_neighbor"
    assert neighbor["nearest_neighbor_cosine"] >= 0.8


def test_activation_density_warning_makes_gate_weak() -> None:
    from mwb.static_compiler import StaticCompiler

    report = StaticCompiler().compile_payload(
        base_hypothesis(
            static_payload(activation_density={"target": 0.3, "control": 0.1, "max_ratio": 1.5})
        )
    )
    density = next(check for check in report.checks if check["name"] == "activation_density")

    assert report.status == "warn"
    assert report.plausibility_gate == "WEAK"
    assert density["status"] == "warn"
    assert "activation_density_mismatch" in report.warnings


def test_plausibility_gate_uses_weakest_link_aggregation() -> None:
    from mwb.static_compiler import StaticCompiler

    report = StaticCompiler().compile_payload(
        base_hypothesis(static_payload(decoder_vector=[0.0, 1.0, 0.0]))
    )

    assert report.status == "fail"
    assert report.plausibility_gate == "FAIL"
    assert "preflight_failed" in report.blockers


def test_failed_static_gate_blocks_claim_bearing_verification() -> None:
    hypothesis = base_hypothesis(static_payload(decoder_vector=[0.0, 1.0, 0.0]))
    lock = PredictionLock(
        wb_ref="lock_static",
        hypothesis_ref="hyp_static",
        hypothesis_spec_hash="sha256:test",
        expected_direction="target_delta_positive",
        expected_controls={"negation_removed": "low_delta"},
        git_state={"commit": "abc123"},
        environment={"python": "3.12"},
    )

    result = run_verify(
        hypothesis,
        prediction_lock=lock,
        diagnostic_only=False,
        dry_run=True,
    )

    assert result.status == "blocked"
    assert result.evidence_posture == "blocked"
    assert "static_compiler_failed" in result.metadata["blockers"]


def test_compile_hypothesis_cli_writes_report(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    monkeypatch.chdir(tmp_path)
    hypothesis_file = tmp_path / "hypothesis.json"
    hypothesis_file.write_text(json.dumps(base_hypothesis(static_payload())), encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(app, ["compile", "hypothesis", str(hypothesis_file)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "pass"
    assert payload["plausibility_gate"] == "PASS"
    assert (tmp_path / ".mechanism" / "static_compiler" / "latest_static_compile.json").exists()
    rebuilt = rebuild_sqlite_index(
        project,
        output_path=tmp_path / ".mechanism" / "rebuilt.sqlite",
    )
    assert rebuilt["counts"]["static_compiler_reports"] == 1
    assert rebuilt["counts"]["static_check_results"] == 3
