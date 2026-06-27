import json
import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mwb.cli import app
from mwb.context import RunContext
from mwb.project import ProjectManager
from mwb.session import SessionManager


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def test_negation_bundle_loads_as_first_class_objects(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    session = SessionManager.start(project, surface="test")
    ctx = RunContext(project=project, session=session)

    bundle = ctx.domains.negation.load("phase3_calibrated")

    assert bundle.wb_type == "DomainBundle"
    assert bundle.targets.wb_type == "ExampleBundle"
    assert bundle.controls.wb_type == "ControlBundle"
    assert bundle.targets.examples
    assert "negation_removed" in bundle.controls.control_families
    assert bundle.targets.bundle_hash == bundle.controls.bundle_hash


def test_demo_negation_dry_run_validates_real_bundle(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["demo", "negation", "--dry-run", "--model", "EleutherAI/pythia-70m-deduped"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "dry_run"
    assert payload["bundle_ref"].startswith("bundle_")
    assert payload["control_bundle_ref"].startswith("ctrl_")


def test_ipython_captures_loaded_bundle(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["ipython", "--execute", "bundle = ctx.domains.negation.load('phase3_calibrated')"],
    )

    assert result.exit_code == 0, result.output
    sessions = sorted((tmp_path / ".mechanism" / "sessions").glob("sess_*"))
    objects_path = sessions[-1] / "namespace_objects.jsonl"
    object_events = [json.loads(line) for line in objects_path.read_text().splitlines()]
    assert object_events[0]["object_type"] == "DomainBundle"


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("MWB_RUN_REAL_ADAPTER_TESTS") != "1",
    reason="set MWB_RUN_REAL_ADAPTER_TESTS=1 to run real TransformerLens/SAELens workflow",
)
def test_real_ctx_model_sae_capture_and_rank(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    session = SessionManager.start(project, surface="test")
    ctx = RunContext(project=project, session=session)

    model = ctx.models.load_tl("EleutherAI/pythia-70m-deduped", device="cpu")
    sae = ctx.saes.load("pythia-70m-deduped-res-sm", hook="blocks.2.hook_resid_post")
    bundle = ctx.domains.negation.load("phase3_calibrated")
    acts = ctx.capture(model, bundle).at("blocks.2.hook_resid_post")
    features = ctx.features.rank(sae, acts, contrast="target_vs_controls", top_k=3)

    assert acts.tensor_space_ref
    assert len(features.rows) == 3
    assert features.rows[0]["feature_index"] >= 0

