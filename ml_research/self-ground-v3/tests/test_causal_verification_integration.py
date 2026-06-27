import json
import os
import subprocess
from pathlib import Path

import pytest

from mwb.context import RunContext
from mwb.project import ProjectManager
from mwb.session import SessionManager


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("MWB_RUN_REAL_ADAPTER_TESTS") != "1",
    reason="set MWB_RUN_REAL_ADAPTER_TESTS=1 to run real causal verification path",
)
def test_real_pythia_sae_resample_ablation_writes_receipt(tmp_path: Path, monkeypatch) -> None:
    from mwb.causal_verification import CausalVerificationService

    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    session = SessionManager.start(project, surface="test")
    ctx = RunContext(project=project, session=session)
    model = ctx.models.load_tl("EleutherAI/pythia-70m-deduped", device="cpu")
    sae = ctx.saes.load("pythia-70m-deduped-res-sm", hook="blocks.2.hook_resid_post")

    run = CausalVerificationService(project).resample_ablate_sae_feature(
        hypothesis_ref="hyp_real_causal",
        model=model._backend_model,
        sae=sae._backend_sae,
        hook_point="blocks.2.hook_resid_post",
        feature_index=0,
        clean_prompt="The opposite of good is",
        corrupt_prompt="The opposite of bad is",
        target_token=" bad",
        foil_token=" good",
        diagnostic_only=True,
    )

    run_dir = Path(run.metadata["run_dir"])
    receipts = [
        json.loads(line)
        for line in (run_dir / "intervention_receipts.jsonl").read_text().splitlines()
    ]

    assert run.evidence_posture == "diagnostic_only"
    assert receipts[0]["operation"] == "resample_ablate"
    assert receipts[0]["backend_executed"] is True
    assert "kl_drift" in run.metrics
