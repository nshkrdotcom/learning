import math
import os
import subprocess
from pathlib import Path

import pytest

from mwb.context import RunContext
from mwb.project import ProjectManager
from mwb.session import SessionManager
from mwb.static_compiler import StaticCompiler


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
    reason="set MWB_RUN_REAL_ADAPTER_TESTS=1 to run real TransformerLens/SAELens compiler path",
)
def test_real_adapter_weights_compile_static_projection(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    session = SessionManager.start(project, surface="test")
    ctx = RunContext(project=project, session=session)

    model = ctx.models.load_tl("EleutherAI/pythia-70m-deduped", device="cpu")
    sae = ctx.saes.load("pythia-70m-deduped-res-sm", hook="blocks.2.hook_resid_post")
    raw_model = model._backend_model
    raw_sae = sae._backend_sae
    target_token = int(raw_model.to_single_token(" bad"))
    foil_token = int(raw_model.to_single_token(" good"))
    decoder_matrix = raw_sae.W_dec.detach().cpu().float()
    unembedding_matrix = raw_model.W_U.detach().cpu().float()
    feature_index = 0
    neighbor_index = 1

    payload = {
        "wb_ref": "hyp_real_static",
        "title": "real adapter static compiler integration",
        "units": [f"feature_{feature_index}"],
        "example_bundle_ref": "bundle_real",
        "control_bundle_ref": "ctrl_real",
        "expected_effect": "target_delta > controls",
        "required_controls": ["negation_removed"],
        "static_compiler": {
            "tensor_space_ref": "blocks.2.hook_resid_post",
            "unembedding_space_ref": "unembed.W_U",
            "target_token_ids": [target_token],
            "foil_token_ids": [foil_token],
            "decoder_vector": decoder_matrix[feature_index].tolist(),
            "unembedding": {
                str(target_token): _unembedding_vector(unembedding_matrix, target_token),
                str(foil_token): _unembedding_vector(unembedding_matrix, foil_token),
            },
            "dictionary": {
                "feature_id": f"feature_{feature_index}",
                "decoder_vectors": {
                    f"feature_{feature_index}": decoder_matrix[feature_index].tolist(),
                    f"feature_{neighbor_index}": decoder_matrix[neighbor_index].tolist(),
                },
            },
            "activation_density": {"target": 0.1, "control": 0.1, "max_ratio": 1.5},
        },
    }

    report = StaticCompiler().compile_payload(payload)
    projection = next(
        check for check in report.checks if check["name"] == "decoder_unembed_projection"
    )

    assert report.plausibility_gate in {"PASS", "WEAK", "FAIL"}
    assert math.isfinite(projection["score"])
    assert projection["normalization"] == "l2_cosine"


def _unembedding_vector(unembedding_matrix, token_id: int) -> list[float]:
    if int(unembedding_matrix.shape[0]) <= token_id:
        return unembedding_matrix[:, token_id].tolist()
    return unembedding_matrix[token_id].tolist()
