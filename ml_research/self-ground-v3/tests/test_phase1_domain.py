import json
import subprocess
from pathlib import Path

from mwb.artifacts import ArtifactRegistry
from mwb.domain.objects import (
    ControlBundle,
    ExampleBundle,
    ModelIdentity,
    TensorSpace,
    WorkbenchObject,
    is_workbench_object,
    object_from_dict,
)
from mwb.project import ProjectManager
from mwb.refs import stable_ref
from mwb.sqlite_index import fetch_payload, insert_payload


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def test_stable_ref_is_deterministic_and_prefixed() -> None:
    first = stable_ref("model", "EleutherAI/pythia-70m-deduped", "main")
    second = stable_ref("model", "EleutherAI/pythia-70m-deduped", "main")
    different = stable_ref("model", "EleutherAI/pythia-70m-deduped", "other")

    assert first == second
    assert first.startswith("model_")
    assert first != different


def test_workbench_object_protocol_and_fingerprint_changes() -> None:
    obj = WorkbenchObject(
        wb_ref="obj_001",
        wb_type="FeatureRanking",
        wb_version="1",
        parents=["obj_parent"],
        metadata={"score": 1.0},
    )
    changed = obj.model_copy(update={"wb_version": "2"})

    assert is_workbench_object(obj)
    assert obj.wb_parents() == ["obj_parent"]
    assert obj.wb_summary()["metadata"]["score"] == 1.0
    assert obj.wb_fingerprint() != changed.wb_fingerprint()


def test_domain_objects_roundtrip_through_json() -> None:
    model = ModelIdentity(
        wb_ref="model_pythia",
        provider="huggingface",
        model_name="EleutherAI/pythia-70m-deduped",
        backend="TransformerLens",
        backend_version="2.x",
        tokenizer_ref="tok_001",
        config_hash="sha256:abc",
    )
    encoded = json.loads(model.model_dump_json())
    decoded = object_from_dict(encoded)

    assert isinstance(decoded, ModelIdentity)
    assert decoded.model_name == model.model_name
    assert decoded.wb_fingerprint() == model.wb_fingerprint()


def test_bundles_preserve_control_identity() -> None:
    examples = ExampleBundle(
        wb_ref="bundle_examples",
        name="phase3_targets",
        domain="negation",
        examples=[{"prompt": "The answer is not yes", "target": "no"}],
        source="research/bundles/negation_phase3_calibrated.yaml",
    )
    controls = ControlBundle(
        wb_ref="bundle_controls",
        name="phase3_controls",
        domain="negation",
        control_families={
            "negation_removed": [{"prompt": "The answer is yes", "target": "yes"}]
        },
        source="research/bundles/negation_phase3_calibrated.yaml",
        parents=[examples.wb_ref],
    )

    assert controls.wb_parents() == [examples.wb_ref]
    assert controls.control_families["negation_removed"][0]["target"] == "yes"


def test_artifact_registry_hashes_file_and_indexes_sqlite(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    source = tmp_path / "feature_rankings.csv"
    source.write_text("feature,score\n1,0.7\n")

    artifact = ArtifactRegistry(project).register_path(
        source,
        role="feature_ranking_table",
        created_by_ref="obj_features",
        parents=["obj_parent"],
    )

    stored = fetch_payload(project.sqlite_path, "artifacts", artifact.artifact_ref)
    assert artifact.sha256.startswith("sha256:")
    assert artifact.byte_count == source.stat().st_size
    assert stored["role"] == "feature_ranking_table"
    assert stored["parents"] == ["obj_parent"]


def test_sqlite_payload_roundtrip(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    space = TensorSpace(
        wb_ref="space_resid_2",
        model_ref="model_pythia",
        hook_point="blocks.2.hook_resid_post",
        axis_names=["batch", "position", "d_model"],
        dtype="float32",
        shape=[1, 8, 512],
    )

    insert_payload(
        project.sqlite_path,
        "tensor_spaces",
        space.wb_ref,
        space.model_dump(mode="json"),
    )
    stored = fetch_payload(project.sqlite_path, "tensor_spaces", space.wb_ref)

    assert stored["hook_point"] == "blocks.2.hook_resid_post"
    assert stored["shape"] == [1, 8, 512]
