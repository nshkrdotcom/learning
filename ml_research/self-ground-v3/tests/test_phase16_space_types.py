import json
import sqlite3
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from mwb.cli import app
from mwb.domain.objects import MechanisticUnitRef, TensorSpace
from mwb.project import ProjectManager


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def residual_space(
    ref: str,
    *,
    hook: str,
    normalization_context: str = "post_ln",
) -> TensorSpace:
    return TensorSpace(
        wb_ref=ref,
        model_ref="model_pythia",
        backend="TransformerLens",
        hook_point=hook,
        layer=2,
        stream_kind="residual_stream",
        basis="model_native",
        normalization_context=normalization_context,
        axis_names=["batch", "position", "d_model"],
        token_position_semantics="all",
        dtype="float32",
        shape=[None, None, 512],
    )


def sae_unit(ref: str, *, space_ref: str, dictionary_ref: str) -> MechanisticUnitRef:
    return MechanisticUnitRef(
        wb_ref=ref,
        unit_kind="sae_feature",
        model_ref="model_pythia",
        tensor_space_ref=space_ref,
        read_space_ref=space_ref,
        write_space_ref=space_ref,
        dictionary_ref=dictionary_ref,
        feature_index=123,
        hook_point="blocks.2.hook_resid_post",
        valid_operations=[
            "observe_activation",
            "decode_to_residual",
            "ablate",
            "resample_ablate",
            "amplify",
            "project_to_unembedding",
            "compare_decoder_cosine",
        ],
        invalid_operations=["qk_composition", "attention_pattern_patch"],
    )


def test_space_check_blocks_incompatible_sae_dictionary_comparison() -> None:
    from mwb.space_types import SpaceTypeService

    payload = {
        "operation": "compare_decoder_cosine",
        "spaces": [
            residual_space("space_a", hook="blocks.2.hook_resid_post").model_dump(mode="json")
        ],
        "units": [
            sae_unit("unit_a", space_ref="space_a", dictionary_ref="dict_a").model_dump(
                mode="json"
            ),
            sae_unit("unit_b", space_ref="space_a", dictionary_ref="dict_b").model_dump(
                mode="json"
            ),
        ],
    }

    report = SpaceTypeService().check_payload(payload)

    assert report.status == "fail"
    assert "incompatible_dictionary" in report.blockers


def test_pre_ln_post_ln_projection_requires_explicit_transform() -> None:
    from mwb.domain.objects import SpaceTransform
    from mwb.space_types import SpaceTypeService

    pre = residual_space(
        "space_pre",
        hook="blocks.2.hook_resid_pre",
        normalization_context="pre_ln",
    )
    post = residual_space(
        "space_post",
        hook="blocks.2.hook_resid_post",
        normalization_context="post_ln",
    )
    payload = {
        "operation": "project_to_unembedding",
        "source_space_ref": "space_pre",
        "target_space_ref": "space_post",
        "spaces": [pre.model_dump(mode="json"), post.model_dump(mode="json")],
    }
    transform = SpaceTransform(
        wb_ref="xform_fold_ln",
        from_space_ref="space_pre",
        to_space_ref="space_post",
        transform_kind="fold_layernorm",
        provenance_ref="preflight_report_001",
    )

    blocked = SpaceTypeService().check_payload(payload)
    allowed = SpaceTypeService().check_payload(
        {**payload, "transforms": [transform.model_dump(mode="json")]}
    )

    assert blocked.status == "fail"
    assert "normalization_context_mismatch" in blocked.blockers
    assert allowed.status == "pass"
    assert allowed.transform_refs == ["xform_fold_ln"]


def test_transform_without_provenance_fails() -> None:
    from mwb.domain.objects import SpaceTransform
    from mwb.space_types import SpaceTypeService

    pre = residual_space(
        "space_pre",
        hook="blocks.2.hook_resid_pre",
        normalization_context="pre_ln",
    )
    post = residual_space(
        "space_post",
        hook="blocks.2.hook_resid_post",
        normalization_context="post_ln",
    )
    transform = SpaceTransform(
        wb_ref="xform_fold_ln",
        from_space_ref="space_pre",
        to_space_ref="space_post",
        transform_kind="fold_layernorm",
        provenance_ref="",
    )
    payload = {
        "operation": "project_to_unembedding",
        "source_space_ref": "space_pre",
        "target_space_ref": "space_post",
        "spaces": [pre.model_dump(mode="json"), post.model_dump(mode="json")],
        "transforms": [transform.model_dump(mode="json")],
    }

    report = SpaceTypeService().check_payload(payload)

    assert report.status == "fail"
    assert "missing_transform_provenance" in report.blockers


def test_unknown_space_references_fail_closed() -> None:
    from mwb.space_types import SpaceTypeService

    source = residual_space("space_source", hook="blocks.2.hook_resid_post")
    unit = sae_unit("unit_a", space_ref="missing_unit_space", dictionary_ref="dict_a")
    payload = {
        "operation": "ablate",
        "source_space_ref": "space_source",
        "target_space_ref": "missing_target_space",
        "spaces": [source.model_dump(mode="json")],
        "units": [unit.model_dump(mode="json")],
    }

    report = SpaceTypeService().check_payload(payload)

    assert report.status == "fail"
    assert "unknown_target_space" in report.blockers
    assert "unknown_unit_space" in report.blockers


def test_wrong_hook_patch_target_fails() -> None:
    from mwb.space_types import SpaceTypeService

    source = residual_space("space_source", hook="blocks.2.hook_resid_post")
    target = residual_space("space_target", hook="blocks.3.hook_resid_post")
    unit = sae_unit("unit_a", space_ref="space_source", dictionary_ref="dict_a")
    payload = {
        "operation": "ablate",
        "source_space_ref": "space_source",
        "target_space_ref": "space_target",
        "spaces": [source.model_dump(mode="json"), target.model_dump(mode="json")],
        "units": [unit.model_dump(mode="json")],
    }

    report = SpaceTypeService().check_payload(payload)

    assert report.status == "fail"
    assert "wrong_hook_point" in report.blockers


def test_mechanistic_unit_registry_accepts_and_rejects_operations() -> None:
    from mwb.space_types import MechanisticUnitRegistry

    unit = sae_unit("unit_a", space_ref="space_a", dictionary_ref="dict_a")
    registry = MechanisticUnitRegistry([unit])

    assert registry.check_operation("unit_a", "ablate").status == "pass"
    invalid = registry.check_operation("unit_a", "attention_pattern_patch")
    assert invalid.status == "fail"
    assert "invalid_operation_for_unit" in invalid.blockers


def test_space_check_cli_writes_report(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")
    monkeypatch.chdir(tmp_path)
    fixture = tmp_path / "space_check.json"
    fixture.write_text(
        json.dumps(
            {
                "operation": "compare_decoder_cosine",
                "spaces": [
                    residual_space("space_a", hook="blocks.2.hook_resid_post").model_dump(
                        mode="json"
                    )
                ],
                "units": [
                    sae_unit("unit_a", space_ref="space_a", dictionary_ref="dict_a").model_dump(
                        mode="json"
                    ),
                    sae_unit("unit_b", space_ref="space_a", dictionary_ref="dict_b").model_dump(
                        mode="json"
                    ),
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["space", "check", str(fixture)])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert "incompatible_dictionary" in payload["blockers"]
    assert (tmp_path / ".mechanism" / "space_checks" / "latest_space_check.json").exists()


def test_doctor_repairs_rebuildable_sqlite_schema_drift(tmp_path: Path) -> None:
    from mwb.doctor import run_doctor

    init_git_repo(tmp_path)
    ProjectManager.init(tmp_path, name="self-ground")
    sqlite_path = tmp_path / ".mechanism" / "workbench.sqlite"
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("drop table space_checks")

    report = run_doctor(tmp_path)
    with sqlite3.connect(sqlite_path) as conn:
        restored = conn.execute(
            "select name from sqlite_master where type = 'table' and name = 'space_checks'"
        ).fetchone()

    assert report.status == "ok"
    assert restored is not None
