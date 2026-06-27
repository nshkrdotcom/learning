import json
import subprocess
from pathlib import Path

from ruamel.yaml import YAML
from typer.testing import CliRunner

from mwb.cli import app
from mwb.project import Project, ProjectManager
from mwb.sqlite_index import fetch_payload, rebuild_sqlite_index
from mwb.workflows.diagnosis import DiagnosisService, ProbeRegistry
from mwb.workflows.next_probe import build_next_probe, load_next_probe_payload


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def write_diagnostic_run(project: Project, run_ref: str = "run_diag") -> Path:
    run_dir = project.mechanism_dir / "runs" / run_ref
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_ref": run_ref,
                "source_hypothesis_ref": "hyp_diag",
                "status": "insufficient_evidence",
                "tried_axes": {"layers": ["1", "2"], "patch_modes": ["delta"]},
                "available_axes": {
                    "layers": ["0", "1", "2"],
                    "patch_modes": ["delta", "direct"],
                },
                "backend_capabilities": {"direct_patch": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "control_metrics.json").write_text(
        json.dumps(
            {
                "target_delta": 0.5,
                "matched_control_delta": 0.45,
                "specificity_gap": 0.05,
                "family_min_gap": -0.02,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "blocker_report.json").write_text(
        json.dumps(
            {
                "wb_ref": f"blocker_{run_ref}",
                "wb_type": "BlockerReport",
                "run_ref": run_ref,
                "blockers": ["control_leaky", "specificity_gap_failed"],
                "primary_blocker": "control_leaky",
                "blocking_metrics": [
                    {
                        "name": "matched_control_delta",
                        "value": 0.45,
                        "threshold": 0.4,
                        "status": "failed",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "scientific_debt.json").write_text(
        json.dumps(
            {
                "wb_ref": f"debt_{run_ref}",
                "run_ref": run_ref,
                "items": [
                    {
                        "debt_ref": f"debt_{run_ref}_controls",
                        "status": "unresolved",
                        "blocker": "control_leaky",
                        "reason": "matched controls moved with the target intervention",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_diagnosis_tree_from_blocker_report_preserves_provenance(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    run_dir = write_diagnostic_run(project)

    service = DiagnosisService(project)
    tree = service.diagnose_run_dir(run_dir)
    service.write_diagnosis(run_dir, tree)

    assert tree.primary_blocker == "control_leaky"
    assert [node["blocker"] for node in tree.nodes[:2]] == [
        "control_leaky",
        "specificity_gap_failed",
    ]
    assert tree.negative_evidence[0]["blocker"] == "control_leaky"
    assert any(
        ref["artifact"] == "blocker_report.json" and ref["jsonpath"] == "$.blockers[0]"
        for ref in tree.source_refs
    )
    written = json.loads((run_dir / "diagnosis_tree.json").read_text(encoding="utf-8"))
    assert written["wb_ref"] == tree.wb_ref
    assert written["source_run_ref"] == "run_diag"


def test_probe_template_registry_materializes_deterministic_supported_probe(
    tmp_path: Path,
) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    run_dir = write_diagnostic_run(project)
    payload = load_next_probe_payload(run_dir)
    plan = build_next_probe(payload)
    tree = DiagnosisService(project).diagnose_run_dir(run_dir)
    registry = ProbeRegistry()

    probe_a = registry.materialize(plan, tree, payload)
    probe_b = registry.materialize(plan, tree, payload)

    assert probe_a.wb_ref == probe_b.wb_ref
    assert probe_a.template_id == "sweep_axis_extension.v1"
    assert probe_a.probe_kind == "sweep_axis_extension"
    assert probe_a.runnable is True
    assert probe_a.command == [
        "uv",
        "run",
        "mwb",
        "sweep",
        "hyp_diag",
        "--axis",
        "layer=0",
        "--dry-run",
    ]
    assert any(ref["jsonpath"] == "$.available_axes.layers[0]" for ref in probe_a.provenance)


def test_next_probe_materialize_and_run_probe_cli_execute_real_dry_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    run_dir = write_diagnostic_run(project)
    runner = CliRunner()

    diagnose = runner.invoke(app, ["diagnose", "latest"])
    materialize = runner.invoke(app, ["next-probe", "latest", "--materialize"])
    run_probe = runner.invoke(app, ["run-probe", str(run_dir / "probe.yaml")])

    assert diagnose.exit_code == 0, diagnose.output
    assert materialize.exit_code == 0, materialize.output
    assert run_probe.exit_code == 0, run_probe.output
    probe_payload = YAML().load((run_dir / "probe.yaml").read_text(encoding="utf-8"))
    assert probe_payload["source_run_ref"] == "run_diag"
    assert probe_payload["next_probe_ref"] == json.loads(materialize.output)["wb_ref"]
    assert probe_payload["runnable"] is True
    assert all(ref["artifact"] for ref in probe_payload["provenance"])
    run_payload = json.loads(run_probe.output)
    assert run_payload["status"] == "dry_run"
    assert Path(run_payload["run_dir"]).exists()
    assert json.loads((Path(run_payload["run_dir"]) / "run_manifest.json").read_text())[
        "source_hypothesis_ref"
    ] == "hyp_diag"


def test_unsupported_probe_commands_are_not_emitted_or_runnable(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    run_dir = write_diagnostic_run(project, run_ref="run_unsupported")
    blocker = json.loads((run_dir / "blocker_report.json").read_text(encoding="utf-8"))
    blocker["blockers"] = ["dictionary_interference"]
    blocker["primary_blocker"] = "dictionary_interference"
    (run_dir / "blocker_report.json").write_text(json.dumps(blocker) + "\n", encoding="utf-8")
    payload = load_next_probe_payload(run_dir)
    plan = build_next_probe(payload)
    service = DiagnosisService(project)
    tree = service.diagnose_run_dir(run_dir)
    probe = service.materialize_probe(run_dir, plan=plan, tree=tree)

    assert probe.runnable is False
    assert probe.command == []
    assert probe.status == "blocked"

    runner = CliRunner()
    run_probe = runner.invoke(app, ["run-probe", str(run_dir / "probe.yaml")])
    assert run_probe.exit_code == 1
    assert "unsupported probe kind" in run_probe.output


def test_artifact_incomplete_materialization_records_blocked_probe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    run_dir = project.mechanism_dir / "runs" / "run_incomplete"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_ref": "run_incomplete",
                "source_hypothesis_ref": "hyp_incomplete",
                "status": "dry_run",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    plain = runner.invoke(app, ["next-probe", "latest"])
    materialized = runner.invoke(app, ["next-probe", "latest", "--materialize"])

    assert plain.exit_code == 1, plain.output
    assert materialized.exit_code == 0, materialized.output
    probe = YAML().load((run_dir / "probe.yaml").read_text(encoding="utf-8"))
    assert probe["status"] == "blocked"
    assert probe["runnable"] is False
    assert probe["command"] == []


def test_diagnosis_and_materialized_probe_restore_to_sqlite(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    run_dir = write_diagnostic_run(project)
    service = DiagnosisService(project)
    tree = service.write_diagnosis(run_dir, service.diagnose_run_dir(run_dir))
    probe = service.materialize_probe(run_dir)

    report = rebuild_sqlite_index(project, output_path=tmp_path / "rebuilt.sqlite")

    assert report["counts"]["diagnosis_trees"] == 1
    assert report["counts"]["materialized_probes"] == 1
    indexed_tree = fetch_payload(tmp_path / "rebuilt.sqlite", "diagnosis_trees", tree.wb_ref)
    indexed_probe = fetch_payload(tmp_path / "rebuilt.sqlite", "materialized_probes", probe.wb_ref)
    assert indexed_tree["primary_blocker"] == "control_leaky"
    assert indexed_probe["diagnosis_tree_ref"] == tree.wb_ref
