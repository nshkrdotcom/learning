import json
import sqlite3
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mwb.cli import app
from mwb.project import Project, ProjectManager


def init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_phase12_fixture(project: Project) -> dict[str, str]:
    refs = {
        "unit": "unit_sae_12300",
        "hypothesis": "hyp_negation_detector",
        "run": "run_phase12_control_leaky",
        "card": "mc_phase12_control_leaky",
        "claim": "claim_phase12_negation",
        "debt": "debt_phase12_control_leaky",
        "artifact": "art_phase12_plot",
        "cell": "cell_000001",
        "object": "obj_phase12_features",
    }

    hypotheses_dir = project.mechanism_dir / "hypotheses"
    hypotheses_dir.mkdir(parents=True, exist_ok=True)
    (hypotheses_dir / f"{refs['hypothesis']}.json").write_text(
        json.dumps(
            {
                "wb_ref": refs["hypothesis"],
                "wb_type": "Hypothesis",
                "wb_version": "1",
                "created_at": "2026-06-26T00:00:00Z",
                "parents": [refs["unit"], "bundle_targets", "bundle_controls"],
                "metadata": {},
                "title": "Negation detector",
                "units": [refs["unit"]],
                "example_bundle_ref": "bundle_targets",
                "control_bundle_ref": "bundle_controls",
                "expected_effect": "target delta exceeds controls",
                "required_controls": ["matched_controls"],
                "alternative_explanations": ["control_leaky"],
                "requested_evidence_tier": "causal_necessity",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    run_dir = project.mechanism_dir / "runs" / refs["run"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_ref": refs["run"],
                "source_hypothesis_ref": refs["hypothesis"],
                "source_kind": "phase12_fixture",
                "status": "insufficient_evidence",
                "claim_bearing": False,
                "evidence_posture": "diagnostic_insufficient",
                "created_at": "2026-06-26T00:01:00Z",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "blocker_report.json").write_text(
        json.dumps(
            {
                "wb_ref": "blocker_phase12_control_leaky",
                "wb_type": "BlockerReport",
                "run_ref": refs["run"],
                "blockers": ["control_leaky"],
                "primary_blocker": "control_leaky",
                "blocking_metrics": [
                    {"name": "matched_control_delta", "status": "failed", "value": 0.93}
                ],
                "parents": [refs["run"]],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "mechanism_card.json").write_text(
        json.dumps(
            {
                "wb_ref": refs["card"],
                "wb_type": "MechanismCard",
                "wb_version": "1",
                "title": "MechanismCard: phase 12",
                "status": "insufficient_evidence",
                "evidence_tier": "association",
                "run_ref": refs["run"],
                "allowed_language": ["is associated with"],
                "blocked_language": ["implements", "mechanism for"],
                "artifact_refs": [],
                "claim_ref": refs["claim"],
                "blockers": ["control_leaky"],
                "metadata": {
                    "claim_ref": refs["claim"],
                    "blockers": ["control_leaky"],
                },
                "parents": [refs["run"]],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "scientific_debt.json").write_text(
        json.dumps(
            {
                "run_ref": refs["run"],
                "mechanism_card_ref": refs["card"],
                "status": "insufficient_evidence",
                "items": [
                    {
                        "debt_ref": refs["debt"],
                        "kind": "controls",
                        "blocker": "control_leaky",
                        "description": "Controls moved too much.",
                        "required_resolution": "Refresh controls.",
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    session_dir = project.mechanism_dir / "sessions" / "sess_phase12"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "session_ref": "sess_phase12",
                "project_ref": project.project_ref,
                "surface": "ipython",
                "mode": "scratch",
                "started_at": "2026-06-26T00:00:00Z",
                "ended_at": None,
                "workspace": ".mechanism",
                "sqlite_path": ".mechanism/workbench.sqlite",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (session_dir / "cells.jsonl").write_text(
        json.dumps(
            {
                "cell_ref": refs["cell"],
                "session_ref": "sess_phase12",
                "execution_index": 1,
                "source_hash": "sha256:test",
                "status": "ok",
                "created_object_refs": [refs["object"]],
                "mutated_object_refs": [],
                "artifact_refs": [refs["artifact"]],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (session_dir / "namespace_objects.jsonl").write_text(
        json.dumps(
            {
                "event": "object_registered",
                "session_ref": "sess_phase12",
                "cell_ref": refs["cell"],
                "variable_name": "features",
                "object_ref": refs["object"],
                "object_type": "FeatureRanking",
                "parents": [refs["unit"]],
                "object_payload": {
                    "wb_ref": refs["object"],
                    "wb_type": "FeatureRanking",
                    "wb_version": "1",
                    "created_at": "2026-06-26T00:00:00Z",
                    "parents": [refs["unit"]],
                    "metadata": {"artifact_refs": [refs["artifact"]]},
                    "dictionary_ref": "dict_phase12",
                    "activation_ref": "acts_phase12",
                    "contrast": "target_vs_controls",
                    "rows": [{"feature_index": 12300, "score": 0.7}],
                },
                "created_at": "2026-06-26T00:00:00Z",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (session_dir / "artifacts.jsonl").write_text(
        json.dumps(
            {
                "artifact_ref": refs["artifact"],
                "session_ref": "sess_phase12",
                "cell_ref": refs["cell"],
                "path": ".mechanism/artifacts/figures/phase12.png",
                "role": "figure",
                "sha256": "sha256:abc",
                "byte_count": 3,
                "mime_type": "image/png",
                "created_by_ref": refs["object"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return refs


def test_evidence_edge_accepts_required_relations_and_rejects_unknown() -> None:
    from mwb.domain.objects import EvidenceEdge

    required = {
        "supports",
        "contradicts",
        "depends_on",
        "derived_from",
        "tested_by",
        "confounded_by",
        "fails_on",
        "generalizes_to",
        "cited_by",
    }

    for relation in required:
        edge = EvidenceEdge(
            wb_ref=f"edge_{relation}",
            src_ref="src",
            dst_ref="dst",
            relation=relation,
        )
        assert edge.relation == relation

    with pytest.raises(ValueError):
        EvidenceEdge(wb_ref="edge_bad", src_ref="src", dst_ref="dst", relation="mentions")


def test_graph_rebuild_persists_jsonl_and_rebuildable_sqlite(
    tmp_path: Path, monkeypatch
) -> None:
    from mwb.evidence_graph import EvidenceGraphService
    from mwb.sqlite_index import fetch_payload, rebuild_sqlite_index

    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    refs = write_phase12_fixture(project)
    monkeypatch.chdir(tmp_path)

    report = EvidenceGraphService(project).rebuild()

    assert report["status"] == "ok"
    assert report["counts"]["edges"] >= 12
    edge_path = project.mechanism_dir / "graph" / "evidence_edges.jsonl"
    edges = read_jsonl(edge_path)
    edge_relations = {edge["relation"] for edge in edges}
    assert {"depends_on", "tested_by", "supports", "contradicts", "fails_on"} <= edge_relations
    assert any(
        edge["src_ref"] == refs["cell"]
        and edge["dst_ref"] == refs["artifact"]
        and edge["relation"] == "derived_from"
        for edge in edges
    )

    with sqlite3.connect(project.sqlite_path) as conn:
        indexed_edges = conn.execute("select count(*) from evidence_edges").fetchone()[0]
    assert indexed_edges == len(edges)

    rebuilt = tmp_path / "rebuilt.sqlite"
    rebuilt_report = rebuild_sqlite_index(project, output_path=rebuilt)
    assert rebuilt_report["counts"]["evidence_edges"] == len(edges)
    assert fetch_payload(rebuilt, "evidence_edges", edges[0]["wb_ref"])["relation"] in edge_relations


def test_graph_query_cli_answers_phase12_questions(tmp_path: Path, monkeypatch) -> None:
    init_git_repo(tmp_path)
    project = ProjectManager.init(tmp_path, name="self-ground")
    refs = write_phase12_fixture(project)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    rebuild = runner.invoke(app, ["graph", "rebuild"])
    assert rebuild.exit_code == 0, rebuild.output
    assert json.loads(rebuild.output)["status"] == "ok"

    claims = runner.invoke(app, ["graph", "query", "claims-depending-on", refs["unit"]])
    controls = runner.invoke(app, ["graph", "query", "controls-contradicting", refs["run"]])
    cells = runner.invoke(app, ["graph", "query", "cells-producing", refs["artifact"]])
    debt = runner.invoke(app, ["graph", "query", "debt-blocking", refs["claim"]])

    assert claims.exit_code == 0, claims.output
    assert controls.exit_code == 0, controls.output
    assert cells.exit_code == 0, cells.output
    assert debt.exit_code == 0, debt.output
    assert refs["claim"] in [row["ref"] for row in json.loads(claims.output)["results"]]
    assert "control:control_leaky" in [row["ref"] for row in json.loads(controls.output)["results"]]
    assert refs["cell"] in [row["ref"] for row in json.loads(cells.output)["results"]]
    assert refs["debt"] in [row["ref"] for row in json.loads(debt.output)["results"]]
