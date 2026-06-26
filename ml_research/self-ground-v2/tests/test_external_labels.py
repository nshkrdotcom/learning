from __future__ import annotations

import json
from pathlib import Path

from helpers_project import populate_project, runner

from mechledger.cli import app


def label_record(label_id: str = "L001") -> dict[str, object]:
    return {
        "label_id": label_id,
        "source": "neuronpedia",
        "source_url": "https://example.invalid/feature",
        "source_model": "explainer-v1",
        "label_text": "negation feature",
        "feature_id": "sae_123",
        "model": "pythia",
        "layer_or_hook": "blocks.2.hook_resid_post",
        "sae_release": "release",
        "sae_id": "sae-id",
        "created_at": "2026-06-25T00:00:00Z",
        "confidence": 0.7,
        "license": "CC-BY",
        "linked_claims": [],
        "notes": "external metadata only",
    }


def test_labels_import_list_show_link_and_export_metadata(tmp_path: Path) -> None:
    populate_project(tmp_path)
    source = tmp_path / "labels.jsonl"
    source.write_text(json.dumps(label_record()) + "\n", encoding="utf-8")

    imported = runner.invoke(
        app,
        ["labels", "import", str(source)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert imported.exit_code == 0, imported.output
    registry = tmp_path / "research/literature/external_labels.jsonl"
    assert registry.exists()

    listed = runner.invoke(
        app,
        ["labels", "list"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    shown = runner.invoke(
        app,
        ["labels", "show", "L001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert listed.exit_code == 0 and "L001" in listed.output
    assert shown.exit_code == 0 and "negation feature" in shown.output

    linked = runner.invoke(
        app,
        ["labels", "link", "L001", "--claim", "C001"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert linked.exit_code == 0, linked.output
    linked_payload = json.loads(registry.read_text(encoding="utf-8").splitlines()[0])
    assert linked_payload["linked_claims"] == ["C001"]

    out = tmp_path / "bundles/ro-crate"
    exported = runner.invoke(
        app,
        ["export", "ro-crate", "--out", str(out)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert exported.exit_code == 0, exported.output
    crate = json.loads((out / "ro-crate-metadata.json").read_text(encoding="utf-8"))
    assert any(
        entity.get("@id") == "research/literature/external_labels.jsonl#L001"
        for entity in crate["@graph"]
    )


def test_labels_invalid_import_and_unknown_claim_link_fail(tmp_path: Path) -> None:
    populate_project(tmp_path)
    bad_source = tmp_path / "bad_labels.jsonl"
    bad_source.write_text(json.dumps({"label_id": "L_BAD"}) + "\n", encoding="utf-8")

    invalid = runner.invoke(
        app,
        ["labels", "import", str(bad_source)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert invalid.exit_code == 2
    assert "label_text" in invalid.output

    source = tmp_path / "labels.jsonl"
    source.write_text(json.dumps(label_record("L002")) + "\n", encoding="utf-8")
    runner.invoke(
        app,
        ["labels", "import", str(source)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    unknown = runner.invoke(
        app,
        ["labels", "link", "L002", "--claim", "C999"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert unknown.exit_code == 2
    assert "Unknown claim" in unknown.output
