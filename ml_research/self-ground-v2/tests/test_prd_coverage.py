from __future__ import annotations

import json
from pathlib import Path

VALID_STATUSES = {
    "implemented",
    "partially_implemented",
    "deferred_by_prd",
    "intentionally_out_of_scope",
    "missing",
    "ambiguous_or_requires_decision",
}


def _coverage() -> dict:
    return json.loads(Path("docs/prd_coverage_0430_0432.json").read_text(encoding="utf-8"))


def test_prd_coverage_names_source_documents() -> None:
    coverage = _coverage()
    source_documents = set(coverage["source_documents"])

    assert "0430_revised_v6.md" in source_documents
    assert "0431_selfground_refactor.md" in source_documents
    assert "0432_selfground_refactor.md" in source_documents


def test_prd_coverage_status_values_are_valid() -> None:
    coverage = _coverage()

    assert coverage["entries"]
    assert {entry["status"] for entry in coverage["entries"]} <= VALID_STATUSES


def test_prd_coverage_has_milestone_50_entries() -> None:
    coverage = _coverage()
    sections = {entry["doc_section"] for entry in coverage["entries"]}

    for section in ("50.0", "50.1", "50.2", "50.3", "50.4", "50.5", "50.6"):
        assert section in sections


def test_prd_coverage_explicitly_marks_deferred_features() -> None:
    coverage = _coverage()

    assert any(entry["status"] == "deferred_by_prd" for entry in coverage["entries"])
    assert any(
        "dashboard server" in entry["title"].lower()
        and entry["status"] == "deferred_by_prd"
        for entry in coverage["entries"]
    )
