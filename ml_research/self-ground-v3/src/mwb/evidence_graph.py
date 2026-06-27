from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from mwb.domain.objects import EvidenceEdge, EvidenceRelation
from mwb.project import Project
from mwb.refs import stable_ref
from mwb.sqlite_index import connect, initialize_schema, insert_payload
from mwb.time import utc_now

GRAPH_DIR = "graph"
EDGE_FILE = "evidence_edges.jsonl"
SUMMARY_FILE = "graph_summary.json"
QUERY_KINDS = {
    "claims-depending-on",
    "controls-contradicting",
    "cells-producing",
    "debt-blocking",
}


class EvidenceGraphService:
    def __init__(self, project: Project) -> None:
        self.project = project
        self.graph_dir = project.mechanism_dir / GRAPH_DIR
        self.edge_path = self.graph_dir / EDGE_FILE
        self.summary_path = self.graph_dir / SUMMARY_FILE

    def rebuild(self) -> dict[str, Any]:
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        edges: list[EvidenceEdge] = []
        self._collect_hypothesis_edges(edges)
        self._collect_session_edges(edges)
        self._collect_run_edges(edges)
        self._collect_claim_edges(edges)

        deduped = sorted(_dedupe_edges(edges), key=lambda edge: edge.wb_ref)
        _write_jsonl(self.edge_path, [edge.model_dump(mode="json") for edge in deduped])
        summary = self._summary(deduped)
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        initialize_schema(self.project.sqlite_path)
        with connect(self.project.sqlite_path) as conn:
            conn.execute("delete from evidence_edges")
        for edge in deduped:
            insert_payload(
                self.project.sqlite_path,
                "evidence_edges",
                edge.wb_ref,
                edge.model_dump(mode="json"),
            )
        return {"status": "ok", "edge_path": str(self.edge_path), "counts": summary["counts"]}

    def load_edges(self) -> list[EvidenceEdge]:
        return [
            EvidenceEdge.model_validate(payload)
            for payload in _read_jsonl(self.edge_path)
        ]

    def query(self, kind: str, ref: str) -> dict[str, Any]:
        if kind not in QUERY_KINDS:
            expected = ", ".join(sorted(QUERY_KINDS))
            raise ValueError(f"unknown graph query kind {kind!r}; expected one of: {expected}")
        edges = self.load_edges()
        if kind == "claims-depending-on":
            results = self._query_claims_depending_on(edges, ref)
        elif kind == "controls-contradicting":
            results = self._query_controls_contradicting(edges, ref)
        elif kind == "cells-producing":
            results = self._query_cells_producing(edges, ref)
        else:
            results = self._query_debt_blocking(edges, ref)
        return {
            "status": "ok",
            "query": kind,
            "ref": ref,
            "count": len(results),
            "results": results,
        }

    def _collect_hypothesis_edges(self, edges: list[EvidenceEdge]) -> None:
        hypotheses_dir = self.project.mechanism_dir / "hypotheses"
        for path in sorted(hypotheses_dir.glob("*.json")) if hypotheses_dir.exists() else []:
            payload = _read_json(path)
            hypothesis_ref = _payload_ref(payload, fallback=path.stem)
            for unit_ref in _string_list(payload.get("units")):
                edges.append(
                    make_edge(
                        unit_ref,
                        hypothesis_ref,
                        "depends_on",
                        source_ref=hypothesis_ref,
                        source_path=path,
                        metadata={"src_type": "mechanistic_unit", "dst_type": "hypothesis"},
                    )
                )
            for field, src_type in [
                ("example_bundle_ref", "example_bundle"),
                ("control_bundle_ref", "control_bundle"),
            ]:
                value = payload.get(field)
                if value:
                    edges.append(
                        make_edge(
                            str(value),
                            hypothesis_ref,
                            "depends_on",
                            source_ref=hypothesis_ref,
                            source_path=path,
                            metadata={"src_type": src_type, "dst_type": "hypothesis"},
                        )
                    )
            seen_parents = set(_string_list(payload.get("units")))
            seen_parents.update(
                str(payload[key])
                for key in ["example_bundle_ref", "control_bundle_ref"]
                if payload.get(key)
            )
            for parent_ref in _string_list(payload.get("parents")):
                if parent_ref in seen_parents:
                    continue
                edges.append(
                    make_edge(
                        parent_ref,
                        hypothesis_ref,
                        "depends_on",
                        source_ref=hypothesis_ref,
                        source_path=path,
                        metadata={"src_type": "parent", "dst_type": "hypothesis"},
                    )
                )

    def _collect_session_edges(self, edges: list[EvidenceEdge]) -> None:
        sessions_dir = self.project.mechanism_dir / "sessions"
        for session_dir in sorted(sessions_dir.glob("sess_*")) if sessions_dir.exists() else []:
            for cell in _read_jsonl(session_dir / "cells.jsonl"):
                cell_ref = str(cell.get("cell_ref", ""))
                if not cell_ref:
                    continue
                for artifact_ref in _string_list(cell.get("artifact_refs")):
                    edges.append(
                        make_edge(
                            cell_ref,
                            artifact_ref,
                            "derived_from",
                            source_ref=cell_ref,
                            source_path=session_dir / "cells.jsonl",
                            metadata={"src_type": "cell", "dst_type": "artifact"},
                        )
                    )
            for event in _read_jsonl(session_dir / "namespace_objects.jsonl"):
                object_ref = str(event.get("object_ref", ""))
                cell_ref = str(event.get("cell_ref", ""))
                if object_ref and cell_ref:
                    edges.append(
                        make_edge(
                            cell_ref,
                            object_ref,
                            "derived_from",
                            source_ref=object_ref,
                            source_path=session_dir / "namespace_objects.jsonl",
                            metadata={
                                "src_type": "cell",
                                "dst_type": "object",
                                "object_type": event.get("object_type"),
                            },
                        )
                    )
                for parent_ref in _string_list(event.get("parents")):
                    edges.append(
                        make_edge(
                            parent_ref,
                            object_ref,
                            "derived_from",
                            source_ref=object_ref,
                            source_path=session_dir / "namespace_objects.jsonl",
                            metadata={
                                "src_type": "object",
                                "dst_type": "object",
                                "object_type": event.get("object_type"),
                            },
                        )
                    )
                object_payload = event.get("object_payload")
                if isinstance(object_payload, dict):
                    metadata = object_payload.get("metadata")
                    if isinstance(metadata, dict):
                        for artifact_ref in _string_list(metadata.get("artifact_refs")):
                            edges.append(
                                make_edge(
                                    object_ref,
                                    artifact_ref,
                                    "derived_from",
                                    source_ref=object_ref,
                                    source_path=session_dir / "namespace_objects.jsonl",
                                    metadata={"src_type": "object", "dst_type": "artifact"},
                                )
                            )
            for artifact in _read_jsonl(session_dir / "artifacts.jsonl"):
                artifact_ref = str(artifact.get("artifact_ref") or artifact.get("ref") or "")
                if not artifact_ref:
                    continue
                cell_ref = artifact.get("cell_ref")
                if cell_ref:
                    edges.append(
                        make_edge(
                            str(cell_ref),
                            artifact_ref,
                            "derived_from",
                            source_ref=artifact_ref,
                            source_path=session_dir / "artifacts.jsonl",
                            metadata={"src_type": "cell", "dst_type": "artifact"},
                        )
                    )
                creator_ref = artifact.get("created_by_ref")
                if creator_ref:
                    edges.append(
                        make_edge(
                            str(creator_ref),
                            artifact_ref,
                            "derived_from",
                            source_ref=artifact_ref,
                            source_path=session_dir / "artifacts.jsonl",
                            metadata={"src_type": "object", "dst_type": "artifact"},
                        )
                    )
                for parent_ref in _string_list(artifact.get("parents")):
                    edges.append(
                        make_edge(
                            parent_ref,
                            artifact_ref,
                            "depends_on",
                            source_ref=artifact_ref,
                            source_path=session_dir / "artifacts.jsonl",
                            metadata={"src_type": "parent", "dst_type": "artifact"},
                        )
                    )

    def _collect_run_edges(self, edges: list[EvidenceEdge]) -> None:
        runs_dir = self.project.mechanism_dir / "runs"
        for run_dir in sorted(runs_dir.glob("*")) if runs_dir.exists() else []:
            if not run_dir.is_dir():
                continue
            manifest = _read_json(run_dir / "run_manifest.json")
            run_ref = str(manifest.get("run_ref") or run_dir.name)
            hypothesis_ref = manifest.get("source_hypothesis_ref")
            if hypothesis_ref:
                edges.append(
                    make_edge(
                        str(hypothesis_ref),
                        run_ref,
                        "tested_by",
                        source_ref=run_ref,
                        source_path=run_dir / "run_manifest.json",
                        metadata={"src_type": "hypothesis", "dst_type": "run"},
                    )
                )
            for result in _read_jsonl(run_dir / "verification_results.jsonl"):
                result_ref = str(result.get("result_ref") or "")
                if not result_ref:
                    continue
                edges.append(
                    make_edge(
                        run_ref,
                        result_ref,
                        "derived_from",
                        source_ref=run_ref,
                        source_path=run_dir / "verification_results.jsonl",
                        metadata={"src_type": "run", "dst_type": "verification_result"},
                    )
                )
                result_hypothesis_ref = result.get("hypothesis_ref")
                if result_hypothesis_ref:
                    edges.append(
                        make_edge(
                            str(result_hypothesis_ref),
                            result_ref,
                            "tested_by",
                            source_ref=result_ref,
                            source_path=run_dir / "verification_results.jsonl",
                            metadata={"src_type": "hypothesis", "dst_type": "verification_result"},
                        )
                    )
            blocker_report = _read_json(run_dir / "blocker_report.json")
            self._collect_blocker_edges(
                edges,
                run_ref,
                blocker_report,
                run_dir / "blocker_report.json",
            )

            card = _read_json(run_dir / "mechanism_card.json")
            card_ref = str(card.get("wb_ref") or card.get("mechanism_card_ref") or "")
            claim_ref = _claim_ref(card)
            if card_ref:
                edges.append(
                    make_edge(
                        run_ref,
                        card_ref,
                        "supports",
                        source_ref=card_ref,
                        source_path=run_dir / "mechanism_card.json",
                        metadata={
                            "src_type": "run",
                            "dst_type": "mechanism_card",
                            "evidence_tier": card.get("evidence_tier"),
                            "status": card.get("status"),
                        },
                    )
                )
            if card_ref and claim_ref:
                edges.append(
                    make_edge(
                        card_ref,
                        claim_ref,
                        "supports",
                        source_ref=card_ref,
                        source_path=run_dir / "mechanism_card.json",
                        metadata={
                            "src_type": "mechanism_card",
                            "dst_type": "claim",
                            "evidence_tier": card.get("evidence_tier"),
                            "status": card.get("status"),
                        },
                    )
                )
            debt = _read_json(run_dir / "scientific_debt.json")
            self._collect_debt_edges(
                edges,
                debt,
                claim_ref=claim_ref,
                fallback_target=card_ref or run_ref,
                source_path=run_dir / "scientific_debt.json",
            )

    def _collect_claim_edges(self, edges: list[EvidenceEdge]) -> None:
        claims_dir = self.project.mechanism_dir / "claims"
        for path in sorted(claims_dir.glob("*.json")) if claims_dir.exists() else []:
            payload = _read_json(path)
            claim_ref = str(payload.get("claim_ref") or payload.get("wb_ref") or path.stem)
            card_ref = str(payload.get("mechanism_card_ref") or payload.get("wb_ref") or "")
            run_ref = payload.get("run_ref")
            if card_ref and card_ref != claim_ref:
                edges.append(
                    make_edge(
                        card_ref,
                        claim_ref,
                        "supports",
                        source_ref=claim_ref,
                        source_path=path,
                        metadata={"src_type": "mechanism_card", "dst_type": "claim"},
                    )
                )
            if run_ref:
                edges.append(
                    make_edge(
                        str(run_ref),
                        claim_ref,
                        "depends_on",
                        source_ref=claim_ref,
                        source_path=path,
                        metadata={"src_type": "run", "dst_type": "claim"},
                    )
                )
            for parent_ref in _string_list(payload.get("parents")):
                edges.append(
                    make_edge(
                        parent_ref,
                        claim_ref,
                        "depends_on",
                        source_ref=claim_ref,
                        source_path=path,
                        metadata={"src_type": "parent", "dst_type": "claim"},
                    )
                )
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                for draft_ref in _string_list(metadata.get("draft_refs")):
                    edges.append(
                        make_edge(
                            claim_ref,
                            draft_ref,
                            "cited_by",
                            source_ref=claim_ref,
                            source_path=path,
                            metadata={"src_type": "claim", "dst_type": "draft"},
                        )
                    )

    def _collect_blocker_edges(
        self,
        edges: list[EvidenceEdge],
        run_ref: str,
        blocker_report: dict[str, Any],
        source_path: Path,
    ) -> None:
        if not blocker_report:
            return
        blocker_ref = str(
            blocker_report.get("wb_ref") or stable_ref("blocker", run_ref, blocker_report)
        )
        for blocker in _string_list(blocker_report.get("blockers")):
            control_ref = f"control:{blocker}" if "control" in blocker else f"blocker:{blocker}"
            if "control" in blocker:
                edges.append(
                    make_edge(
                        control_ref,
                        run_ref,
                        "contradicts",
                        source_ref=blocker_ref,
                        source_path=source_path,
                        metadata={
                            "src_type": "control",
                            "dst_type": "run",
                            "blocker": blocker,
                            "blocking_metrics": blocker_report.get("blocking_metrics", []),
                        },
                    )
                )
            edges.append(
                make_edge(
                    blocker_ref,
                    run_ref,
                    "confounded_by",
                    source_ref=blocker_ref,
                    source_path=source_path,
                    metadata={
                        "src_type": "blocker_report",
                        "dst_type": "run",
                        "blocker": blocker,
                    },
                )
            )
            edges.append(
                make_edge(
                    blocker_ref,
                    run_ref,
                    "fails_on",
                    source_ref=blocker_ref,
                    source_path=source_path,
                    metadata={
                        "src_type": "blocker_report",
                        "dst_type": "run",
                        "blocker": blocker,
                    },
                )
            )

    def _collect_debt_edges(
        self,
        edges: list[EvidenceEdge],
        debt: dict[str, Any],
        *,
        claim_ref: str | None,
        fallback_target: str,
        source_path: Path,
    ) -> None:
        if not debt:
            return
        target_ref = claim_ref or fallback_target
        for item in debt.get("items", []):
            if not isinstance(item, dict):
                continue
            debt_ref = str(item.get("debt_ref") or stable_ref("debt", fallback_target, item))
            metadata = {
                "src_type": "scientific_debt",
                "dst_type": "claim" if claim_ref else "mechanism_card",
                "kind": item.get("kind"),
                "blocker": item.get("blocker"),
                "description": item.get("description"),
            }
            edges.append(
                make_edge(
                    debt_ref,
                    target_ref,
                    "fails_on",
                    source_ref=debt_ref,
                    source_path=source_path,
                    metadata=metadata,
                )
            )
            if item.get("blocker"):
                edges.append(
                    make_edge(
                        debt_ref,
                        target_ref,
                        "confounded_by",
                        source_ref=debt_ref,
                        source_path=source_path,
                        metadata=metadata,
                    )
                )

    def _query_claims_depending_on(
        self, edges: list[EvidenceEdge], ref: str
    ) -> list[dict[str, Any]]:
        traversable = {
            "depends_on",
            "derived_from",
            "tested_by",
            "supports",
            "generalizes_to",
            "cited_by",
        }
        adjacency: dict[str, list[EvidenceEdge]] = {}
        for edge in edges:
            if edge.relation in traversable:
                adjacency.setdefault(edge.src_ref, []).append(edge)

        queue = deque([ref])
        paths = {ref: [ref]}
        visited = {ref}
        results: list[dict[str, Any]] = []
        while queue:
            current = queue.popleft()
            for edge in adjacency.get(current, []):
                if edge.dst_ref in visited:
                    continue
                visited.add(edge.dst_ref)
                paths[edge.dst_ref] = [*paths[current], edge.dst_ref]
                if _is_claim_ref(edge.dst_ref, edge):
                    results.append(_result(edge.dst_ref, edge, path=paths[edge.dst_ref]))
                queue.append(edge.dst_ref)
        return _unique_results(results)

    def _query_controls_contradicting(
        self, edges: list[EvidenceEdge], ref: str
    ) -> list[dict[str, Any]]:
        return _unique_results(
            _result(edge.src_ref, edge)
            for edge in edges
            if edge.relation == "contradicts" and edge.dst_ref == ref
        )

    def _query_cells_producing(self, edges: list[EvidenceEdge], ref: str) -> list[dict[str, Any]]:
        return _unique_results(
            _result(edge.src_ref, edge)
            for edge in edges
            if edge.relation == "derived_from"
            and edge.dst_ref == ref
            and edge.src_ref.startswith("cell_")
        )

    def _query_debt_blocking(self, edges: list[EvidenceEdge], ref: str) -> list[dict[str, Any]]:
        return _unique_results(
            _result(edge.src_ref, edge)
            for edge in edges
            if edge.relation in {"fails_on", "confounded_by"}
            and edge.dst_ref == ref
            and edge.src_ref.startswith("debt_")
        )

    def _summary(self, edges: list[EvidenceEdge]) -> dict[str, Any]:
        relation_counts: dict[str, int] = {}
        nodes: set[str] = set()
        for edge in edges:
            relation_counts[edge.relation] = relation_counts.get(edge.relation, 0) + 1
            nodes.add(edge.src_ref)
            nodes.add(edge.dst_ref)
        return {
            "created_at": utc_now(),
            "counts": {
                "edges": len(edges),
                "nodes": len(nodes),
                "relations": relation_counts,
            },
        }


def make_edge(
    src_ref: str,
    dst_ref: str,
    relation: EvidenceRelation,
    *,
    source_ref: str | None = None,
    source_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvidenceEdge:
    payload = dict(metadata or {})
    source_path_text = str(source_path) if source_path is not None else None
    edge_ref = stable_ref(
        "edge",
        src_ref,
        dst_ref,
        relation,
        source_ref or "",
        source_path_text or "",
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")),
    )
    return EvidenceEdge(
        wb_ref=edge_ref,
        src_ref=str(src_ref),
        dst_ref=str(dst_ref),
        relation=relation,
        source_ref=source_ref,
        source_path=source_path_text,
        metadata=payload,
        parents=[str(src_ref), str(dst_ref)],
    )


def _claim_ref(card: dict[str, Any]) -> str | None:
    if not card:
        return None
    if card.get("claim_ref"):
        return str(card["claim_ref"])
    metadata = card.get("metadata")
    if isinstance(metadata, dict) and metadata.get("claim_ref"):
        return str(metadata["claim_ref"])
    return None


def _dedupe_edges(edges: list[EvidenceEdge]) -> list[EvidenceEdge]:
    deduped: dict[str, EvidenceEdge] = {}
    for edge in edges:
        if edge.src_ref and edge.dst_ref:
            deduped[edge.wb_ref] = edge
    return list(deduped.values())


def _is_claim_ref(ref: str, edge: EvidenceEdge) -> bool:
    return ref.startswith("claim_") or edge.metadata.get("dst_type") == "claim"


def _payload_ref(payload: dict[str, Any], *, fallback: str) -> str:
    return str(
        payload.get("wb_ref")
        or payload.get("run_ref")
        or payload.get("mechanism_card_ref")
        or payload.get("claim_ref")
        or fallback
    )


def _result(ref: str, edge: EvidenceEdge, *, path: list[str] | None = None) -> dict[str, Any]:
    return {
        "ref": ref,
        "edge_ref": edge.wb_ref,
        "relation": edge.relation,
        "src_ref": edge.src_ref,
        "dst_ref": edge.dst_ref,
        "source_ref": edge.source_ref,
        "source_path": edge.source_path,
        "metadata": edge.metadata,
        **({"path": path} if path is not None else {}),
    }


def _unique_results(rows: Any) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(str(row["ref"]), row)
    return list(unique.values())


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
