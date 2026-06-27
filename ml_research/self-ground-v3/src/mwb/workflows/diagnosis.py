from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from mwb.domain.objects import DiagnosisTree, MaterializedProbe, NextProbePlan
from mwb.project import Project
from mwb.refs import stable_ref
from mwb.sqlite_index import initialize_schema, insert_payload
from mwb.workflows.next_probe import build_next_probe, load_next_probe_payload
from mwb.workflows.sweep import parse_axes, write_sweep_run

IMPLEMENTED_PROBE_KINDS = {"sweep_axis_extension", "switch_patch_mode"}


class DiagnosisService:
    def __init__(self, project: Project | None = None) -> None:
        self.project = project

    def diagnose_run_dir(self, run_dir: Path) -> DiagnosisTree:
        payload = load_next_probe_payload(run_dir)
        payload["scientific_debt"] = _read_json(run_dir / "scientific_debt.json")
        payload["_scientific_debt_source_refs"] = _scientific_debt_source_refs(run_dir, payload)
        return self.diagnose_payload(payload)

    def diagnose_payload(self, payload: dict[str, Any]) -> DiagnosisTree:
        run_ref = str(payload.get("run_ref", "unknown"))
        status = str(payload.get("status", "insufficient_evidence"))
        blockers = [str(blocker) for blocker in payload.get("blockers", [])]
        if not blockers:
            blockers = [build_next_probe(payload).diagnosis["primary"]]
        primary = str(
            payload.get("primary_blocker") or build_next_probe(payload).diagnosis["primary"]
        )
        if primary not in blockers:
            blockers = [primary, *blockers]

        source_refs = list(payload.get("_source_refs", []))
        source_refs.extend(payload.get("_scientific_debt_source_refs", []))
        nodes = _diagnosis_nodes(run_ref, primary, blockers, payload)
        scientific_debt = _scientific_debt_items(payload)
        negative_evidence = _negative_evidence(payload, scientific_debt)
        return DiagnosisTree(
            wb_ref=stable_ref("diag", run_ref, status, primary, blockers, negative_evidence),
            source_run_ref=run_ref,
            status=status,
            primary_blocker=primary,
            nodes=nodes,
            source_refs=_dedupe_refs(source_refs),
            negative_evidence=negative_evidence,
            scientific_debt=scientific_debt,
            parents=[run_ref],
        )

    def write_diagnosis(self, run_dir: Path, tree: DiagnosisTree) -> DiagnosisTree:
        payload = tree.model_dump(mode="json")
        _write_json(run_dir / "diagnosis_tree.json", payload)
        _write_yaml(run_dir / "diagnosis_tree.yaml", payload)
        (run_dir / "diagnosis_tree.md").write_text(
            _render_diagnosis_markdown(tree),
            encoding="utf-8",
        )
        if self.project is not None:
            initialize_schema(self.project.sqlite_path)
            insert_payload(self.project.sqlite_path, "diagnosis_trees", tree.wb_ref, payload)
        return tree

    def materialize_probe(
        self,
        run_dir: Path,
        *,
        plan: NextProbePlan | None = None,
        tree: DiagnosisTree | None = None,
    ) -> MaterializedProbe:
        payload = load_next_probe_payload(run_dir)
        plan = plan or build_next_probe(payload)
        if not (run_dir / "next_probe.json").exists():
            from mwb.workflows.next_probe import write_next_probe

            write_next_probe(run_dir, plan)
        tree = tree or self.write_diagnosis(run_dir, self.diagnose_run_dir(run_dir))
        if not (run_dir / "diagnosis_tree.json").exists():
            self.write_diagnosis(run_dir, tree)
        probe = ProbeRegistry().materialize(plan, tree, payload)
        self.write_probe(run_dir, probe)
        return probe

    def write_probe(self, run_dir: Path, probe: MaterializedProbe) -> MaterializedProbe:
        payload = probe.model_dump(mode="json")
        _write_json(run_dir / "probe.json", payload)
        _write_yaml(run_dir / "probe.yaml", payload)
        (run_dir / "probe.md").write_text(_render_probe_markdown(probe), encoding="utf-8")
        if self.project is not None:
            initialize_schema(self.project.sqlite_path)
            insert_payload(self.project.sqlite_path, "materialized_probes", probe.wb_ref, payload)
        return probe

    def run_probe(self, probe_path: Path) -> dict[str, Any]:
        probe = load_materialized_probe(probe_path)
        if not probe.runnable or probe.probe_kind not in IMPLEMENTED_PROBE_KINDS:
            raise ValueError(f"unsupported probe kind: {probe.probe_kind}")
        if self.project is None:
            raise ValueError("run-probe requires a project")

        axis = probe.parameters.get("axis")
        target_ref = str(probe.parameters.get("target_ref") or probe.source_run_ref)
        if not isinstance(axis, dict) or not axis:
            raise ValueError(f"probe {probe.wb_ref} is missing axis parameters")
        axis_args = [f"{name}={value}" for name, value in axis.items()]
        config = parse_axes(axis_args)
        config["axis_source"] = "materialized_probe"
        config["source_probe_ref"] = probe.wb_ref
        run_dir, output = write_sweep_run(
            project=self.project,
            hypothesis_payload={"wb_ref": target_ref},
            config=config,
            dry_run=True,
        )
        report = {
            "status": output["status"],
            "probe_ref": probe.wb_ref,
            "probe_kind": probe.probe_kind,
            "run_ref": output["run_ref"],
            "run_dir": str(run_dir),
            "claim_bearing": False,
        }
        insert_payload(
            self.project.sqlite_path,
            "materialized_probes",
            probe.wb_ref,
            probe.model_dump(mode="json"),
        )
        return report


class ProbeRegistry:
    def materialize(
        self,
        plan: NextProbePlan,
        tree: DiagnosisTree,
        payload: dict[str, Any],
    ) -> MaterializedProbe:
        kind = str(plan.recommendation.get("kind", "recommendation_without_command"))
        if kind == "smallest_axis_extension":
            return self._materialize_axis_extension(plan, tree, payload)
        if kind == "switch_patch_mode":
            return self._materialize_patch_mode(plan, tree, payload)
        return self._unsupported(plan, tree, payload, kind)

    def _materialize_axis_extension(
        self,
        plan: NextProbePlan,
        tree: DiagnosisTree,
        payload: dict[str, Any],
    ) -> MaterializedProbe:
        layer = _first_untried("layers", payload)
        if layer is None:
            return self._unsupported(plan, tree, payload, "smallest_axis_extension")
        target_ref = _target_ref(payload, plan.source_run_ref)
        parameters = {"axis": {"layer": layer}, "target_ref": target_ref, "dry_run": True}
        command = ["uv", "run", "mwb", "sweep", target_ref, "--axis", f"layer={layer}", "--dry-run"]
        return _probe(
            plan=plan,
            tree=tree,
            payload=payload,
            template_id="sweep_axis_extension.v1",
            probe_kind="sweep_axis_extension",
            status="ready",
            runnable=True,
            command=command,
            parameters=parameters,
            extra_provenance=_axis_provenance(payload, "available_axes.layers", layer),
        )

    def _materialize_patch_mode(
        self,
        plan: NextProbePlan,
        tree: DiagnosisTree,
        payload: dict[str, Any],
    ) -> MaterializedProbe:
        patch_mode = _first_untried("patch_modes", payload, preferred="direct")
        if patch_mode is None:
            return self._unsupported(plan, tree, payload, "switch_patch_mode")
        target_ref = _target_ref(payload, plan.source_run_ref)
        parameters = {"axis": {"patch_mode": patch_mode}, "target_ref": target_ref, "dry_run": True}
        command = [
            "uv",
            "run",
            "mwb",
            "sweep",
            target_ref,
            "--axis",
            f"patch_mode={patch_mode}",
            "--dry-run",
        ]
        return _probe(
            plan=plan,
            tree=tree,
            payload=payload,
            template_id="switch_patch_mode.v1",
            probe_kind="switch_patch_mode",
            status="ready",
            runnable=True,
            command=command,
            parameters=parameters,
            extra_provenance=_axis_provenance(payload, "available_axes.patch_modes", patch_mode),
        )

    def _unsupported(
        self,
        plan: NextProbePlan,
        tree: DiagnosisTree,
        payload: dict[str, Any],
        kind: str,
    ) -> MaterializedProbe:
        return _probe(
            plan=plan,
            tree=tree,
            payload=payload,
            template_id=f"unsupported.{kind}.v1",
            probe_kind=kind,
            status="blocked",
            runnable=False,
            command=[],
            parameters={"reason": "No implemented probe runner exists for this recommendation."},
            extra_provenance=[],
        )


def load_materialized_probe(path: Path) -> MaterializedProbe:
    if path.suffix in {".yaml", ".yml"}:
        payload = YAML().load(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid probe payload: {path}")
    return MaterializedProbe.model_validate(payload)


def _diagnosis_nodes(
    run_ref: str,
    primary: str,
    blockers: list[str],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for index, blocker in enumerate(blockers):
        nodes.append(
            {
                "node_ref": stable_ref("diag_node", run_ref, blocker, index),
                "kind": "blocker",
                "blocker": blocker,
                "priority": index,
                "status": "primary" if blocker == primary else "secondary",
                "source_refs": _matching_refs(payload, f"$.blockers[{index}]"),
            }
        )
    for index, metric in enumerate(payload.get("blocking_metrics", [])):
        if not isinstance(metric, dict):
            continue
        nodes.append(
            {
                "node_ref": stable_ref("diag_node", run_ref, "metric", metric, index),
                "kind": "blocking_metric",
                "blocker": primary,
                "metric": metric,
                "status": str(metric.get("status", "reported")),
                "source_refs": _matching_refs(payload, f"$.blocking_metrics[{index}]"),
            }
        )
    return nodes


def _negative_evidence(
    payload: dict[str, Any],
    scientific_debt: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in scientific_debt:
        if item.get("status") == "unresolved":
            evidence.append(
                {
                    "kind": "scientific_debt",
                    "debt_ref": item.get("debt_ref"),
                    "blocker": item.get("blocker"),
                    "status": item.get("status"),
                    "reason": item.get("reason"),
                }
            )
    for metric in payload.get("blocking_metrics", []):
        if isinstance(metric, dict) and str(metric.get("status", "")).startswith("fail"):
            evidence.append(
                {
                    "kind": "blocking_metric",
                    "blocker": payload.get("primary_blocker"),
                    "metric": metric.get("name"),
                    "status": metric.get("status"),
                    "value": metric.get("value"),
                    "threshold": metric.get("threshold"),
                }
            )
    return evidence


def _scientific_debt_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    debt = payload.get("scientific_debt", {})
    if not isinstance(debt, dict):
        return []
    items = debt.get("items", [])
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _scientific_debt_source_refs(run_dir: Path, payload: dict[str, Any]) -> list[dict[str, str]]:
    debt = payload.get("scientific_debt", {})
    items = debt.get("items", []) if isinstance(debt, dict) else []
    refs = []
    if not isinstance(items, list):
        return refs
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        for key in sorted(item):
            refs.append(
                {
                    "artifact": "scientific_debt.json",
                    "path": str(run_dir / "scientific_debt.json"),
                    "jsonpath": f"$.items[{index}].{key}",
                    "field": f"scientific_debt.{key}",
                }
            )
    return refs


def _first_untried(
    axis_name: str,
    payload: dict[str, Any],
    *,
    preferred: str | None = None,
) -> str | None:
    tried = {str(value) for value in payload.get("tried_axes", {}).get(axis_name, [])}
    available = [str(value) for value in payload.get("available_axes", {}).get(axis_name, [])]
    untried = [value for value in available if value not in tried]
    if preferred is not None and preferred in untried:
        return preferred
    return untried[0] if untried else None


def _target_ref(payload: dict[str, Any], fallback: str) -> str:
    return str(payload.get("source_hypothesis_ref") or fallback)


def _axis_provenance(
    payload: dict[str, Any],
    field: str,
    value: str,
) -> list[dict[str, str]]:
    matching = [
        ref
        for ref in payload.get("_source_refs", [])
        if ref.get("field") == field
        and _jsonpath_index_value(payload, ref.get("jsonpath")) == value
    ]
    if matching:
        return matching
    return [
        {
            "artifact": "run_manifest.json",
            "path": "",
            "jsonpath": f"$.{field}",
            "field": field,
        }
    ]


def _jsonpath_index_value(payload: dict[str, Any], jsonpath: str | None) -> str | None:
    if not jsonpath or not jsonpath.startswith("$."):
        return None
    parts = jsonpath[2:].split(".")
    current: Any = payload
    for part in parts:
        if "[" in part and part.endswith("]"):
            name, index_text = part[:-1].split("[", 1)
            current = current.get(name, []) if isinstance(current, dict) else []
            try:
                current = current[int(index_text)]
            except (IndexError, TypeError, ValueError):
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return str(current) if current is not None else None


def _probe(
    *,
    plan: NextProbePlan,
    tree: DiagnosisTree,
    payload: dict[str, Any],
    template_id: str,
    probe_kind: str,
    status: str,
    runnable: bool,
    command: list[str],
    parameters: dict[str, Any],
    extra_provenance: list[dict[str, str]],
) -> MaterializedProbe:
    provenance = _dedupe_refs(
        [*tree.source_refs, *payload.get("_source_refs", []), *extra_provenance]
    )
    return MaterializedProbe(
        wb_ref=stable_ref("probe", plan.wb_ref, tree.wb_ref, template_id, command, parameters),
        source_run_ref=plan.source_run_ref,
        next_probe_ref=plan.wb_ref,
        diagnosis_tree_ref=tree.wb_ref,
        template_id=template_id,
        probe_kind=probe_kind,
        status=status,
        runnable=runnable,
        command=command,
        parameters=parameters,
        provenance=provenance,
        parents=[plan.wb_ref, tree.wb_ref, plan.source_run_ref],
    )


def _matching_refs(payload: dict[str, Any], jsonpath: str) -> list[dict[str, str]]:
    return [ref for ref in payload.get("_source_refs", []) if ref.get("jsonpath") == jsonpath]


def _dedupe_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for ref in refs:
        artifact = str(ref.get("artifact", ""))
        jsonpath = str(ref.get("jsonpath", ""))
        field = str(ref.get("field", ""))
        key = (artifact, jsonpath, field)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "artifact": artifact,
                "path": str(ref.get("path", "")),
                "jsonpath": jsonpath,
                "field": field,
            }
        )
    return deduped


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    yaml = YAML()
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(payload, handle)


def _render_diagnosis_markdown(tree: DiagnosisTree) -> str:
    return "\n".join(
        [
            f"# Diagnosis Tree: {tree.source_run_ref}",
            "",
            f"Status: {tree.status}",
            f"Primary blocker: `{tree.primary_blocker}`",
            "",
            "Nodes:",
            *[
                f"- {node['kind']}: {node.get('blocker', node.get('status'))}"
                for node in tree.nodes
            ],
            "",
            "Negative evidence:",
            *[
                f"- {item['kind']}: {item.get('blocker') or item.get('metric')}"
                for item in tree.negative_evidence
            ],
            "",
        ]
    )


def _render_probe_markdown(probe: MaterializedProbe) -> str:
    command = " ".join(probe.command) if probe.command else "unsupported"
    return "\n".join(
        [
            f"# Materialized Probe: {probe.source_run_ref}",
            "",
            f"Template: `{probe.template_id}`",
            f"Kind: `{probe.probe_kind}`",
            f"Status: {probe.status}",
            f"Runnable: {str(probe.runnable).lower()}",
            f"Command: `{command}`",
            "",
        ]
    )
