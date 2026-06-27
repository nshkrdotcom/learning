from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mwb.domain.objects import (
    AlternativeExplanation,
    HypothesisState,
    HypothesisTransitionReceipt,
    HypothesisWorkflowState,
)
from mwb.project import Project
from mwb.refs import stable_ref
from mwb.sqlite_index import initialize_schema, insert_payload
from mwb.time import utc_now
from mwb.workflows.runs import resolve_run_path

LINEAR_TRANSITIONS = {
    "noticed": {"triaged"},
    "triaged": {"structurally_plausible"},
    "structurally_plausible": {"cheap_proxy_supported"},
    "cheap_proxy_supported": {"exact_patch_supported"},
    "exact_patch_supported": {"control_clean"},
    "control_clean": {"generalized"},
    "generalized": {"claimable"},
}
TERMINAL_STATES = {
    "structurally_impossible",
    "proxy_false_positive",
    "control_leaky",
    "self_repair_confounded",
    "off_manifold",
    "task_artifact",
    "dictionary_artifact",
    "abandoned",
}


class HypothesisLifecycleService:
    def __init__(self, project: Project) -> None:
        self.project = project
        self.hypotheses_dir = project.mechanism_dir / "hypotheses"
        self.hypotheses_dir.mkdir(parents=True, exist_ok=True)

    def transition(
        self,
        hypothesis_ref: str,
        *,
        to_state: HypothesisWorkflowState,
        evidence_tier: str | None = None,
        claim_status: str | None = None,
        approved_by: str | None = None,
        decision_ref: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        current = self.load_state(hypothesis_ref)
        from_state = current.state if current else "noticed"
        self._validate_transition(
            from_state=from_state,
            to_state=to_state,
            approved_by=approved_by,
            decision_ref=decision_ref,
        )
        state = HypothesisState(
            wb_ref=hypothesis_ref,
            hypothesis_ref=hypothesis_ref,
            state=to_state,
            evidence_tier=evidence_tier or (current.evidence_tier if current else "none"),
            claim_status=claim_status if claim_status is not None else (
                current.claim_status if current else None
            ),
            approved_by=approved_by,
            decision_ref=decision_ref,
            parents=[hypothesis_ref],
        )
        receipt = HypothesisTransitionReceipt(
            wb_ref=stable_ref(
                "hyptrans",
                hypothesis_ref,
                from_state,
                to_state,
                evidence_tier or "",
                claim_status or "",
                approved_by or "",
                decision_ref or "",
                utc_now(),
            ),
            hypothesis_ref=hypothesis_ref,
            from_state=from_state,
            to_state=to_state,
            evidence_tier=state.evidence_tier,
            claim_status=state.claim_status,
            approved_by=approved_by,
            decision_ref=decision_ref,
            reason=reason,
            parents=[hypothesis_ref],
        )
        self._write_state(state)
        self._append_transition(hypothesis_ref, receipt)
        initialize_schema(self.project.sqlite_path)
        insert_payload(
            self.project.sqlite_path,
            "hypothesis_states",
            hypothesis_ref,
            state.model_dump(mode="json"),
        )
        insert_payload(
            self.project.sqlite_path,
            "hypothesis_transitions",
            receipt.wb_ref,
            receipt.model_dump(mode="json"),
        )
        return {
            "status": "ok",
            "state": state.model_dump(mode="json"),
            "receipt": receipt.model_dump(mode="json"),
        }

    def load_state(self, hypothesis_ref: str) -> HypothesisState | None:
        path = self._state_path(hypothesis_ref)
        if not path.exists():
            return None
        return HypothesisState.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def explain(self, run_ref_or_path: str) -> dict[str, Any]:
        run_dir = resolve_run_path(Path(run_ref_or_path), project=self.project)
        manifest = _read_json(run_dir / "run_manifest.json")
        if not manifest:
            raise FileNotFoundError(f"run manifest not found for {run_ref_or_path}")
        blocker_report = _read_json(run_dir / "blocker_report.json")
        metrics = _read_json(run_dir / "control_metrics.json")
        run_ref = str(manifest.get("run_ref") or run_dir.name)
        hypothesis_ref = str(manifest.get("source_hypothesis_ref") or "unknown_hypothesis")
        alternatives = [
            alternative.model_dump(mode="json")
            for alternative in alternatives_from_blockers(
                hypothesis_ref=hypothesis_ref,
                run_ref=run_ref,
                blocker_report=blocker_report,
                metrics=metrics,
            )
        ]
        payload = {
            "wb_ref": hypothesis_ref,
            "hypothesis_ref": hypothesis_ref,
            "run_ref": run_ref,
            "source_run_dir": str(run_dir),
            "created_at": utc_now(),
            "alternatives": alternatives,
        }
        (run_dir / "alternative_explanations.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        hyp_path = self.hypotheses_dir / f"{hypothesis_ref}_alternatives.json"
        hyp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        initialize_schema(self.project.sqlite_path)
        insert_payload(
            self.project.sqlite_path,
            "alternative_explanations",
            hypothesis_ref,
            payload,
        )
        return {"status": "ok", **payload}

    def _validate_transition(
        self,
        *,
        from_state: HypothesisWorkflowState,
        to_state: HypothesisWorkflowState,
        approved_by: str | None,
        decision_ref: str | None,
    ) -> None:
        if to_state == from_state:
            return
        if to_state == "claimable" and not (approved_by and decision_ref):
            raise ValueError("claimable promotion requires --approved-by and --decision-ref")
        allowed = set(LINEAR_TRANSITIONS.get(from_state, set())) | TERMINAL_STATES
        if to_state not in allowed:
            raise ValueError(f"invalid hypothesis transition: {from_state} -> {to_state}")

    def _state_path(self, hypothesis_ref: str) -> Path:
        return self.hypotheses_dir / f"{hypothesis_ref}_lifecycle.json"

    def _transition_path(self, hypothesis_ref: str) -> Path:
        return self.hypotheses_dir / f"{hypothesis_ref}_transitions.jsonl"

    def _write_state(self, state: HypothesisState) -> None:
        self._state_path(state.hypothesis_ref).write_text(
            json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _append_transition(
        self,
        hypothesis_ref: str,
        receipt: HypothesisTransitionReceipt,
    ) -> None:
        with self._transition_path(hypothesis_ref).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt.model_dump(mode="json"), sort_keys=True) + "\n")


def alternatives_from_blockers(
    *,
    hypothesis_ref: str,
    run_ref: str,
    blocker_report: dict[str, Any],
    metrics: dict[str, Any],
) -> list[AlternativeExplanation]:
    blockers = _string_list(blocker_report.get("blockers"))
    if not blockers:
        blockers = ["insufficient_effect_size"]
    alternatives: list[AlternativeExplanation] = []
    for blocker in blockers:
        alternatives.append(
            _alternative_for_blocker(
                hypothesis_ref=hypothesis_ref,
                run_ref=run_ref,
                blocker=blocker,
                blocker_report=blocker_report,
                metrics=metrics,
            )
        )
    return alternatives


def _alternative_for_blocker(
    *,
    hypothesis_ref: str,
    run_ref: str,
    blocker: str,
    blocker_report: dict[str, Any],
    metrics: dict[str, Any],
) -> AlternativeExplanation:
    config = _ALTERNATIVE_CONFIG.get(blocker, _ALTERNATIVE_CONFIG["insufficient_effect_size"])
    evidence_for = _evidence_for(blocker_report, metrics, blocker)
    return AlternativeExplanation(
        wb_ref=stable_ref("alt", hypothesis_ref, run_ref, config["id"], evidence_for),
        hypothesis_ref=hypothesis_ref,
        explanation_id=config["id"],
        source_ref=run_ref,
        blocker=blocker,
        evidence_for=evidence_for,
        evidence_against=[],
        next_test=config["next_test"],
        parents=[hypothesis_ref, run_ref],
    )


def _evidence_for(
    blocker_report: dict[str, Any],
    metrics: dict[str, Any],
    blocker: str,
) -> list[str]:
    evidence = []
    for item in blocker_report.get("blocking_metrics", []):
        if isinstance(item, dict):
            name = item.get("name", "metric")
            status = item.get("status", "unknown")
            value = item.get("value")
            threshold = item.get("threshold")
            evidence.append(f"{name} {status} value={value} threshold={threshold}")
    if not evidence and metrics:
        for key, value in sorted(metrics.items()):
            evidence.append(f"{key}={value}")
    if not evidence:
        evidence.append(f"blocker={blocker}")
    return evidence


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


_ALTERNATIVE_CONFIG = {
    "control_leaky": {
        "id": "control_leaky",
        "next_test": "Refresh controls or run a polarity-only bundle to separate target semantics.",
    },
    "density_matching_failed": {
        "id": "generic_high_activity_feature",
        "next_test": "Run per-example density-matched feature controls.",
    },
    "dictionary_interference": {
        "id": "decoder_neighbor_interference",
        "next_test": "Run decoder-neighbor cosine scan and neighbor ablation controls.",
    },
    "neighbor_feature_interference": {
        "id": "decoder_neighbor_interference",
        "next_test": "Run decoder-neighbor cosine scan and neighbor ablation controls.",
    },
    "self_repair_suspected": {
        "id": "downstream_self_repair",
        "next_test": "Run downstream backup scan after the intervention.",
    },
    "off_manifold_intervention": {
        "id": "off_manifold_intervention",
        "next_test": "Rerun with reduced intervention strength and KL/norm telemetry.",
    },
    "specificity_gap_failed": {
        "id": "task_artifact",
        "next_test": "Audit example geometry and construct stricter heldout controls.",
    },
    "artifact_incomplete": {
        "id": "artifact_incomplete",
        "next_test": "Complete missing artifacts before interpreting the result.",
    },
    "insufficient_effect_size": {
        "id": "insufficient_effect_size",
        "next_test": "Increase power or run a narrower exact verification probe.",
    },
}
