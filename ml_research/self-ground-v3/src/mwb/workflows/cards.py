from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mwb.domain.objects import MechanismCard
from mwb.refs import slugify
from mwb.workflows.blockers import diagnose_blockers


def card_from_run(run_dir: Path) -> MechanismCard:
    manifest = _read_json(run_dir / "run_manifest.json")
    metrics = _read_json(run_dir / "control_metrics.json")
    blocker_report = _read_json(run_dir / "blocker_report.json")
    run_ref = manifest.get("run_ref", run_dir.name)
    blockers = blocker_report.get("blockers")
    if not blockers:
        blockers = diagnose_blockers(metrics, thresholds={"control_leaky_ratio": 0.8})["blockers"]

    status = manifest.get("status", "insufficient_evidence")
    evidence_tier = _evidence_tier(status, _non_policy_blockers(blockers))
    policy_report = manifest.get("policy_report", {})
    if not policy_report and isinstance(manifest.get("metadata"), dict):
        policy_report = manifest.get("metadata", {}).get("policy_report", {})
    if isinstance(policy_report, dict):
        blockers = _dedupe([*blockers, *[str(item) for item in policy_report.get("blockers", [])]])
        evidence_tier = _apply_claim_ceiling(evidence_tier, policy_report.get("claim_ceiling"))
    allowed, blocked = language_for_tier(evidence_tier, blockers)
    claim_ref = f"claim_{slugify(str(run_ref))}"
    card = MechanismCard(
        wb_ref=f"mc_{slugify(str(run_ref))}",
        title=f"MechanismCard: {run_ref}",
        status=status,
        evidence_tier=evidence_tier,
        run_ref=run_ref,
        allowed_language=allowed,
        blocked_language=blocked,
        artifact_refs=[],
        metadata={
            "blockers": blockers,
            "claim_ref": claim_ref,
            "control_metrics": metrics,
            "policy_profile": (
                str(policy_report.get("policy_profile", "strict"))
                if isinstance(policy_report, dict)
                else "strict"
            ),
            "policy_report": policy_report if isinstance(policy_report, dict) else {},
            "scientific_debt": scientific_debt_items(run_ref, blockers, status),
        },
        parents=[run_ref],
    )
    return card


def write_card(run_dir: Path, card: MechanismCard, mechanism_dir: Path | None = None) -> None:
    payload = card_payload(card)
    (run_dir / "mechanism_card.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "mechanism_card.md").write_text(render_card_markdown(card), encoding="utf-8")
    (run_dir / "scientific_debt.json").write_text(
        json.dumps(scientific_debt_payload(card), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if mechanism_dir is not None:
        cards_dir = mechanism_dir / "cards"
        claims_dir = mechanism_dir / "claims"
        cards_dir.mkdir(parents=True, exist_ok=True)
        claims_dir.mkdir(parents=True, exist_ok=True)
        (cards_dir / f"{card.wb_ref}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        claim_ref = str(card.metadata["claim_ref"])
        (claims_dir / f"{claim_ref}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def card_payload(card: MechanismCard) -> dict[str, Any]:
    payload = card.model_dump(mode="json")
    payload["claim_ref"] = card.metadata["claim_ref"]
    payload["blockers"] = card.metadata.get("blockers", [])
    return payload


def render_card_markdown(card: MechanismCard) -> str:
    blockers = card.metadata.get("blockers", [])
    scientific_debt = card.metadata.get("scientific_debt", [])
    return "\n".join(
        [
            f"# {card.title}",
            "",
            f"Status: {card.status}",
            f"Evidence tier earned: {card.evidence_tier}",
            f"Claim ref: {card.metadata['claim_ref']}",
            "",
            "Allowed language:",
            *[f"- {item}" for item in card.allowed_language],
            "",
            "Blocked language:",
            *[f"- {item}" for item in card.blocked_language],
            "",
            "Blockers:",
            *[f"- {item}" for item in blockers],
            "",
            "Scientific debt:",
            *[f"- {item['kind']}: {item['description']}" for item in scientific_debt],
            "",
        ]
    )


def language_for_tier(evidence_tier: str, blockers: list[str]) -> tuple[list[str], list[str]]:
    if evidence_tier == "association":
        allowed = ["is associated with", "is correlated with", "is a candidate marker for"]
        blocked = ["causes", "is necessary for", "is sufficient for", "implements", "mechanism for"]
    elif evidence_tier == "projection":
        allowed = ["projects onto", "has decoder-level evidence for"]
        blocked = ["causes", "is necessary for", "is sufficient for", "implements", "mechanism for"]
    elif evidence_tier == "causal_necessity":
        allowed = ["is causally implicated in", "contributes causally to"]
        blocked = ["is sufficient for", "implements", "mechanism for"]
    elif evidence_tier == "causal_sufficiency":
        allowed = ["is sufficient under the tested intervention", "can drive the tested effect"]
        blocked = ["implements", "mechanism for", "fully explains"]
    elif evidence_tier == "mediation":
        allowed = ["mediates the tested effect", "partly mediates the tested behavior"]
        blocked = ["fully implements", "complete mechanism for"]
    elif evidence_tier == "generalization":
        allowed = ["generalizes across the tested settings", "is robust across tested settings"]
        blocked = ["fully implements", "complete mechanism for"]
    elif evidence_tier == "mechanism":
        allowed = [
            "implements",
            "is a mechanism for",
            "is necessary and sufficient within the tested scope",
        ]
        blocked = []
    else:
        allowed = ["is structurally compatible with"]
        blocked = ["implements", "mechanism for"]
    if "control_leaky" in blockers:
        for term in ["specificity", "implements", "mechanism for"]:
            if term not in blocked:
                blocked.append(term)
    return allowed, blocked


def _evidence_tier(status: str, blockers: list[str]) -> str:
    status_to_tier = {
        "projection_supported": "projection",
        "causal_necessity_supported": "causal_necessity",
        "causal_sufficiency_supported": "causal_sufficiency",
        "mediation_supported": "mediation",
        "generalization_supported": "generalization",
        "mechanism_supported": "mechanism",
    }
    if "control_leaky" in blockers and status_to_tier.get(status) != "mechanism_supported":
        return "association"
    if status in status_to_tier and not blockers:
        return status_to_tier[status]
    if (
        status in {"causal_necessity_supported", "candidate_evidence"}
        and "control_leaky" not in blockers
    ):
        return "causal_necessity"
    return "association"


def _apply_claim_ceiling(evidence_tier: str, ceiling: Any) -> str:
    if not ceiling or ceiling == "diagnostic_only":
        return "association" if ceiling == "diagnostic_only" else evidence_tier
    order = [
        "association",
        "projection",
        "causal_necessity",
        "causal_sufficiency",
        "mediation",
        "generalization",
        "mechanism",
    ]
    ceiling_text = str(ceiling)
    if evidence_tier not in order or ceiling_text not in order:
        return evidence_tier
    return order[min(order.index(evidence_tier), order.index(ceiling_text))]


def _non_policy_blockers(blockers: list[str]) -> list[str]:
    return [blocker for blocker in blockers if not blocker.startswith("policy_")]


def scientific_debt_payload(card: MechanismCard) -> dict[str, Any]:
    return {
        "run_ref": card.run_ref,
        "mechanism_card_ref": card.wb_ref,
        "status": card.status,
        "items": list(card.metadata.get("scientific_debt", [])),
    }


def scientific_debt_items(run_ref: str, blockers: list[str], status: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if "control_leaky" in blockers:
        items.append(
            {
                "debt_ref": f"debt_{slugify(str(run_ref))}_control_leaky",
                "kind": "controls",
                "blocker": "control_leaky",
                "description": (
                    "Controls moved too much to support specificity or mechanism wording."
                ),
                "required_resolution": (
                    "Refresh controls or rerun the smallest untried axis extension."
                ),
            }
        )
    if "insufficient_heldout_generalization" in blockers or status == "insufficient_evidence":
        items.append(
            {
                "debt_ref": f"debt_{slugify(str(run_ref))}_heldout_generalization",
                "kind": "heldout_generalization",
                "blocker": "insufficient_heldout_generalization",
                "description": "Heldout generalization is not established for stronger claims.",
                "required_resolution": "Run heldout tasks after controls pass.",
            }
        )
    if not any(item["kind"] == "causal_sufficiency" for item in items):
        items.append(
            {
                "debt_ref": f"debt_{slugify(str(run_ref))}_causal_sufficiency",
                "kind": "causal_sufficiency",
                "blocker": "insufficient_effect_size",
                "description": "Causal sufficiency has not been established.",
                "required_resolution": (
                    "Run a scoped sufficiency intervention with passing controls."
                ),
            }
        )
    return items


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped
