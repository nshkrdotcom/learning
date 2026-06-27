from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mwb.domain.objects import ClaimGrammarReport
from mwb.policy_profiles import profile_for_name
from mwb.project import Project
from mwb.refs import stable_ref
from mwb.sqlite_index import initialize_schema, insert_payload

JsonDict = dict[str, Any]
CLAIM_ORDER = [
    "association",
    "projection",
    "causal_necessity",
    "causal_sufficiency",
    "mediation",
    "generalization",
    "mechanism",
]
CLAIM_VERBS = {
    "association": [
        "associated with",
        "correlated with",
        "enriched for",
        "responds to",
        "is a candidate marker for",
    ],
    "projection": ["projects toward", "is aligned with", "is structurally compatible with"],
    "causal_necessity": [
        "is causally implicated in",
        "contributes causally to",
        "is necessary under this intervention",
    ],
    "causal_sufficiency": ["can induce", "can restore", "is sufficient under this setup"],
    "mediation": ["mediates", "partially mediates", "lies on a causal path for"],
    "generalization": ["generalizes", "holds across"],
    "mechanism": ["implements", "is the mechanism for", "realizes"],
}
CLAIM_REQUIREMENTS = {
    "association": ["association"],
    "projection": ["association", "static_projection_or_path_algebra"],
    "causal_necessity": [
        "association",
        "static_projection_or_path_algebra",
        "causal_necessity",
        "specificity_controls",
        "telemetry_clean",
    ],
    "causal_sufficiency": [
        "association",
        "static_projection_or_path_algebra",
        "causal_necessity",
        "causal_sufficiency",
        "specificity_controls",
        "telemetry_clean",
    ],
    "mediation": [
        "association",
        "static_projection_or_path_algebra",
        "causal_necessity",
        "specificity_controls",
        "telemetry_clean",
        "mediation",
    ],
    "generalization": [
        "association",
        "static_projection_or_path_algebra",
        "causal_necessity",
        "specificity_controls",
        "telemetry_clean",
        "generalization_minimum",
    ],
    "mechanism": [
        "association",
        "static_projection_or_path_algebra",
        "causal_necessity",
        "causal_sufficiency",
        "specificity_controls",
        "telemetry_clean",
        "alternative_explanations_resolved",
        "generalization_minimum",
    ],
}
TIER_EVIDENCE = {
    "association": ["association"],
    "projection": ["association", "static_projection_or_path_algebra"],
    "causal_necessity": CLAIM_REQUIREMENTS["causal_necessity"],
    "causal_sufficiency": CLAIM_REQUIREMENTS["causal_sufficiency"],
    "mediation": [*CLAIM_REQUIREMENTS["causal_necessity"], "mediation"],
    "generalization": [*CLAIM_REQUIREMENTS["causal_necessity"], "generalization_minimum"],
    "mechanism": CLAIM_REQUIREMENTS["mechanism"],
}
BLOCKED_BY = {
    "metadata_mismatch": [
        "projection",
        "causal_necessity",
        "causal_sufficiency",
        "mediation",
        "generalization",
        "mechanism",
    ],
    "backend_untrusted": ["causal_necessity", "causal_sufficiency", "mediation", "mechanism"],
    "control_leaky": ["generalization", "mechanism"],
    "off_manifold_intervention": ["causal_necessity", "causal_sufficiency", "mechanism"],
    "dictionary_interference": ["mechanism"],
    "self_repair_suspected": ["mediation", "mechanism"],
    "insufficient_heldout_generalization": ["generalization", "mechanism"],
    "policy_generalization_required": ["mechanism"],
}
CAVEATED_BY = {
    "control_leaky": ["causal_necessity", "causal_sufficiency", "mediation"],
    "dictionary_interference": ["projection", "causal_necessity"],
    "self_repair_suspected": ["causal_necessity", "causal_sufficiency"],
    "insufficient_heldout_generalization": [
        "causal_necessity",
        "causal_sufficiency",
        "mediation",
    ],
}
GENERALIZATION_CAVEAT = (
    "Do not generalize beyond the tested model, layer, dictionary, and prompt bundle."
)


class ClaimGrammarService:
    def __init__(self, project: Project | None = None) -> None:
        self.project = project

    def check_file(self, path: Path) -> ClaimGrammarReport:
        return self.check_claim(json.loads(path.read_text(encoding="utf-8")))

    def check_claim(self, payload: JsonDict) -> ClaimGrammarReport:
        text = str(payload["text"])
        card = dict(payload.get("mechanism_card") or payload)
        claim_ref = str(payload.get("claim_ref") or card.get("claim_ref") or "claim_unknown")
        claim_type = str(payload.get("claim_type") or infer_claim_type(text))
        policy_profile = profile_for_name(
            str(payload.get("policy_profile") or card.get("policy_profile") or "strict")
        )
        evidence_tier = str(card.get("evidence_tier") or card.get("status") or "association")
        if "evidence" in card:
            evidence = {str(item) for item in card.get("evidence", [])}
        else:
            evidence = set(TIER_EVIDENCE.get(evidence_tier, []))
        requirements = _requirements_for_profile(claim_type, policy_profile.name)
        missing = [requirement for requirement in requirements if requirement not in evidence]
        blockers = _card_blockers(card)
        if (
            claim_type == "mechanism"
            and policy_profile.require_generalization_for_mechanism_word
            and "generalization_minimum" not in evidence
            and "policy_generalization_required" not in blockers
        ):
            blockers.append("policy_generalization_required")
        blocking_blockers = [
            blocker for blocker in blockers if claim_type in BLOCKED_BY.get(blocker, [])
        ]
        caveat_blockers = [
            blocker for blocker in blockers if claim_type in CAVEATED_BY.get(blocker, [])
        ]
        debt = _scientific_debt(card)
        blocking_debt = [
            item for item in debt if _debt_blocks_claim(item, claim_type)
        ]
        required_caveats = _required_caveats(claim_type, caveat_blockers, debt)

        if missing or blocking_blockers or blocking_debt:
            status = "blocked"
        elif required_caveats:
            status = "caveated"
        else:
            status = "allowed"

        supported_claim_type = supported_claim_type_for_evidence(evidence, blockers, debt)
        if policy_profile.require_generalization_for_mechanism_word:
            supported_claim_type = _cap_mechanism_without_generalization(
                supported_claim_type,
                evidence,
            )
        override = _override(payload.get("override"), status)
        report = ClaimGrammarReport(
            wb_ref=stable_ref(
                "claimgrammar",
                claim_ref,
                text,
                claim_type,
                evidence_tier,
                status,
                missing,
                blockers,
                blocking_debt,
            ),
            claim_ref=claim_ref,
            claim_type=claim_type,
            status=status,
            requested_text=text,
            evidence_tier=evidence_tier,
            policy_profile=policy_profile.name,
            supported_claim_type=supported_claim_type,
            missing_requirements=missing,
            blockers=blocking_blockers or blockers,
            blocking_debt=blocking_debt,
            required_caveats=required_caveats,
            allowed_verbs=CLAIM_VERBS[claim_type],
            blocked_verbs=CLAIM_VERBS[claim_type] if status == "blocked" else [],
            suggested_replacements=_suggested_replacements(supported_claim_type),
            override=override,
            parents=[str(card.get("wb_ref") or card.get("mechanism_card_ref") or claim_ref)],
        )
        return report

    def write_report(self, report: ClaimGrammarReport) -> Path:
        if self.project is None:
            raise ValueError("ClaimGrammarService.write_report requires a project")
        output_dir = self.project.mechanism_dir / "claims"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{report.claim_ref}_grammar_report.json"
        payload = report.model_dump(mode="json")
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        initialize_schema(self.project.sqlite_path)
        insert_payload(self.project.sqlite_path, "claim_grammar_reports", report.wb_ref, payload)
        return output


def infer_claim_type(text: str) -> str:
    lowered = text.lower()
    ordered = [
        ("mechanism", ["implements", "mechanism for", "realizes"]),
        ("mediation", ["mediates", "causal path"]),
        ("generalization", ["generalizes", "held-out", "holds across", "across held"]),
        ("causal_sufficiency", ["sufficient", "restore", "induce"]),
        ("causal_necessity", ["causally implicated", "contributes causally", "necessary"]),
        ("projection", ["projects", "aligned with", "structurally compatible"]),
        ("association", ["associated with", "correlated", "candidate marker", "responds to"]),
    ]
    for claim_type, markers in ordered:
        if any(marker in lowered for marker in markers):
            return claim_type
    return "association"


def supported_claim_type_for_evidence(
    evidence: set[str],
    blockers: list[str],
    debt: list[JsonDict],
) -> str:
    supported = "association"
    for claim_type in CLAIM_ORDER:
        requirements = CLAIM_REQUIREMENTS[claim_type]
        if all(requirement in evidence for requirement in requirements):
            supported = claim_type
    for claim_type in reversed(CLAIM_ORDER):
        if _claim_order(claim_type) <= _claim_order(supported):
            if not any(claim_type in BLOCKED_BY.get(blocker, []) for blocker in blockers):
                if not any(_debt_blocks_claim(item, claim_type) for item in debt):
                    return claim_type
    return "association"


def _requirements_for_profile(claim_type: str, profile_name: str) -> list[str]:
    requirements = list(CLAIM_REQUIREMENTS[claim_type])
    if profile_name == "exploratory" and claim_type == "mechanism":
        requirements = [
            requirement
            for requirement in requirements
            if requirement != "generalization_minimum"
        ]
    return requirements


def _cap_mechanism_without_generalization(
    supported_claim_type: str,
    evidence: set[str],
) -> str:
    if supported_claim_type != "mechanism" or "generalization_minimum" in evidence:
        return supported_claim_type
    return "causal_sufficiency" if "causal_sufficiency" in evidence else "causal_necessity"


def _card_blockers(card: JsonDict) -> list[str]:
    blockers = card.get("blockers") or card.get("metadata", {}).get("blockers", [])
    return [str(blocker) for blocker in blockers]


def _scientific_debt(card: JsonDict) -> list[JsonDict]:
    debt = card.get("scientific_debt") or card.get("metadata", {}).get("scientific_debt", [])
    if not isinstance(debt, list):
        return []
    return [item for item in debt if isinstance(item, dict) and item.get("status") == "unresolved"]


def _debt_blocks_claim(item: JsonDict, claim_type: str) -> bool:
    upgrade_to = str(item.get("required_to_upgrade_to") or "")
    if upgrade_to not in CLAIM_ORDER:
        return False
    return _claim_order(claim_type) >= _claim_order(upgrade_to)


def _required_caveats(
    claim_type: str,
    caveat_blockers: list[str],
    debt: list[JsonDict],
) -> list[str]:
    caveats: list[str] = []
    if caveat_blockers:
        caveats.append(GENERALIZATION_CAVEAT)
    for item in debt:
        upgrade_to = str(item.get("required_to_upgrade_to") or "")
        if upgrade_to == "generalization" and _claim_order(claim_type) < _claim_order(upgrade_to):
            caveats.append(GENERALIZATION_CAVEAT)
    return _dedupe(caveats)


def _override(raw: Any, status: str) -> JsonDict:
    if not isinstance(raw, dict):
        return {}
    return {
        "visible": raw.get("visibility") == "inline",
        "applied": False,
        "reason": raw.get("reason"),
        "requested_by": raw.get("requested_by"),
        "message": f"Overrides cannot upgrade {status} claim grammar results.",
    }


def _suggested_replacements(supported_claim_type: str) -> list[str]:
    if supported_claim_type == "association":
        return ["is associated with", "is a candidate marker for"]
    if supported_claim_type == "projection":
        return ["projects toward", "is structurally compatible with"]
    if supported_claim_type == "causal_necessity":
        return ["contributes causally to", "is causally implicated in"]
    return CLAIM_VERBS[supported_claim_type]


def _claim_order(claim_type: str) -> int:
    try:
        return CLAIM_ORDER.index(claim_type)
    except ValueError:
        return 0


def _dedupe(values: list[str]) -> list[str]:
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped
