from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from self_ground.behavioral_tasks import (
    TASK_FAMILY_ORDER,
    BehavioralTask,
    write_behavioral_tasks_jsonl,
)
from self_ground.io import write_config, write_jsonl
from self_ground.logit_scoring import token_id_for_single_token_string


class CandidateTaskTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_id: str
    family: str
    prompt_template: str
    target_token: str
    foil_token: str
    control_prompt_template: str | None = None
    metadata: dict[str, Any] = {}


class CandidateTaskBank(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "self_ground.task_bank.v1"
    model_name: str
    families: list[str]
    templates: list[CandidateTaskTemplate]
    metadata: dict[str, Any] = {}


def stable_task_id(*parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return digest[:12]


def candidate_to_behavioral_task(template: CandidateTaskTemplate) -> BehavioralTask:
    control = template.control_prompt_template
    if control is None:
        control = template.prompt_template.replace(" not ", " ")
    task_id = f"{template.family}_{stable_task_id(template.template_id, template.prompt_template)}"
    return BehavioralTask(
        id=task_id,
        family=template.family,
        concept=str(template.metadata.get("concept", template.template_id)),
        prompt=template.prompt_template,
        target_tokens=[template.target_token],
        foil_tokens=[template.foil_token],
        control_prompt=control,
        control_target_tokens=[template.foil_token],
        control_foil_tokens=[template.target_token],
        expected_baseline_direction="positive",
        metadata={
            **template.metadata,
            "template_id": template.template_id,
            "source": "phase3_task_bank",
        },
    )


def task_bank_to_behavioral_tasks(bank: CandidateTaskBank) -> list[BehavioralTask]:
    return [candidate_to_behavioral_task(template) for template in bank.templates]


def _template_id(family: str, subject: str, foil: str, target: str, idx: int) -> str:
    raw = f"{family}|{subject}|{foil}|{target}|{idx}"
    return f"{family}_{stable_task_id(raw)}"


def _copula_templates(
    *,
    family: str,
    subjects: list[str],
    pairs: list[tuple[str, str]],
    per_family_candidates: int,
) -> list[CandidateTaskTemplate]:
    templates: list[CandidateTaskTemplate] = []
    idx = 0
    for subject in subjects:
        for foil, target in pairs:
            if len(templates) >= per_family_candidates:
                return templates
            prompt_subject = f"The {subject}"
            templates.append(
                CandidateTaskTemplate(
                    template_id=_template_id(family, subject, foil, target, idx),
                    family=family,
                    prompt_template=f"{prompt_subject} was not {foil}. {prompt_subject} was",
                    target_token=f" {target}",
                    foil_token=f" {foil}",
                    control_prompt_template=f"{prompt_subject} was {foil}. {prompt_subject} was",
                    metadata={
                        "subject": subject,
                        "foil_property": foil,
                        "target_property": target,
                        "template_family": "copula_antonym",
                        "template_index": idx,
                    },
                )
            )
            idx += 1
    return templates


def _sentiment_templates(
    *,
    subjects: list[str],
    pairs: list[tuple[str, str]],
    per_family_candidates: int,
) -> list[CandidateTaskTemplate]:
    templates: list[CandidateTaskTemplate] = []
    idx = 0
    # Sentiment calibration for Pythia-70M-deduped is much more reliable when
    # high-yield lexical contrasts (especially positive->negative) are spread
    # across many subjects rather than exhausting many adjectives for the first
    # few subjects. This is still deterministic template generation, not
    # intervention-outcome filtering.
    for foil, target in pairs:
        for subject in subjects:
            if len(templates) >= per_family_candidates:
                return templates
            prompt_subject = f"The {subject}"
            templates.append(
                CandidateTaskTemplate(
                    template_id=_template_id(
                        "sentiment_negation",
                        subject,
                        foil,
                        target,
                        idx,
                    ),
                    family="sentiment_negation",
                    prompt_template=f"{prompt_subject} was not {foil}. {prompt_subject} was",
                    target_token=f" {target}",
                    foil_token=f" {foil}",
                    control_prompt_template=f"{prompt_subject} was {foil}. {prompt_subject} was",
                    metadata={
                        "subject": subject,
                        "foil_property": foil,
                        "target_property": target,
                        "template_family": "sentiment_copula_antonym",
                        "template_index": idx,
                    },
                )
            )
            idx += 1
    return templates


def _state_templates(per_family_candidates: int) -> list[CandidateTaskTemplate]:
    subjects = [
        "machine",
        "engine",
        "device",
        "signal",
        "package",
        "letter",
        "patient",
        "plant",
        "door",
        "window",
        "alarm",
        "connection",
        "battery",
        "computer",
        "system",
        "printer",
        "screen",
        "train",
        "flight",
        "meeting",
    ]
    event_pairs = [
        ("start", "started", "stopped", "Instead, it"),
        ("continue", "continued", "stopped", "Instead, it"),
        ("arrive", "arrived", "missing", "It was"),
        ("open", "opened", "closed", "It was"),
        ("close", "closed", "open", "It was"),
        ("charge", "charged", "dead", "It was"),
        ("connect", "connected", "offline", "It was"),
        ("respond", "responded", "silent", "It was"),
        ("recover", "recovered", "worse", "It was"),
        ("improve", "improved", "worse", "It was"),
        ("finish", "finished", "incomplete", "It was"),
        ("work", "worked", "broken", "It was"),
    ]
    templates: list[CandidateTaskTemplate] = []
    idx = 0
    for subject in subjects:
        for verb, foil, target, continuation in event_pairs:
            if len(templates) >= per_family_candidates:
                return templates
            prefix = f"The {subject}"
            templates.append(
                CandidateTaskTemplate(
                    template_id=_template_id("state_negation", subject, foil, target, idx),
                    family="state_negation",
                    prompt_template=f"{prefix} did not {verb}. {continuation}",
                    target_token=f" {target}",
                    foil_token=f" {foil}",
                    control_prompt_template=f"{prefix} {foil}. {continuation}",
                    metadata={
                        "subject": subject,
                        "event_verb": verb,
                        "foil_state": foil,
                        "target_state": target,
                        "template_family": "event_state_alternative",
                        "template_index": idx,
                    },
                )
            )
            idx += 1
    return templates


def generate_candidate_templates(per_family_candidates: int = 80) -> list[CandidateTaskTemplate]:
    if per_family_candidates < 1:
        raise ValueError("per_family_candidates must be >= 1")
    sentiment_subjects = [
        "movie",
        "service",
        "review",
        "experience",
        "meal",
        "hotel",
        "product",
        "app",
        "book",
        "show",
        "game",
        "flight",
        "support",
        "restaurant",
        "course",
        "event",
        "design",
        "plan",
        "answer",
        "result",
        "story",
        "episode",
        "concert",
        "visit",
        "purchase",
        "device",
        "website",
        "tool",
        "idea",
        "proposal",
    ]
    sentiment_pairs = [
        ("positive", "negative"),
        ("fast", "slow"),
        ("strong", "weak"),
        ("good", "bad"),
        ("reliable", "unreliable"),
        ("clear", "unclear"),
        ("helpful", "unhelpful"),
        ("pleasant", "unpleasant"),
        ("excellent", "poor"),
        ("nice", "awful"),
        ("useful", "useless"),
        ("enjoyable", "boring"),
    ]
    property_subjects = [
        "object",
        "box",
        "water",
        "surface",
        "fabric",
        "path",
        "room",
        "door",
        "window",
        "metal",
        "stone",
        "wood",
        "paper",
        "rope",
        "road",
        "screen",
        "light",
        "sound",
        "tool",
        "container",
    ]
    property_pairs = [
        ("heavy", "light"),
        ("open", "closed"),
        ("hot", "cold"),
        ("rough", "smooth"),
        ("full", "empty"),
        ("wet", "dry"),
        ("safe", "dangerous"),
        ("bright", "dark"),
        ("clean", "dirty"),
        ("sharp", "dull"),
        ("hard", "soft"),
        ("loud", "quiet"),
        ("wide", "narrow"),
        ("high", "low"),
        ("deep", "shallow"),
        ("tight", "loose"),
        ("long", "short"),
        ("thick", "thin"),
        ("fast", "slow"),
        ("strong", "weak"),
        ("new", "old"),
        ("fresh", "stale"),
    ]
    return [
        *_sentiment_templates(
            subjects=sentiment_subjects,
            pairs=sentiment_pairs,
            per_family_candidates=per_family_candidates,
        ),
        *_copula_templates(
            family="property_negation",
            subjects=property_subjects,
            pairs=property_pairs,
            per_family_candidates=per_family_candidates,
        ),
        *_state_templates(per_family_candidates),
    ]


def validate_candidate_template_tokens(
    *,
    model_adapter,
    template: CandidateTaskTemplate,
) -> tuple[bool, dict[str, Any] | None]:
    fields = {
        "target_token": template.target_token,
        "foil_token": template.foil_token,
    }
    for field, token in fields.items():
        try:
            token_id_for_single_token_string(model_adapter, token)
        except Exception as exc:
            return False, {
                "template_id": template.template_id,
                "family": template.family,
                "reason": "tokenization_failed",
                "field": field,
                "token": token,
                "exception_class": type(exc).__name__,
                "exception_message": str(exc),
            }
    return True, None


def build_candidate_task_bank(
    *,
    model_adapter,
    model_name: str,
    per_family_candidates: int = 80,
) -> tuple[CandidateTaskBank, list[BehavioralTask], list[dict[str, Any]]]:
    raw_templates = generate_candidate_templates(per_family_candidates=per_family_candidates)
    accepted: list[CandidateTaskTemplate] = []
    rejections: list[dict[str, Any]] = []
    for template in raw_templates:
        ok, rejection = validate_candidate_template_tokens(
            model_adapter=model_adapter,
            template=template,
        )
        if ok:
            accepted.append(template)
        elif rejection is not None:
            rejections.append(rejection)
    counts = Counter(template.family for template in accepted)
    bank = CandidateTaskBank(
        model_name=model_name,
        families=list(TASK_FAMILY_ORDER),
        templates=accepted,
        metadata={
            "per_family_candidates_requested": per_family_candidates,
            "accepted_by_family": {
                family: int(counts.get(family, 0)) for family in TASK_FAMILY_ORDER
            },
            "rejected_count": len(rejections),
            "tokenizer_model": model_name,
        },
    )
    return bank, task_bank_to_behavioral_tasks(bank), rejections


def write_task_bank_artifacts(
    *,
    bank: CandidateTaskBank,
    tasks: list[BehavioralTask],
    rejections: list[dict[str, Any]],
    out: str | Path,
) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_config(bank.model_dump(mode="json"), out_path)
    stem = out_path.with_suffix("")
    if stem.name.endswith("_candidate_bank"):
        prefix = stem.name[: -len("_candidate_bank")]
        tasks_path = stem.with_name(f"{prefix}_candidate_tasks.jsonl")
        rejections_path = stem.with_name(f"{prefix}_candidate_rejections.jsonl")
    else:
        tasks_path = stem.with_name(f"{stem.name}_tasks.jsonl")
        rejections_path = stem.with_name(f"{stem.name}_rejections.jsonl")
    write_behavioral_tasks_jsonl(tasks, tasks_path)
    write_jsonl(rejections, rejections_path)
    readme = f"""# Phase 3 Candidate Task Bank

- schema: `{bank.schema_version}`
- model tokenizer: `{bank.model_name}`
- families: `{bank.families}`
- accepted templates: `{len(bank.templates)}`
- rejected templates: `{len(rejections)}`
- accepted by family: `{bank.metadata.get("accepted_by_family")}`

Targets and foils were filtered through the active model tokenizer. Rejections
are written next to this file with explicit tokenization failure reasons.
"""
    (out_path.parent / "README.md").write_text(readme, encoding="utf-8")
