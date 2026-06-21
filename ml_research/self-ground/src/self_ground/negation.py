from __future__ import annotations

import hashlib
import itertools
import random
import re
from collections.abc import Iterable

from self_ground.data import MinimalPair
from self_ground.purity import score_control_purity

FAMILIES = ("copula", "do_support", "existential", "modal")
NEGATION_MARKERS = (" not ", "n't", " no ", " cannot ", " can't ")
DECOY_MARKERS = (" often ", " sometimes ")
_NEGATION_RE = re.compile(r"\b(?:not|no|never|cannot|can't|isn't|doesn't|don't|won't|n't)\b")

COPULA_ENTITIES = [
    "The dog",
    "The teacher",
    "The engine",
    "The bridge",
    "The soup",
    "The manager",
    "The laptop",
    "The singer",
    "The contract",
    "The bus",
]
COPULA_PROPERTIES = [
    "friendly",
    "reliable",
    "broken",
    "finished",
    "available",
    "expensive",
    "dangerous",
    "popular",
    "accurate",
    "stable",
]

DO_SUPPORT_ENTITIES = [
    "The committee",
    "The lawyer",
    "The student",
    "The vendor",
    "The pilot",
]
DO_SUPPORT_PAIRS = [
    ("approve", "the request"),
    ("trust", "the report"),
    ("support", "the proposal"),
    ("fix", "the bug"),
    ("believe", "the witness"),
    ("like", "the plan"),
]

EXISTENTIAL_LOCATIONS = [
    "the drawer",
    "the pipe",
    "the design",
    "the schedule",
    "the contract",
    "the warehouse",
]
EXISTENTIAL_OBJECTS = [
    ("key", "a key"),
    ("leak", "a leak"),
    ("flaw", "a flaw"),
    ("delay", "a delay"),
    ("typo", "a typo"),
    ("guard", "a guard"),
]

MODAL_ENTITIES = [
    "The pilot",
    "The server",
    "The witness",
    "The bridge",
    "The battery",
]
MODAL_VERBS = [
    "land safely",
    "restart",
    "testify",
    "hold the weight",
    "charge fully",
]


def contains_negation_marker(text: str) -> bool:
    return bool(_NEGATION_RE.search(text.lower()))


def _stable_id(family: str, fields: Iterable[str]) -> str:
    payload = "|".join([family, *fields]).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"{family}-{digest}"


def _build_pair(
    *,
    family: str,
    x_pos: str,
    x_neg: str,
    x_para: str,
    x_decoy: str,
    held_constant: list[str],
) -> MinimalPair:
    return MinimalPair(
        id=_stable_id(family, [x_pos, x_neg, x_para, x_decoy]),
        domain="negation",
        concept="negation_scope",
        template_family=family,
        x_pos=x_pos,
        x_neg=x_neg,
        x_para=x_para,
        x_decoy=x_decoy,
        held_constant=held_constant,
        changed_variable="negation_presence",
        control_purity=score_control_purity(
            x_pos,
            x_neg,
            x_para=x_para,
            x_decoy=x_decoy,
        ),
    )


def _take_shuffled(items: list[tuple], n: int | None, rng: random.Random) -> list[tuple]:
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled if n is None else shuffled[:n]


def _gen_copula(n: int | None, rng: random.Random) -> list[MinimalPair]:
    pairs = []
    combos = list(itertools.product(COPULA_ENTITIES, COPULA_PROPERTIES))
    for entity, prop in _take_shuffled(combos, n, rng):
        pairs.append(
            _build_pair(
                family="copula",
                x_pos=f"{entity} is not {prop}.",
                x_neg=f"{entity} is {prop}.",
                x_para=f"{entity} isn't {prop}.",
                x_decoy=f"{entity} is often {prop}.",
                held_constant=["entity", "property", "syntax_frame", "topic"],
            )
        )
    return pairs


def _gen_do_support(n: int | None, rng: random.Random) -> list[MinimalPair]:
    pairs = []
    combos = list(itertools.product(DO_SUPPORT_ENTITIES, DO_SUPPORT_PAIRS))
    for entity, (verb, obj) in _take_shuffled(combos, n, rng):
        pairs.append(
            _build_pair(
                family="do_support",
                x_pos=f"{entity} does not {verb} {obj}.",
                x_neg=f"{entity} does {verb} {obj}.",
                x_para=f"{entity} doesn't {verb} {obj}.",
                x_decoy=f"{entity} does often {verb} {obj}.",
                held_constant=["entity", "verb", "object", "syntax_frame", "topic"],
            )
        )
    return pairs


def _gen_existential(n: int | None, rng: random.Random) -> list[MinimalPair]:
    pairs = []
    combos = list(itertools.product(EXISTENTIAL_OBJECTS, EXISTENTIAL_LOCATIONS))
    for (bare, indefinite), location in _take_shuffled(combos, n, rng):
        pairs.append(
            _build_pair(
                family="existential",
                x_pos=f"There is no {bare} in {location}.",
                x_neg=f"There is {indefinite} in {location}.",
                x_para=f"There isn't {indefinite} in {location}.",
                x_decoy=f"There is sometimes {indefinite} in {location}.",
                held_constant=["object", "location", "syntax_frame", "topic"],
            )
        )
    return pairs


def _gen_modal(n: int | None, rng: random.Random) -> list[MinimalPair]:
    pairs = []
    combos = list(itertools.product(MODAL_ENTITIES, MODAL_VERBS))
    for entity, verb in _take_shuffled(combos, n, rng):
        pairs.append(
            _build_pair(
                family="modal",
                x_pos=f"{entity} cannot {verb}.",
                x_neg=f"{entity} can {verb}.",
                x_para=f"{entity} can't {verb}.",
                x_decoy=f"{entity} can sometimes {verb}.",
                held_constant=["entity", "verb", "syntax_frame", "topic"],
            )
        )
    return pairs


def generate_negation_pairs(per_family: int = 15, seed: int = 7) -> list[MinimalPair]:
    rng = random.Random(seed)
    pairs: list[MinimalPair] = []
    pairs.extend(_gen_copula(per_family, rng))
    pairs.extend(_gen_do_support(per_family, rng))
    pairs.extend(_gen_existential(per_family, rng))
    pairs.extend(_gen_modal(per_family, rng))
    return pairs
