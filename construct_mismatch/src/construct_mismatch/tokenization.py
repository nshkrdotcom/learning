from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

CERTAINTY_TARGETS = (
    " certain",
    " uncertain",
    " sure",
    " unsure",
    " definite",
    " doubtful",
    " confident",
    " hesitant",
    " clear",
    " unclear",
    " obvious",
    " ambiguous",
)

SENTIMENT_TARGETS = (
    " great",
    " terrible",
    " good",
    " bad",
    " excellent",
    " awful",
    " amazing",
    " poor",
    " wonderful",
    " disappointing",
)

CANDIDATE_TARGETS = {
    "certainty": CERTAINTY_TARGETS,
    "sentiment": SENTIMENT_TARGETS,
}


@dataclass(frozen=True)
class TokenizationRow:
    construct: str
    raw_string: str
    token_ids: list[int]
    decoded_text: str
    n_tokens: int
    is_single_token: bool
    usable_as_target: bool


def inspect_string(
    construct: str,
    raw_string: str,
    encode_text: Callable[[str], list[int]],
    decode_tokens: Callable[[Iterable[int]], str],
) -> TokenizationRow:
    token_ids = list(encode_text(raw_string))
    decoded_text = decode_tokens(token_ids)
    is_single = len(token_ids) == 1
    return TokenizationRow(
        construct=construct,
        raw_string=raw_string,
        token_ids=token_ids,
        decoded_text=decoded_text,
        n_tokens=len(token_ids),
        is_single_token=is_single,
        usable_as_target=is_single and decoded_text == raw_string,
    )


def inspect_candidates(
    encode_text: Callable[[str], list[int]],
    decode_tokens: Callable[[Iterable[int]], str],
) -> list[TokenizationRow]:
    rows: list[TokenizationRow] = []
    for construct, targets in CANDIDATE_TARGETS.items():
        for raw_string in targets:
            rows.append(inspect_string(construct, raw_string, encode_text, decode_tokens))
    return rows


def validate_single_token_targets(
    targets: Iterable[str],
    encode_text: Callable[[str], list[int]],
    decode_tokens: Callable[[Iterable[int]], str],
) -> dict[str, int]:
    target_to_id: dict[str, int] = {}
    invalid: list[str] = []
    for target in sorted(set(targets)):
        row = inspect_string("target", target, encode_text, decode_tokens)
        if not row.usable_as_target:
            invalid.append(f"{target!r} -> ids={row.token_ids}, decoded={row.decoded_text!r}")
        else:
            target_to_id[target] = row.token_ids[0]
    if invalid:
        details = "\n".join(invalid)
        raise ValueError(f"Targets are invalid in single-token mode:\n{details}")
    return target_to_id


def usable_target_pairs(
    pairs: Iterable[tuple[str, str]],
    encode_text: Callable[[str], list[int]],
    decode_tokens: Callable[[Iterable[int]], str],
) -> list[tuple[str, str]]:
    usable: list[tuple[str, str]] = []
    for class_a_target, class_b_target in pairs:
        rows = [
            inspect_string("target", class_a_target, encode_text, decode_tokens),
            inspect_string("target", class_b_target, encode_text, decode_tokens),
        ]
        if all(row.usable_as_target for row in rows):
            usable.append((class_a_target, class_b_target))
    if not usable:
        raise ValueError("No usable single-token target pairs found.")
    return usable
