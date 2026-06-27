from __future__ import annotations

import re
from itertools import count

from mwb.hashing import sha256_text

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")
_COUNTERS: dict[str, count] = {}


def slugify(value: str) -> str:
    slug = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    return slug or "unnamed"


def ref_from_name(prefix: str, name: str) -> str:
    return f"{prefix}_{slugify(name)}"


def next_ref(prefix: str) -> str:
    counter = _COUNTERS.setdefault(prefix, count(1))
    return f"{prefix}_{next(counter):06d}"


def stable_ref(prefix: str, *parts: object, length: int = 16) -> str:
    material = "\x1f".join(str(part) for part in parts)
    digest = sha256_text(material).split(":", 1)[1][:length]
    return f"{prefix}_{digest}"
