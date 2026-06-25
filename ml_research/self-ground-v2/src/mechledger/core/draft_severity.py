from __future__ import annotations

from enum import StrEnum


class DraftSeverity(StrEnum):
    # Draft severity is about commit-blocking prose findings; it has no `serious`.
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"
