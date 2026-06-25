from __future__ import annotations

import difflib
import re
from pathlib import Path


def format_project(project_root: str | Path, *, write: bool = False) -> dict[str, str]:
    root = Path(project_root)
    changes: dict[str, str] = {}
    for path in [
        root / "research" / "logs" / "claim_ledger.md",
        root / "research" / "logs" / "decision_log.md",
        *sorted((root / "research" / "experiments").glob("*.md")),
    ]:
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        formatted = _normalize_markdown(original)
        if formatted != original:
            diff = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    formatted.splitlines(keepends=True),
                    fromfile=str(path),
                    tofile=str(path),
                )
            )
            changes[str(path)] = diff
            if write:
                path.write_text(formatted, encoding="utf-8")
    return changes


def _normalize_markdown(text: str) -> str:
    text = re.sub(r"^(###\s+C[0-9A-Za-z_-]+)\s+—\s+", r"\1 - ", text, flags=re.MULTILINE)
    text = re.sub(r"^(##\s+D[0-9A-Za-z_-]+)\s+—\s+", r"\1 - ", text, flags=re.MULTILINE)
    text = re.sub(
        r"^(###\s+C[0-9A-Za-z_-]+\s+-\s+.*?)\n{3,}```yaml",
        r"\1\n\n```yaml",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^(##\s+D[0-9A-Za-z_-]+\s+-\s+.*?)\n{3,}```yaml",
        r"\1\n\n```yaml",
        text,
        flags=re.MULTILINE,
    )
    if not text.endswith("\n"):
        text += "\n"
    return text
