from __future__ import annotations

import difflib
import re

from mechledger.project import Project

HEADING_SEPARATOR = re.compile(r"^(#{2,3}\s+[CD][0-9]+[A-Za-z0-9_-]*)\s+(?:-|—)\s+(.+?)\s*$")


def format_project(project: Project, *, write: bool = False) -> tuple[bool, str]:
    paths = [
        project.root / project.config.default_claim_ledger,
        project.root / project.config.default_decision_log,
    ]
    changed = False
    output: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        before = path.read_text(encoding="utf-8")
        after = normalize_ledger_text(before)
        if before == after:
            continue
        changed = True
        output.extend(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=str(path),
                tofile=str(path),
                lineterm="",
            )
        )
        if write:
            path.write_text(after, encoding="utf-8")
    for template in [
        project.root / "research/experiments/TEMPLATE_experiment.md",
        project.root / "research/paper/draft.md",
    ]:
        if template.exists():
            trimmed = (
                "\n".join(
                    line.rstrip() for line in template.read_text(encoding="utf-8").splitlines()
                )
                + "\n"
            )
            if trimmed != template.read_text(encoding="utf-8"):
                changed = True
                if write:
                    template.write_text(trimmed, encoding="utf-8")
    return changed, "\n".join(output) + ("\n" if output else "")


def normalize_ledger_text(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        match = HEADING_SEPARATOR.match(line)
        if match:
            out.append(f"{match.group(1)} - {match.group(2)}")
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            if index < len(lines) and lines[index].strip().startswith("```yaml"):
                out.append("")
                out.append(lines[index].rstrip())
                index += 1
            continue
        out.append(line)
        index += 1
    return "\n".join(out).rstrip() + "\n"
