from __future__ import annotations

from pathlib import Path

from mechledger.io import utc_now, write_json


def close_session(project_root: str | Path, *, accept: bool = False) -> Path:
    root = Path(project_root)
    drafts = root / ".mechledger" / "session_drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now().replace(":", "").replace("-", "")
    draft = drafts / f"{timestamp}.md"
    content = f"""\
## {utc_now()[:10]}

```yaml
entry_id: R{utc_now()[:10]}-draft
linked_runs: []
linked_claims: []
linked_decisions: []
open_questions: []
copilot_session_id: null
```

### Question

### Context

### Hypothesis

### Work done

### Result

### Interpretation

### Decision

### Open questions
"""
    draft.write_text(content, encoding="utf-8")
    if accept:
        log = root / "research" / "logs" / "research_log.md"
        existing = log.read_text(encoding="utf-8") if log.exists() else "# Research Log\n"
        log.write_text(existing.rstrip() + "\n\n" + content, encoding="utf-8")
        write_json(
            root / ".mechledger" / "last_session_close.json",
            {
                "last_session_close_at": utc_now(),
                "last_session_close_commit": None,
                "last_session_close_entry_id": f"R{utc_now()[:10]}-draft",
            },
        )
        draft.unlink(missing_ok=True)
        return log
    return draft
