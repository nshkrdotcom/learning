from __future__ import annotations

from pathlib import Path

from mechledger.indexer import index_project


def project_status(project_root: str | Path) -> str:
    root = Path(project_root)
    index = index_project(root)
    lines = [
        f"Project: {root.name}",
        f"Root: {root}",
        "Schema: 0.1.0",
        "Index: fresh",
        "",
        "Claims:",
    ]
    for status, count in sorted(index.claim_count_by_status.items()):
        lines.append(f"  {status}: {count}")
    lines.extend(
        [
            "",
            f"Experiments: {index.experiment_count}",
            f"Runs: {index.run_count}",
            "",
            "Scientific Debt:",
        ]
    )
    if index.debt_count_by_severity:
        for severity, count in sorted(index.debt_count_by_severity.items()):
            lines.append(f"  {severity}: {count}")
    else:
        lines.append("  none: 0")
    return "\n".join(lines) + "\n"
