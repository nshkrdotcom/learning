from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mechledger.cli import app

runner = CliRunner()


def write_claim_ledger(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Claim Ledger

### C001 - Scoped decoded intervention claim

```yaml
claim_id: C001
status: single_run_evidence
allowed:
  - preliminary
forbidden:
  - is the mechanism
  - proves that
required_caveats:
  - single run
debt_flags:
  - missing_empirical_null
linked_runs: []
```
""",
        encoding="utf-8",
    )


def test_init_is_idempotent_and_updates_gitignore(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".mechledger/project.json").exists()
    assert (tmp_path / "research/logs/claim_ledger.md").exists()
    gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert ".mechledger/runs/" in gitignore
    assert "research/" not in gitignore

    human = tmp_path / "research/paper/draft.md"
    human.write_text("human draft\n", encoding="utf-8")
    second = runner.invoke(app, ["init"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    assert second.exit_code == 0
    assert "skipped" in second.output
    assert human.read_text(encoding="utf-8") == "human draft\n"


def test_draft_check_tags_overrides_caveats_and_debt(tmp_path: Path) -> None:
    runner.invoke(app, ["init"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    write_claim_ledger(tmp_path / "research/logs/claim_ledger.md")
    draft = tmp_path / "research/paper/draft.md"
    draft.write_text(
        """The result is preliminary and from a single run. [CLAIM:C001]

This proves that the feature is the mechanism. [CLAIM:C001]

We do not claim this is the mechanism. [CLAIM:C001]

This proves that the feature is the mechanism. [CLAIM:C001]
<!-- mechledger-disable forbidden_language: reviewed in D001 -->

LaTeX says this is the mechanism without caveat. \\claim{C001}

HTML tag also works. <!-- CLAIM:C001 -->
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["draft", "check", str(draft), "--format", "json"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    violation_types = [item["violation_type"] for item in payload["violations"]]
    assert "forbidden_language" in violation_types
    assert "missing_required_caveat" in violation_types
    assert "unresolved_scientific_debt" in violation_types
    assert payload["overrides"][0]["violation_type"] == "forbidden_language"
    assert all(
        "do not claim this is the mechanism" not in item["window"].lower()
        for item in payload["violations"]
        if item["violation_type"] == "forbidden_language"
    )


def test_draft_check_real_false_positive_probe_has_no_blocker() -> None:
    result = runner.invoke(
        app,
        [
            "draft",
            "check",
            "tests/fixtures/real/false_positive_probe.md",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert not any(
        item["violation_type"] == "forbidden_language" for item in payload["violations"]
    )


def test_index_check_format_and_hooks(tmp_path: Path) -> None:
    runner.invoke(app, ["init"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    write_claim_ledger(tmp_path / "research/logs/claim_ledger.md")
    bad_spacing = tmp_path / "research/logs/decision_log.md"
    bad_spacing.write_text(
        """# Decision Log

## D001 — Use single-run caveat


```yaml
decision_id: D001
status: accepted
```
""",
        encoding="utf-8",
    )

    check = runner.invoke(
        app, ["index", "--check"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    assert check.exit_code == 0, check.output
    dry = runner.invoke(app, ["format"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    assert dry.exit_code == 1
    assert "---" in dry.output and "+++" in dry.output
    assert "## D001 — Use single-run caveat" in bad_spacing.read_text(encoding="utf-8")
    written = runner.invoke(
        app, ["format", "--write"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    assert written.exit_code == 0, written.output
    assert "## D001 - Use single-run caveat" in bad_spacing.read_text(encoding="utf-8")

    hooks = runner.invoke(
        app, ["install-hooks"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    assert hooks.exit_code == 0
    config = tmp_path / ".pre-commit-config.yaml"
    assert "mechledger draft check --staged" in config.read_text(encoding="utf-8")
