from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from helpers_project import (
    create_run,
    init_project,
    runner,
    write_claim_ledger,
    write_decision_log,
    write_run_ledger,
)

from mechledger.alias import append_alias, resolve_run_id
from mechledger.cli import app


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.PIPE, text=True)


def _init_git(root: Path) -> None:
    _git(root, "init")
    _git(root, "config", "user.email", "mechledger@example.test")
    _git(root, "config", "user.name", "MechLedger Test")


def _run_json(tmp_path: Path, run_id: str, *, status: str, run_class: str = "diagnostic") -> None:
    run_dir = tmp_path / ".mechledger/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "experiment_id": "E001",
                "run_class": run_class,
                "status": status,
                "started_at": "2026-06-25T00:00:00Z",
                "finished_at": "2026-06-25T00:00:01Z",
                "exit_code": 1 if status == "failed" else None,
                "command": "python script.py",
                "argv": ["python", "script.py"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for name in ["events.jsonl", "metrics.jsonl", "artifacts.jsonl"]:
        (run_dir / name).write_text("", encoding="utf-8")
    (run_dir / "artifact_manifest.json").write_text('{"artifacts": []}\n', encoding="utf-8")
    with (tmp_path / ".mechledger/alias_cache.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"{run_id}\t2026-06-25T00:00:00Z\tE001\t{run_id}\n")


def _passing_metrics_source() -> str:
    metrics: dict[str, Any] = {
        "intended_direction_pass_rate": 1.0,
        "baseline_contrast": 0.5,
        "positive_control_pass_rate": 0.95,
        "random_null_seed_count": 30,
        "percentile_rank": 0.99,
        "paired_test_name": "sign",
        "paired_by": "task_id",
        "paired_test_n_pairs": 40,
        "paired_test_p_value": 0.01,
        "effect_direction": "positive",
        "sign_consistency": 0.9,
        "target_delta": 0.4,
        "matched_control_delta": 0.05,
        "specificity_gap": 0.35,
        "top_control_ratio": 0.2,
        "multi_control_min_gap": 0.1,
        "family_min_gap": 0.1,
        "relative_norm_drift": 0.1,
        "nonfinite_rate": 0.0,
        "skip_rate": 0.0,
        "metadata_compatible": True,
    }
    return repr(metrics)


def test_staged_draft_reads_worktree_and_can_pass_with_staged_claim_update(
    tmp_path: Path,
) -> None:
    _init_git(tmp_path)
    init_project(tmp_path)
    claim_path = tmp_path / "research/logs/claim_ledger.md"
    draft = tmp_path / "research/paper/draft.md"
    claim_path.write_text(
        """# Claim Ledger

### C777 - Staged claim

```yaml
claim_id: C777
status: single_run_evidence
allowed: [preliminary]
forbidden: [proves]
required_caveats: [single run]
debt_flags: []
linked_runs: []
```
""",
        encoding="utf-8",
    )
    draft.write_text("This proves the mechanism. [CLAIM:C777]\n", encoding="utf-8")
    _git(tmp_path, "add", "-f", "research/logs/claim_ledger.md", "research/paper/draft.md")
    draft.write_text(
        "This is preliminary and from a single run. [CLAIM:C777]\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app, ["draft", "check", "--staged"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )

    assert result.exit_code == 0, result.output


def test_staged_draft_and_index_skip_when_only_irrelevant_paths_are_staged(
    tmp_path: Path,
) -> None:
    _init_git(tmp_path)
    init_project(tmp_path)
    (tmp_path / "notes.txt").write_text("not relevant\n", encoding="utf-8")
    _git(tmp_path, "add", "notes.txt")

    draft = runner.invoke(
        app, ["draft", "check", "--staged"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    index = runner.invoke(
        app, ["index", "--check", "--staged"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )

    assert draft.exit_code == 0, draft.output
    assert "no staged draft or claim files changed" in draft.output
    assert index.exit_code == 0, index.output
    assert "no staged research/indexed files changed" in index.output


def test_staged_index_runs_for_research_and_project_config_paths(tmp_path: Path) -> None:
    _init_git(tmp_path)
    init_project(tmp_path)
    write_claim_ledger(tmp_path)
    for rel in ["research/logs/claim_ledger.md", ".mechledger/project.json"]:
        _git(tmp_path, "reset")
        _git(tmp_path, "add", "-f", rel)
        result = runner.invoke(
            app,
            ["index", "--check", "--staged"],
            catch_exceptions=False,
            env={"PWD": str(tmp_path)},
        )
        assert result.exit_code == 0, result.output
        assert "index check passed" in result.output


def test_install_hooks_merges_precommit_config_and_direct_hook_is_git_local(
    tmp_path: Path,
) -> None:
    _init_git(tmp_path)
    init_project(tmp_path)
    existing = tmp_path / ".pre-commit-config.yaml"
    existing.write_text(
        """repos:
- repo: https://example.test/other
  rev: v1
  hooks:
  - id: other-hook
""",
        encoding="utf-8",
    )

    portable = runner.invoke(
        app, ["install-hooks"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    direct = runner.invoke(
        app, ["install-hooks", "--direct"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )

    assert portable.exit_code == 0, portable.output
    config = existing.read_text(encoding="utf-8")
    assert "https://example.test/other" in config
    assert "mechledger draft check --staged" in config
    assert "mechledger index --check --staged" in config
    assert direct.exit_code == 0, direct.output
    assert "For portable hooks" in direct.output
    hook = tmp_path / ".git/hooks/pre-commit"
    assert hook.exists()
    assert "mechledger draft check --staged" in hook.read_text(encoding="utf-8")
    assert not (tmp_path / ".mechledger/hooks/pre-commit").exists()


def test_format_check_reports_changes_without_mutating(tmp_path: Path) -> None:
    init_project(tmp_path)
    claim_path = tmp_path / "research/logs/claim_ledger.md"
    claim_path.write_text(
        """# Claim Ledger

### C001 — Em dash heading


```yaml
claim_id: C001
status: unsupported
allowed: []
forbidden: []
```
""",
        encoding="utf-8",
    )
    before = claim_path.read_text(encoding="utf-8")

    check = runner.invoke(
        app, ["format", "--check"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )

    assert check.exit_code == 1
    assert "C001 - Em dash heading" in check.output
    assert claim_path.read_text(encoding="utf-8") == before

    write = runner.invoke(
        app, ["format", "--write"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    clean = runner.invoke(
        app, ["format", "--check"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )

    assert write.exit_code == 0, write.output
    assert clean.exit_code == 0, clean.output
    assert "format check passed" in clean.output.lower()


def test_index_uses_temp_fallback_when_mechledger_dir_is_read_only(tmp_path: Path) -> None:
    init_project(tmp_path)
    write_claim_ledger(tmp_path)
    mechledger_dir = tmp_path / ".mechledger"
    original_mode = stat.S_IMODE(mechledger_dir.stat().st_mode)
    try:
        mechledger_dir.chmod(0o555)
        result = runner.invoke(app, ["index"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    finally:
        mechledger_dir.chmod(original_mode)

    assert result.exit_code == 0, result.output
    assert "mechledger_cache_" in result.output
    assert not (tmp_path / ".mechledger/index.sqlite").exists()


def test_run_capture_fails_when_run_directory_cannot_be_written(tmp_path: Path) -> None:
    init_project(tmp_path)
    runs_dir = tmp_path / ".mechledger/runs"
    original_mode = stat.S_IMODE(runs_dir.stat().st_mode)
    try:
        runs_dir.chmod(0o555)
        result = runner.invoke(
            app,
            ["run", "--run-id", "RUN_NOWRITE", "--", sys.executable, "-c", "print('x')"],
            catch_exceptions=False,
            env={"PWD": str(tmp_path)},
        )
    finally:
        runs_dir.chmod(original_mode)

    assert result.exit_code == 2
    assert "RUN_NOWRITE" in result.output or "Permission" in result.output
    assert not (tmp_path / ".mechledger/runs/RUN_NOWRITE").exists()


def test_alias_resolution_forms_and_malformed_rebuild_marker(tmp_path: Path) -> None:
    init_project(tmp_path)
    cache = tmp_path / ".mechledger/alias_cache.txt"
    cache.write_text(
        "20260625T120000Z_e001_verify_path_abcd\t2026-06-25T12:00:00Z\tE001\tverify_path\n"
        "20260625T121000Z_e001_other_path_abcd\t2026-06-25T12:10:00Z\tE001\tother_path\n"
        "malformed partial line\n",
        encoding="utf-8",
    )

    assert resolve_run_id(apply_project(tmp_path), "20260625T120000Z") == (
        "20260625T120000Z_e001_verify_path_abcd"
    )
    assert resolve_run_id(apply_project(tmp_path), "e001_verify") == (
        "20260625T120000Z_e001_verify_path_abcd"
    )
    assert resolve_run_id(apply_project(tmp_path), "latest") == (
        "20260625T121000Z_e001_other_path_abcd"
    )
    assert resolve_run_id(apply_project(tmp_path), "latest:2") == (
        "20260625T120000Z_e001_verify_path_abcd"
    )
    assert resolve_run_id(apply_project(tmp_path), "#1") == (
        "20260625T120000Z_e001_verify_path_abcd"
    )
    assert (tmp_path / ".mechledger/cache/alias_rebuild_required").exists()


def test_alias_ambiguity_and_no_sweep_during_normal_resolution(tmp_path: Path) -> None:
    init_project(tmp_path)
    create_run(tmp_path, run_id="RUN_ALIAS_ONE")
    create_run(tmp_path, run_id="RUN_ALIAS_TWO")
    (tmp_path / ".mechledger/alias_cache.txt").write_text(
        "RUN_ALIAS_ONE\t2026-06-25T00:00:00Z\tE001\tshared\n"
        "RUN_ALIAS_TWO\t2026-06-25T00:01:00Z\tE001\tshared\n",
        encoding="utf-8",
    )
    ambiguous = runner.invoke(
        app, ["gate", "check", "RUN_ALIAS"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    assert ambiguous.exit_code == 2
    assert "ambiguous" in ambiguous.output
    assert "RUN_ALIAS_ONE" in ambiguous.output and "RUN_ALIAS_TWO" in ambiguous.output

    (tmp_path / ".mechledger/alias_cache.txt").unlink()
    latest = runner.invoke(
        app, ["gate", "check", "latest"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    direct = runner.invoke(
        app, ["gate", "check", "RUN_ALIAS_ONE"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    assert latest.exit_code == 2
    assert "No runs are recorded" in latest.output
    assert direct.exit_code in {0, 1}
    assert not (tmp_path / ".mechledger/alias_cache.txt").exists()

    rebuilt = runner.invoke(app, ["index"], catch_exceptions=False, env={"PWD": str(tmp_path)})
    assert rebuilt.exit_code == 0, rebuilt.output
    assert (tmp_path / ".mechledger/alias_cache.txt").exists()


def test_alias_append_is_newline_terminated(tmp_path: Path) -> None:
    init_project(tmp_path)
    project = apply_project(tmp_path)
    append_alias(project, "RUN_LOCKED", "E001", "locked")

    payload = (tmp_path / ".mechledger/alias_cache.txt").read_bytes()
    assert payload.endswith(b"\n")


def test_failed_or_cancelled_runs_cannot_promote_to_evidence_classes(tmp_path: Path) -> None:
    init_project(tmp_path)
    write_decision_log(tmp_path, status="accepted")
    _run_json(tmp_path, "RUN_FAILED", status="failed")
    _run_json(tmp_path, "RUN_CANCELLED", status="cancelled")

    for run_id in ["RUN_FAILED", "RUN_CANCELLED"]:
        result = runner.invoke(
            app,
            [
                "run",
                "reclassify",
                run_id,
                "--to",
                "serious_evidence_run",
                "--decision",
                "D001",
                "--reason",
                "should not promote terminal failed state",
            ],
            catch_exceptions=False,
            env={"PWD": str(tmp_path)},
        )
        assert result.exit_code == 2
        assert "failed/cancelled" in result.output


def test_claim_proposal_staleness_uses_semantic_hash(tmp_path: Path) -> None:
    init_project(tmp_path)
    write_claim_ledger(tmp_path)
    create_run(tmp_path, run_id="RUN_STALE")
    proposal = runner.invoke(
        app,
        ["claim", "propose", "--run", "RUN_STALE", "--regenerate"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    assert proposal.exit_code == 0, proposal.output
    ledger_path = tmp_path / "research/logs/claim_ledger.md"
    original = ledger_path.read_text(encoding="utf-8")
    reformatted = """# Claim Ledger

### C001 - Target feature claim

```yaml
forbidden:
  - proves causally
allowed:
  - preliminary evidence
claim_id: C001
status: candidate_claim
linked_runs: [RUN_E001]
linked_decisions: [D001]
linked_experiments: [E001]
required_caveats:
  - single-run evidence
debt_flags:
  - missing_empirical_null
scope: negation prompts
```

Different prose below the YAML must not affect staleness.
"""
    ledger_path.write_text(reformatted, encoding="utf-8")
    current = runner.invoke(
        app, ["claim", "review", "RUN_STALE"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    assert current.exit_code == 0, current.output
    assert "current" in current.output

    ledger_path.write_text(
        original.replace("candidate_claim", "failed_or_weakened"),
        encoding="utf-8",
    )
    stale_status = runner.invoke(
        app, ["claim", "review", "RUN_STALE"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    assert "stale" in stale_status.output

    ledger_path.write_text(original.replace("RUN_E001", "RUN_OTHER"), encoding="utf-8")
    stale_run = runner.invoke(
        app, ["claim", "review", "RUN_STALE"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    assert "stale" in stale_run.output

    ledger_path.write_text(
        original.replace("preliminary evidence", "new allowed phrase"), encoding="utf-8"
    )
    forced = runner.invoke(
        app,
        ["claim", "review", "RUN_STALE", "--apply", "--yes", "--force-stale"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    proposal_payload = json.loads(
        (tmp_path / ".mechledger/runs/RUN_STALE/claim_update_proposal.json").read_text(
            encoding="utf-8"
        )
    )
    assert forced.exit_code == 0, forced.output
    assert proposal_payload["force_applied"] is True
    assert proposal_payload["review_status"] == "stale"


def test_no_ml_end_to_end_mvp_loop(tmp_path: Path) -> None:
    init_project(tmp_path)
    write_decision_log(tmp_path, status="accepted")
    write_run_ledger(tmp_path)
    write_claim_ledger(tmp_path, status="single_run_evidence")
    draft = tmp_path / "research/paper/draft.md"
    draft.write_text(
        "This is preliminary evidence with a single-run evidence caveat. [CLAIM:C001]\n",
        encoding="utf-8",
    )
    script = tmp_path / "small_script.py"
    script.write_text(
        "from pathlib import Path\n"
        "import os\n"
        "import mechledger as ml\n"
        f"metrics = {_passing_metrics_source()}\n"
        "for name, value in metrics.items():\n"
        "    ml.log_metric(name, value)\n"
        "ml.log_intervention_metadata(feature_id='sae_123', features_modified=['sae_123'])\n"
        "artifact = Path(os.environ['MECHLEDGER_RUN_DIR']) / 'artifacts' / 'result.json'\n"
        "artifact.write_text('{\"ok\": true}\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )

    index = runner.invoke(
        app, ["index", "--check"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    run = runner.invoke(
        app,
        [
            "run",
            "--experiment",
            "E001",
            "--class",
            "serious_evidence_run",
            "--purpose",
            "no ml loop",
            "--",
            sys.executable,
            str(script),
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    annotate = runner.invoke(
        app,
        ["artifact", "annotate", "latest", "A001", "--claim-relevance", "required"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    gate = runner.invoke(
        app, ["gate", "check", "latest"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    propose = runner.invoke(
        app,
        ["claim", "propose", "--run", "latest", "--regenerate"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    review = runner.invoke(
        app, ["claim", "review", "latest"], catch_exceptions=False, env={"PWD": str(tmp_path)}
    )
    draft_check = runner.invoke(
        app,
        ["draft", "check", str(draft)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    appendix = runner.invoke(
        app,
        ["export", "appendix", "--out", str(tmp_path / "research/paper/mechledger_appendix.md")],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )
    bundle = runner.invoke(
        app,
        [
            "export",
            "bundle",
            "--out",
            str(tmp_path / "bundles/manifest_bundle.tar.gz"),
            "--manifest-only",
        ],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert index.exit_code == 0, index.output
    assert run.exit_code == 0, run.output
    assert annotate.exit_code == 0, annotate.output
    assert gate.exit_code == 0, gate.output
    assert propose.exit_code == 0, propose.output
    assert review.exit_code == 0 and "current" in review.output
    assert draft_check.exit_code == 0, draft_check.output
    assert appendix.exit_code == 0, appendix.output
    assert bundle.exit_code == 0, bundle.output
    assert (tmp_path / "research/paper/mechledger_appendix.md").exists()
    assert (tmp_path / "bundles/manifest_bundle.tar.gz").exists()


def apply_project(tmp_path: Path):
    from mechledger.project import find_project

    return find_project(tmp_path)
