from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers_project import populate_project, runner

from mechledger.cli import app
from mechledger.core.claim_ledger import parse_claim_ledger
from mechledger.core.decision_log import parse_decision_log
from mechledger.core.experiment_spec import parse_experiment_spec
from mechledger.core.research_log import parse_research_log
from mechledger.core.run_ledger import parse_run_ledger
from mechledger.project import run_ledger_header


def test_malformed_claim_heading_reports_path_line_rule_and_fix(tmp_path: Path) -> None:
    path = tmp_path / "research/logs/claim_ledger.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        """# Claim Ledger

### Claim C001 is malformed
```yaml
claim_id: C001
status: candidate_claim
allowed: []
forbidden: []
```
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        parse_claim_ledger(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert ":3" in message
    assert "Rule: claim.heading.malformed" in message
    assert "Suggested fix" in message


def test_duplicate_claim_reports_object_id_and_rule(tmp_path: Path) -> None:
    path = tmp_path / "research/logs/claim_ledger.md"
    path.parent.mkdir(parents=True)
    block = """### C001 - Claim
```yaml
claim_id: C001
status: candidate_claim
allowed: []
forbidden: []
```
"""
    path.write_text("# Claim Ledger\n\n" + block + "\n" + block, encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        parse_claim_ledger(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "C001" in message
    assert ":11" in message
    assert "C001" in message
    assert "Rule: claim.id.duplicate" in message
    assert "Suggested fix" in message


def test_claim_yaml_mismatch_reports_line_object_and_rule(tmp_path: Path) -> None:
    path = tmp_path / "research/logs/claim_ledger.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        """# Claim Ledger

### C001 - Claim
```yaml
claim_id: C002
status: candidate_claim
allowed: []
forbidden: []
```
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        parse_claim_ledger(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "C001" in message
    assert ":5" in message
    assert "Rule: claim.id.mismatch" in message
    assert "Suggested fix" in message


def test_duplicate_decision_reports_object_id_and_rule(tmp_path: Path) -> None:
    path = tmp_path / "research/logs/decision_log.md"
    path.parent.mkdir(parents=True)
    block = """## D001 - Decision
```yaml
decision_id: D001
status: accepted
affected_claims: []
affected_experiments: []
```
"""
    path.write_text("# Decision Log\n\n" + block + "\n" + block, encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        parse_decision_log(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert ":11" in message
    assert "D001" in message
    assert "Rule: decision.id.duplicate" in message
    assert "Suggested fix" in message


def test_experiment_missing_required_field_reports_object_id_rule_and_fix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "research/experiments/E001_test.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        """# E001: Test

```yaml
experiment_id: E001
claim_targets: []
source_runs: []
```
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        parse_experiment_spec(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert ":4" in message
    assert "E001" in message
    assert "Rule: experiment.yaml.required_field" in message
    assert "Suggested fix" in message


def test_platform_record_invalid_enum_reports_path_type_field_rule_and_fix(
    tmp_path: Path,
) -> None:
    populate_project(tmp_path)
    path = tmp_path / "research/records/bad_activation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "record_id": "REC-BAD",
                "record_type": "ActivationRecord",
                "activation_id": "ACT-BAD",
                "run_id": "RUN_E001",
                "model": "pythia-70m",
                "tokenizer": None,
                "hook_name": "blocks.2.hook_resid_post",
                "layer": 2,
                "component_type": "residual_stream",
                "batch_index": 0,
                "token_position": 1,
                "generation_step": None,
                "shape": [1, 2],
                "dtype": "float32",
                "device": "cpu",
                "tensor_artifact_path": None,
                "tensor_hash": None,
                "artifact_storage_backend": "s3",
                "content_hash_status": "computed",
                "summary_stats": {"mean": 0.0, "std": 1.0, "norm": 2.0, "max_abs": 1.5},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["records", "validate", str(path)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code != 0
    assert "bad_activation.json" in result.output
    assert "ActivationRecord" in result.output
    assert "ACT-BAD" in result.output
    assert "artifact_storage_backend" in result.output
    assert "Rule:" in result.output
    assert "Suggested fix" in result.output


def test_index_check_surfaces_parser_diagnostics(tmp_path: Path) -> None:
    populate_project(tmp_path)
    claim_path = tmp_path / "research/logs/claim_ledger.md"
    claim_path.write_text(
        """# Claim Ledger

### Bad heading
```yaml
claim_id: C001
status: candidate_claim
allowed: []
forbidden: []
```
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["index", "--check"],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code != 0
    assert "claim_ledger.md" in result.output
    assert ":3" in result.output
    assert "Rule: claim.heading.malformed" in result.output
    assert "Suggested fix" in result.output


def test_research_log_malformed_entry_heading_reports_path_line_rule_and_fix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "research/logs/research_log.md"
    path.parent.mkdir(parents=True)
    path.write_text("# Research Log\n\n## Not a date\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        parse_research_log(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert ":3" in message
    assert "Rule: research_log.heading.malformed" in message
    assert "Suggested fix" in message


def test_run_ledger_invalid_values_report_row_object_rule_and_fix(tmp_path: Path) -> None:
    path = tmp_path / "research/logs/run_ledger.csv"
    path.parent.mkdir(parents=True)
    columns = run_ledger_header().split(",")
    row = dict.fromkeys(columns, "")
    row.update(
        {
            "timestamp": "2026-06-25T00:00:00Z",
            "run_id": "RUN_BAD",
            "status": "done-ish",
            "run_class": "model_backend",
            "phase": "phase",
        }
    )
    path.write_text(
        run_ledger_header()
        + "\n"
        + ",".join(str(row[column]) for column in columns)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        parse_run_ledger(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert ":2" in message
    assert "RUN_BAD" in message
    assert "Rule: run_ledger.status.invalid" in message
    assert "Suggested fix" in message


def test_prediction_lock_invalid_record_surfaces_structured_diagnostic(
    tmp_path: Path,
) -> None:
    populate_project(tmp_path)
    path = tmp_path / "research/predictions/bad.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "prediction_id": "PRED-BAD",
                "feature_id": "sae_123",
                "source_examples_path": "examples.jsonl",
                "prediction_artifact_path": "research/predictions/bad.json",
                "short_label": "bad",
                "predicted_target_direction": "up",
                "predicted_control_direction": "decrease",
                "predicted_relative_magnitude": "target_gt_control",
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["prediction", "lock", str(path)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 2
    assert "bad.json" in result.output
    assert "PRED-BAD" in result.output
    assert "Rule: prediction.validation" in result.output
    assert "predicted_target_direction" in result.output
    assert "Suggested fix" in result.output


def test_external_label_validate_invalid_jsonl_row_surfaces_structured_diagnostic(
    tmp_path: Path,
) -> None:
    populate_project(tmp_path)
    path = tmp_path / "labels.jsonl"
    path.write_text(
        json.dumps(
            {
                "label_id": "L-BAD",
                "source": "paper",
                "label_text": "feature label",
                "feature_id": "sae_123",
                "linked_claims": "C001",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["labels", "validate", str(path)],
        catch_exceptions=False,
        env={"PWD": str(tmp_path)},
    )

    assert result.exit_code == 2
    assert "labels.jsonl:1" in result.output
    assert "L-BAD" in result.output
    assert "Rule: external_label.validation" in result.output
    assert "linked_claims" in result.output
    assert "Suggested fix" in result.output
