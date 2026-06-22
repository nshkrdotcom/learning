from __future__ import annotations

import csv
import json

from self_ground.behavioral_tasks import BehavioralTask, write_behavioral_tasks_jsonl
from self_ground.real_ranking import run_activation_ranking


def test_activation_ranking_artifacts_with_test_local_activations(
    tmp_path,
    tiny_model_adapter,
) -> None:
    out_dir = tmp_path / "ranking"

    result = run_activation_ranking(
        out_dir=out_dir,
        per_family=2,
        model_name="test-local",
        hook_point="test.layer",
        feature_source="residual_dimensions",
        pooling="final_token",
        top_k_features=3,
        model_adapter=tiny_model_adapter,
    )

    assert result.feature_source == "residual_dimensions"
    expected = {
        "config.json",
        "pairs.jsonl",
        "activation_metadata.json",
        "feature_rankings.csv",
        "top_examples.jsonl",
        "README.md",
    }
    assert expected == {path.name for path in out_dir.iterdir()}

    metadata = json.loads((out_dir / "activation_metadata.json").read_text())
    assert metadata["feature_source"] == "residual_dimensions"
    assert metadata["n_pairs"] == 8
    assert metadata["n_conditions"] == 32
    assert metadata["n_features"] == 4

    with (out_dir / "feature_rankings.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "feature_id",
            "score",
            "mean_pos",
            "mean_neg",
            "mean_para",
            "mean_decoy",
            "abs_score",
        ]
        rows = list(reader)
    assert rows
    assert rows[0]["feature_id"].startswith("resid_")

    first_top = json.loads((out_dir / "top_examples.jsonl").read_text().splitlines()[0])
    assert set(first_top) == {
        "feature_id",
        "score",
        "top_pos_examples",
        "top_neg_examples",
        "top_para_examples",
        "top_decoy_examples",
    }
    assert first_top["top_pos_examples"]
    assert first_top["top_pos_examples"][0]["condition"] == "x_pos"


def test_activation_ranking_can_use_calibrated_task_file(
    tmp_path,
    tiny_model_adapter,
) -> None:
    task_file = tmp_path / "tasks.jsonl"
    tasks = [
        BehavioralTask(
            id=f"{family}_0",
            family=family,
            concept="test",
            prompt="The movie was not good. The movie was",
            target_tokens=[" bad"],
            foil_tokens=[" good"],
            control_prompt="The movie was good. The movie was",
            control_type="matched_non_negation",
            control_target_tokens=[" good"],
            control_foil_tokens=[" bad"],
            expected_baseline_direction="positive",
            metadata={"template_family": "test"},
        )
        for family in ["sentiment_negation", "property_negation", "state_negation"]
    ]
    write_behavioral_tasks_jsonl(tasks, task_file)
    out_dir = tmp_path / "ranking_from_tasks"

    result = run_activation_ranking(
        out_dir=out_dir,
        per_family=1,
        model_name="test-local",
        hook_point="test.layer",
        feature_source="residual_dimensions",
        top_k_features=2,
        task_source="file",
        task_file=task_file,
        task_source_id="unit_calibrated_tasks",
        model_adapter=tiny_model_adapter,
    )

    assert result.n_pairs == 3
    metadata = json.loads((out_dir / "activation_metadata.json").read_text())
    assert metadata["task_source"] == "file"
    assert metadata["task_source_id"] == "unit_calibrated_tasks"
    assert metadata["calibrated_task_count_by_family"] == {
        "property_negation": 1,
        "sentiment_negation": 1,
        "state_negation": 1,
    }
    task_source = json.loads((out_dir / "task_source.json").read_text())
    assert task_source["task_file"] == str(task_file)
