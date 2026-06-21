from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import torch

from self_ground.activations import FeatureActivations
from self_ground.real_behavioral_intervention import run_real_behavioral_sae_intervention
from self_ground.sae_compat import SAECompatibilityResult


class TinyBehavioralTokenizer:
    def to_tokens(self, text: str, prepend_bos: bool = False):
        del prepend_bos
        mapping = {
            " bad": [1],
            " good": [2],
            " safe": [3],
            " dangerous": [4],
            " off": [5],
            " on": [6],
            " great": [7],
            " fun": [8],
            " complex": [9],
            " simple": [10],
            " clean": [11],
            " dirty": [12],
            " closed": [13],
            " open": [14],
            " empty": [15],
            " full": [16],
        }
        return torch.tensor(mapping[text])


class TinyBehavioralModel:
    def __init__(self, adapter) -> None:
        self.adapter = adapter

    def to_tokens(self, text: str, prepend_bos: bool = False):
        return TinyBehavioralTokenizer().to_tokens(text, prepend_bos=prepend_bos)

    def run_with_hooks(self, texts: list[str], fwd_hooks):
        activations = self.adapter._activations_for_texts(texts)
        for _, hook_fn in fwd_hooks:
            activations = hook_fn(activations, hook=None)
        return self.adapter._logits_from_activations(texts, activations)


class TinyNaNHookBehavioralModel(TinyBehavioralModel):
    def run_with_hooks(self, texts: list[str], fwd_hooks):
        activations = self.adapter._activations_for_texts(texts)
        for _, hook_fn in fwd_hooks:
            hook_fn(activations, hook=None)
        return torch.full((len(texts), 1, 17), float("nan"), dtype=torch.float32)


class TinyBehavioralModelAdapter:
    model_name = "test-local"
    device = "cpu"

    def __init__(self) -> None:
        self.model = TinyBehavioralModel(self)

    def _activations_for_texts(self, texts: list[str]) -> torch.Tensor:
        rows = []
        for text in texts:
            lowered = text.lower()
            negation = 1.0 if " not " in lowered else 0.2
            control = 1.0 if " is dangerous" in lowered or " was good" in lowered else 0.4
            rows.append([[negation, control, 0.5, 0.25, 0.2, 0.1, 0.05, 0.04]])
        return torch.tensor(rows, dtype=torch.float32)

    def get_activations(self, texts: list[str], hook_point: str) -> torch.Tensor:
        del hook_point
        return self._activations_for_texts(texts)

    def logits_for_texts(self, texts: list[str]) -> torch.Tensor:
        return self._logits_from_activations(texts, self._activations_for_texts(texts))

    def _logits_from_activations(self, texts: list[str], activations: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((len(texts), 1, 17), dtype=torch.float32)
        for idx, text in enumerate(texts):
            lowered = text.lower()
            signal = activations[idx, -1, 0]
            if "movie" in lowered:
                logits[idx, 0, 1] = signal + 1.0
                logits[idx, 0, 2] = 1.0 - signal
            elif "animal" in lowered:
                logits[idx, 0, 3] = signal + 0.8
                logits[idx, 0, 4] = 1.0 - signal
            else:
                logits[idx, 0, 5] = signal + 0.6
                logits[idx, 0, 6] = 1.0 - signal
        return logits


class TinyNaNHookBehavioralModelAdapter(TinyBehavioralModelAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.model = TinyNaNHookBehavioralModel(self)


class TinyNaNBaselineBehavioralModelAdapter(TinyBehavioralModelAdapter):
    def logits_for_texts(self, texts: list[str]) -> torch.Tensor:
        return torch.full((len(texts), 1, 17), float("nan"), dtype=torch.float32)


class TinyBehavioralSAE:
    def __init__(self) -> None:
        metadata = SimpleNamespace(
            model_name="test-local",
            hook_name="blocks.2.hook_resid_post",
        )
        self.sae = SimpleNamespace(
            cfg=SimpleNamespace(
                metadata=metadata,
                d_in=8,
                d_sae=8,
                architecture="standard",
            )
        )

    def encode(self, activation) -> FeatureActivations:
        values = activation.detach().cpu().numpy() if hasattr(activation, "detach") else activation
        return FeatureActivations(
            values=np.asarray(values, dtype=float),
            feature_ids=[f"sae_{idx}" for idx in range(values.shape[-1])],
        )

    def decode(self, feature_activations: FeatureActivations):
        return np.asarray(feature_activations.values, dtype=float)


def _write_sae_ranking(path) -> None:
    path.mkdir(parents=True)
    with (path / "feature_rankings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_id", "score", "abs_score"])
        writer.writeheader()
        for idx, score in enumerate([10.0, 6.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.04]):
            writer.writerow(
                {
                    "feature_id": f"sae_{idx}",
                    "score": score,
                    "abs_score": abs(score),
                }
            )


def test_phase3_behavioral_intervention_artifacts(tmp_path) -> None:
    ranking_dir = tmp_path / "ranking"
    out_dir = tmp_path / "phase3"
    _write_sae_ranking(ranking_dir)

    result = run_real_behavioral_sae_intervention(
        out_dir=out_dir,
        ranking_dir=ranking_dir,
        per_family=2,
        model_name="test-local",
        hook_point="blocks.2.hook_resid_post",
        sae_release="test-release",
        sae_id="blocks.2.hook_resid_post",
        top_k_features=2,
        baseline_mode="top-vs-random-multiseed",
        random_seeds=[7, 11, 13],
        operations=["ablate"],
        patch_mode="delta",
        model_adapter=TinyBehavioralModelAdapter(),
        sae_adapter=TinyBehavioralSAE(),
    )

    assert result.compatible is True
    assert result.task_validation_passed is True
    assert result.n_rows > 0
    for filename in [
        "config.json",
        "behavioral_tasks.jsonl",
        "behavioral_task_validation.json",
        "excluded_behavioral_tasks.jsonl",
        "compatibility.json",
        "feature_sets.json",
        "baseline_task_scores.jsonl",
        "baseline_task_summary.csv",
        "behavioral_intervention_results.jsonl",
        "behavioral_summary.csv",
        "skipped_behavioral_rows.json",
        "mechanism_report.json",
        "mechanism_report.md",
        "README.md",
    ]:
        assert (out_dir / filename).exists(), filename

    rows = [
        json.loads(line)
        for line in (out_dir / "behavioral_intervention_results.jsonl").read_text().splitlines()
    ]
    assert rows
    baseline_by_id = {
        row["task_id"]: row
        for row in [
            json.loads(line)
            for line in (out_dir / "baseline_task_scores.jsonl").read_text().splitlines()
        ]
    }
    assert {"baseline_contrast", "patched_contrast", "control_signed_delta"} <= set(rows[0])
    assert rows[0]["feature_ids"][0].startswith("sae_")
    assert rows[0]["control_type"] == "matched_non_negation"
    assert rows[0]["target_absolute_delta"] >= 0
    assert "relative_norm_drift_mean" in rows[0]
    assert rows[0]["telemetry_provenance"] == "separate_target_and_control_interventions"
    assert "target_intervention_telemetry" in rows[0]
    assert "control_intervention_telemetry" in rows[0]
    assert rows[0]["baseline_contrast"] == baseline_by_id[rows[0]["task_id"]][
        "baseline_prompt_contrast"
    ]
    assert rows[0]["control_baseline_contrast"] == baseline_by_id[rows[0]["task_id"]][
        "baseline_control_contrast"
    ]
    skipped = json.loads((out_dir / "skipped_behavioral_rows.json").read_text())
    assert skipped == {"n_skipped_rows": 0, "reason_counts": {}, "examples": []}
    report = json.loads((out_dir / "mechanism_report.json").read_text())
    assert report["claim_status"] != "strong_candidate_evidence"
    assert report["row_accounting"]["n_skipped_rows"] == 0


def test_phase3_blocks_on_semantic_mismatch_without_rows(tmp_path) -> None:
    ranking_dir = tmp_path / "ranking"
    out_dir = tmp_path / "blocked"
    _write_sae_ranking(ranking_dir)
    sae = TinyBehavioralSAE()
    sae.sae.cfg.metadata.model_name = "other-model"

    result = run_real_behavioral_sae_intervention(
        out_dir=out_dir,
        ranking_dir=ranking_dir,
        per_family=2,
        model_name="test-local",
        hook_point="blocks.2.hook_resid_post",
        sae_release="test-release",
        sae_id="blocks.2.hook_resid_post",
        top_k_features=2,
        model_adapter=TinyBehavioralModelAdapter(),
        sae_adapter=sae,
    )

    assert result.compatible is False
    assert (out_dir / "compatibility.json").exists()
    assert not (out_dir / "behavioral_intervention_results.jsonl").exists()
    assert (out_dir / "blocker.json").exists()
    report = json.loads((out_dir / "mechanism_report.json").read_text())
    assert "SAE compatibility" in report["blocker_reason"]
    assert "SAE compatibility failed" in (out_dir / "README.md").read_text()


def test_phase3_nonfinite_rows_are_accounted_and_blocked(tmp_path) -> None:
    ranking_dir = tmp_path / "ranking"
    out_dir = tmp_path / "phase3_nonfinite"
    _write_sae_ranking(ranking_dir)

    result = run_real_behavioral_sae_intervention(
        out_dir=out_dir,
        ranking_dir=ranking_dir,
        per_family=2,
        model_name="test-local",
        hook_point="blocks.2.hook_resid_post",
        sae_release="test-release",
        sae_id="blocks.2.hook_resid_post",
        top_k_features=2,
        baseline_mode="top",
        operations=["ablate"],
        patch_mode="delta",
        model_adapter=TinyNaNHookBehavioralModelAdapter(),
        sae_adapter=TinyBehavioralSAE(),
    )

    assert result.compatible is True
    assert result.n_rows == 0
    skipped = json.loads((out_dir / "skipped_behavioral_rows.json").read_text())
    assert skipped["n_skipped_rows"] > 0
    assert skipped["reason_counts"]["nonfinite_row_value"] > 0
    assert (out_dir / "behavioral_intervention_results.jsonl").read_text() == ""
    report = json.loads((out_dir / "mechanism_report.json").read_text())
    assert report["claim_status"] == "blocked"
    assert report["row_accounting"]["all_rows_skipped"] is True
    readme = (out_dir / "README.md").read_text()
    assert "Skipped Rows" in readme
    assert "nonfinite_row_value" in readme


def test_phase3_model_load_failure_writes_blocker_artifacts(tmp_path, monkeypatch) -> None:
    ranking_dir = tmp_path / "ranking"
    out_dir = tmp_path / "model_blocked"
    _write_sae_ranking(ranking_dir)

    class FailingModelAdapter:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError("model cache unavailable")

    import self_ground.model as model_module

    monkeypatch.setattr(model_module, "TransformerLensModelAdapter", FailingModelAdapter)

    result = run_real_behavioral_sae_intervention(
        out_dir=out_dir,
        ranking_dir=ranking_dir,
        per_family=1,
        model_name="test-local",
        hook_point="blocks.2.hook_resid_post",
        sae_release="test-release",
        sae_id="blocks.2.hook_resid_post",
        top_k_features=2,
    )

    assert result.compatible is False
    assert (out_dir / "config.json").exists()
    assert (out_dir / "behavioral_tasks.jsonl").exists()
    assert not (out_dir / "behavioral_task_validation.json").exists()
    assert not (out_dir / "behavioral_intervention_results.jsonl").exists()
    blocker = json.loads((out_dir / "blocker.json").read_text())
    assert blocker["blocker_type"] == "model_load_failure"
    assert blocker["exception_class"] == "RuntimeError"
    assert "model cache unavailable" in blocker["exception_message"]
    report = json.loads((out_dir / "mechanism_report.json").read_text())
    assert report["claim_status"] == "blocked"
    assert "model_load_failure" in report["blocker_reason"]
    assert "No fabricated intervention rows were written" in (
        out_dir / "README.md"
    ).read_text()


def test_phase3_post_compat_sae_load_failure_writes_blocker(tmp_path, monkeypatch) -> None:
    ranking_dir = tmp_path / "ranking"
    out_dir = tmp_path / "sae_load_blocked"
    _write_sae_ranking(ranking_dir)

    def compatible_result(**kwargs):
        return SAECompatibilityResult(
            model_name=kwargs["model_name"],
            hook_point=kwargs["hook_point"],
            sae_release=kwargs["sae_release"],
            sae_id=kwargs["sae_id"],
            activation_shape=[4, 1, 8],
            encoded_shape=[4, 1, 8],
            decoded_shape=[4, 1, 8],
            d_model=8,
            d_sae=8,
            shape_compatible=True,
            metadata_compatible=True,
            reconstruction_compatible=True,
            semantically_compatible=True,
            compatible=True,
            status="ok",
        )

    import self_ground.real_behavioral_intervention as phase3_module
    import self_ground.sae as sae_module

    monkeypatch.setattr(phase3_module, "verify_sae_compatibility", compatible_result)

    def failing_sae_load(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("SAE weights unavailable after compatibility")

    monkeypatch.setattr(sae_module.SAELensAdapter, "from_pretrained", failing_sae_load)

    result = run_real_behavioral_sae_intervention(
        out_dir=out_dir,
        ranking_dir=ranking_dir,
        per_family=2,
        model_name="test-local",
        hook_point="blocks.2.hook_resid_post",
        sae_release="test-release",
        sae_id="blocks.2.hook_resid_post",
        top_k_features=2,
        model_adapter=TinyBehavioralModelAdapter(),
    )

    assert result.compatible is False
    assert (out_dir / "compatibility.json").exists()
    assert not (out_dir / "behavioral_intervention_results.jsonl").exists()
    blocker = json.loads((out_dir / "blocker.json").read_text())
    assert blocker["blocker_type"] == "post_compat_sae_load_failure"
    assert "SAE weights unavailable" in blocker["exception_message"]


def test_phase3_nonfinite_baseline_blocks_before_intervention_rows(tmp_path) -> None:
    ranking_dir = tmp_path / "ranking"
    out_dir = tmp_path / "baseline_blocked"
    _write_sae_ranking(ranking_dir)

    result = run_real_behavioral_sae_intervention(
        out_dir=out_dir,
        ranking_dir=ranking_dir,
        per_family=2,
        model_name="test-local",
        hook_point="blocks.2.hook_resid_post",
        sae_release="test-release",
        sae_id="blocks.2.hook_resid_post",
        top_k_features=2,
        model_adapter=TinyNaNBaselineBehavioralModelAdapter(),
        sae_adapter=TinyBehavioralSAE(),
    )

    assert result.compatible is False
    assert (out_dir / "baseline_validation.json").exists()
    baseline_validation = json.loads((out_dir / "baseline_validation.json").read_text())
    assert baseline_validation["finite"] is False
    assert baseline_validation["n_nonfinite_rows"] > 0
    assert baseline_validation["nonfinite_fields"][0]["task_id"]
    assert baseline_validation["nonfinite_fields"][0]["family"]
    assert not (out_dir / "behavioral_intervention_results.jsonl").exists()
    report = json.loads((out_dir / "mechanism_report.json").read_text())
    assert report["claim_status"] == "blocked"
    assert "baseline" in report["blocker_reason"].lower()
    readme = (out_dir / "README.md").read_text()
    assert "baseline scoring produced non-finite values" in readme
