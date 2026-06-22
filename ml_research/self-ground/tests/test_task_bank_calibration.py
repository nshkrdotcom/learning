from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from self_ground.task_bank import CandidateTaskBank, CandidateTaskTemplate

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_phase3_task_bank.py"
SPEC = importlib.util.spec_from_file_location("calibrate_phase3_task_bank", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
calibrate_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibrate_module)
calibrate_task_bank = calibrate_module.calibrate_task_bank


class CalibrationTokenizer:
    def __init__(self) -> None:
        self.ids = {
            " good": 1,
            " bad": 2,
            " light": 3,
            " heavy": 4,
            " stopped": 5,
            " started": 6,
        }

    def to_tokens(self, text: str, prepend_bos: bool = False):
        del prepend_bos
        return torch.tensor([self.ids[text]])


class CalibrationModelAdapter:
    model = CalibrationTokenizer()

    def logits_for_texts(self, texts: list[str]) -> torch.Tensor:
        logits = torch.zeros((len(texts), 1, 16), dtype=torch.float32)
        for idx, text in enumerate(texts):
            if "failcase" in text:
                logits[idx, 0, [1, 3, 5]] = 0.0
                logits[idx, 0, [2, 4, 6]] = 2.0
            else:
                logits[idx, 0, [1, 3, 5]] = 2.0
                logits[idx, 0, [2, 4, 6]] = 0.0
        return logits


def _template(
    *,
    template_id: str,
    family: str,
    marker: str,
    target: str,
    foil: str,
) -> CandidateTaskTemplate:
    return CandidateTaskTemplate(
        template_id=template_id,
        family=family,
        prompt_template=f"The {marker} was not {foil.strip()}. The {marker} was",
        target_token=target,
        foil_token=foil,
        control_prompt_template=f"The {marker} was {foil.strip()}. The {marker} was",
        metadata={"template_family": "test_template"},
    )


def _bank(path: Path, *, include_property_pass: bool = True) -> None:
    templates = [
        _template(
            template_id="sentiment_pass",
            family="sentiment_negation",
            marker="movie",
            target=" good",
            foil=" bad",
        ),
        _template(
            template_id="sentiment_fail",
            family="sentiment_negation",
            marker="failcase_movie",
            target=" good",
            foil=" bad",
        ),
        _template(
            template_id="state_pass",
            family="state_negation",
            marker="machine",
            target=" stopped",
            foil=" started",
        ),
    ]
    if include_property_pass:
        templates.append(
            _template(
                template_id="property_pass",
                family="property_negation",
                marker="object",
                target=" light",
                foil=" heavy",
            )
        )
    else:
        templates.append(
            _template(
                template_id="property_fail",
                family="property_negation",
                marker="failcase_object",
                target=" light",
                foil=" heavy",
            )
        )
    payload = CandidateTaskBank(
        model_name="test-local",
        families=["sentiment_negation", "property_negation", "state_negation"],
        templates=templates,
    )
    path.write_text(json.dumps(payload.model_dump(mode="json")), encoding="utf-8")


def test_task_bank_calibration_writes_kept_and_rejected_artifacts(tmp_path) -> None:
    bank_path = tmp_path / "bank.json"
    out_dir = tmp_path / "calibration"
    _bank(bank_path)

    summary = calibrate_task_bank(
        task_bank_path=bank_path,
        out_dir=out_dir,
        model_name="test-local",
        device="cpu",
        min_baseline_margin=0.1,
        min_per_family=1,
        model_adapter=CalibrationModelAdapter(),
    )

    assert summary["passes_minimum"] is True
    assert summary["kept_by_family"] == {
        "property_negation": 1,
        "sentiment_negation": 1,
        "state_negation": 1,
    }
    assert (out_dir / "candidate_baseline_scores.jsonl").exists()
    assert (out_dir / "calibrated_behavioral_tasks.jsonl").exists()
    exclusions = (out_dir / "calibrated_excluded_behavioral_tasks.jsonl").read_text()
    assert "baseline_wrong_direction" in exclusions


def test_task_bank_calibration_fails_closed_when_required_family_underfills(tmp_path) -> None:
    bank_path = tmp_path / "bank.json"
    out_dir = tmp_path / "calibration_blocked"
    _bank(bank_path, include_property_pass=False)

    summary = calibrate_task_bank(
        task_bank_path=bank_path,
        out_dir=out_dir,
        model_name="test-local",
        device="cpu",
        min_baseline_margin=0.1,
        min_per_family=1,
        model_adapter=CalibrationModelAdapter(),
    )

    assert summary["passes_minimum"] is False
    assert summary["kept_by_family"]["property_negation"] == 0
    blocker = json.loads((out_dir / "blocker.json").read_text())
    assert blocker["blocker_type"] == "task_bank_calibration_failed"
