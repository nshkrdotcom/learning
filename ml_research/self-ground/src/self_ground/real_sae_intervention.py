from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from self_ground.activations import CONDITIONS, condition_text
from self_ground.data import MinimalPair
from self_ground.io import read_minimal_pairs, write_config, write_jsonl
from self_ground.logit_scoring import (
    condition_dict,
    contrast_from_logits,
    delta_dict,
    delta_metrics,
)
from self_ground.negation import generate_negation_pairs
from self_ground.real_ranking import run_activation_ranking
from self_ground.real_residual_intervention import (
    DEFAULT_NEGATIVE_TOKENS,
    DEFAULT_POSITIVE_TOKENS,
)
from self_ground.sae_compat import SAECompatibilityResult, verify_sae_compatibility
from self_ground.sae_interventions import run_sae_decoded_intervention_logits


@dataclass(frozen=True)
class SAEInterventionRun:
    out_dir: Path
    n_pairs: int
    n_features: int
    operation: str
    patch_mode: str
    top_features: list[str]
    compatible: bool


def _condition_texts(pair: MinimalPair) -> list[str]:
    return [condition_text(pair, condition) for condition in CONDITIONS]


def _load_or_generate_pairs(
    *,
    ranking_dir: Path | None,
    pairs_path: str | Path | None,
    per_family: int,
    seed: int,
) -> list[MinimalPair]:
    if pairs_path is not None:
        return read_minimal_pairs(pairs_path)
    if ranking_dir is not None and (ranking_dir / "pairs.jsonl").exists():
        return read_minimal_pairs(ranking_dir / "pairs.jsonl")
    return generate_negation_pairs(per_family=per_family, seed=seed)


def _read_top_sae_features(ranking_dir: Path, top_k_features: int) -> list[str]:
    ranking_path = ranking_dir / "feature_rankings.csv"
    if not ranking_path.exists():
        raise ValueError(f"ranking file does not exist: {ranking_path}")
    with ranking_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"ranking file is empty: {ranking_path}")
    feature_ids = [row["feature_id"] for row in rows[:top_k_features]]
    invalid = [feature_id for feature_id in feature_ids if not feature_id.startswith("sae_")]
    if invalid:
        raise ValueError(
            "Phase 2 SAE intervention requires SAE feature ids; "
            f"got invalid ids: {invalid}"
        )
    return feature_ids


def _write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "feature_set",
        "operation",
        "patch_mode",
        "n_pairs",
        "signed_negation_delta_mean",
        "signed_control_delta_mean",
        "absolute_negation_delta_mean",
        "absolute_control_delta_mean",
        "signed_specificity_score_mean",
        "absolute_specificity_score_mean",
    ]
    n_pairs = len(rows)
    summary = {
        "feature_set": "+".join(rows[0]["feature_ids"]) if rows else "",
        "operation": rows[0]["operation"] if rows else "",
        "patch_mode": rows[0]["patch_mode"] if rows else "",
        "n_pairs": n_pairs,
        "signed_negation_delta_mean": 0.0,
        "signed_control_delta_mean": 0.0,
        "absolute_negation_delta_mean": 0.0,
        "absolute_control_delta_mean": 0.0,
        "signed_specificity_score_mean": 0.0,
        "absolute_specificity_score_mean": 0.0,
    }
    if rows:
        for key in [
            "signed_negation_delta_mean",
            "signed_control_delta_mean",
            "absolute_negation_delta_mean",
            "absolute_control_delta_mean",
            "signed_specificity_score",
            "absolute_specificity_score",
        ]:
            output_key = f"{key}_mean" if key.endswith("_score") else key
            summary[output_key] = sum(row[key] for row in rows) / n_pairs

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)


def _write_readme(
    *,
    out_dir: Path,
    compatible: bool,
    model_name: str,
    hook_point: str,
    sae_release: str,
    sae_id: str,
    operation: str,
    patch_mode: str,
    feature_ids: list[str],
    compatibility: SAECompatibilityResult,
    error: str | None = None,
) -> None:
    if not compatible:
        corrected_model_command = ""
        if compatibility.declared_model:
            if "/" in compatibility.declared_model:
                corrected_model = compatibility.declared_model
            elif compatibility.declared_model.startswith("pythia-"):
                corrected_model = f"EleutherAI/{compatibility.declared_model}"
            else:
                corrected_model = compatibility.declared_model
            corrected_model_command = f"""
If the declared SAE model is intended, rerun with:

```bash
uv run python scripts/check_sae_compatibility.py \\
  --model {corrected_model} \\
  --hook-point {hook_point} \\
  --sae-release {sae_release} \\
  --sae-id {sae_id} \\
  --device cpu \\
  --out runs/check_sae_compatibility.json
```
"""
        text = f"""# SAE Decoded Intervention Blocker

The intervention was not run because SAE compatibility failed.

- requested model: `{model_name}`
- requested hook point: `{hook_point}`
- SAE release: `{sae_release}`
- SAE id: `{sae_id}`
- declared SAE model: `{compatibility.declared_model}`
- declared SAE hook point: `{compatibility.declared_hook_point}`
- declared SAE hook layer: `{compatibility.declared_hook_layer}`
- declared SAE hook type: `{compatibility.declared_hook_type}`
- shape compatible: `{compatibility.shape_compatible}`
- metadata compatible: `{compatibility.metadata_compatible}`
- reconstruction compatible: `{compatibility.reconstruction_compatible}`
- error: `{error}`

Rerun compatibility with:

```bash
uv run python scripts/check_sae_compatibility.py \\
  --model {model_name} \\
  --hook-point {hook_point} \\
  --sae-release {sae_release} \\
  --sae-id {sae_id} \\
  --device cpu \\
  --out runs/check_sae_compatibility.json
```
{corrected_model_command}
No intervention rows were written.
"""
    else:
        text = f"""# Real SAE Decoded Intervention

- model: `{model_name}`
- hook point: `{hook_point}`
- SAE release: `{sae_release}`
- SAE id: `{sae_id}`
- operation: `{operation}`
- patch mode: `{patch_mode}`
- selected features: `{", ".join(feature_ids)}`
- shape compatible: `{compatibility.shape_compatible}`
- metadata compatible: `{compatibility.metadata_compatible}`
- reconstruction compatible: `{compatibility.reconstruction_compatible}`
- declared SAE model: `{compatibility.declared_model}`
- requested model: `{model_name}`
- declared SAE hook point: `{compatibility.declared_hook_point}`
- requested hook point: `{hook_point}`
- reconstruction MSE: `{compatibility.reconstruction_mse}`
- reconstruction relative L2: `{compatibility.reconstruction_l2_relative}`
- reconstruction max absolute error: `{compatibility.reconstruction_max_abs_error}`

This run encodes real hook activations with SAELens, modifies selected SAE
features, decodes back to residual space, patches the real TransformerLens
model, reruns logits, and measures condition-wise logit-contrast deltas.

Metric definitions:

- signed_negation_delta_mean = mean(delta[x_pos], delta[x_para])
- signed_control_delta_mean = mean(delta[x_neg], delta[x_decoy])
- absolute_negation_delta_mean = mean(abs(delta[x_pos]), abs(delta[x_para]))
- absolute_control_delta_mean = mean(abs(delta[x_neg]), abs(delta[x_decoy]))
- signed_specificity_score = signed_negation_delta_mean - signed_control_delta_mean
- absolute_specificity_score = absolute_negation_delta_mean - absolute_control_delta_mean

Compatibility here means semantic metadata match plus shape compatibility plus
finite reconstruction sanity metrics. Limitations: this is a decoded SAE feature
intervention for the configured SAE only. It does not establish broad mechanism
discovery or model introspection.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _validate_inputs(
    *,
    per_family: int,
    top_k_features: int,
    operation: str,
    factor: float,
    patch_mode: str,
    sae_release: str,
    sae_id: str,
) -> None:
    if not sae_release or not sae_id:
        raise ValueError("sae_release and sae_id are required")
    if per_family < 1:
        raise ValueError("per_family must be >= 1")
    if top_k_features < 1:
        raise ValueError("top_k_features must be >= 1")
    if operation not in {"ablate", "amplify"}:
        raise ValueError("operation must be 'ablate' or 'amplify'")
    if operation == "amplify" and factor == 1.0:
        raise ValueError("operation='amplify' requires factor != 1.0")
    if patch_mode not in {"replace", "delta"}:
        raise ValueError("patch_mode must be 'replace' or 'delta'")


def _verify_with_loaded_adapters(
    *,
    model_adapter,
    sae_adapter,
    model_name: str,
    hook_point: str,
    sae_release: str,
    sae_id: str,
) -> SAECompatibilityResult:
    return verify_sae_compatibility(
        model_name=model_name,
        hook_point=hook_point,
        sae_release=sae_release,
        sae_id=sae_id,
        model_adapter=model_adapter,
        sae_adapter=sae_adapter,
    )


def _remove_blocked_outputs(out_dir: Path) -> None:
    for filename in [
        "selected_features.json",
        "intervention_results.jsonl",
        "summary.csv",
    ]:
        path = out_dir / filename
        if path.exists():
            path.unlink()


def run_real_sae_intervention(
    *,
    out_dir: str | Path,
    ranking_dir: str | Path | None = None,
    pairs_path: str | Path | None = None,
    per_family: int = 15,
    seed: int = 7,
    model_name: str = "EleutherAI/pythia-70m-deduped",
    hook_point: str = "blocks.2.hook_resid_post",
    sae_release: str = "",
    sae_id: str = "",
    top_k_features: int = 5,
    operation: Literal["ablate", "amplify"] = "ablate",
    factor: float = 1.0,
    patch_mode: Literal["replace", "delta"] = "delta",
    token_position: int | None = -1,
    device: str | None = "cpu",
    positive_tokens: list[str] | None = None,
    negative_tokens: list[str] | None = None,
    model_adapter=None,
    sae_adapter=None,
) -> SAEInterventionRun:
    _validate_inputs(
        per_family=per_family,
        top_k_features=top_k_features,
        operation=operation,
        factor=factor,
        patch_mode=patch_mode,
        sae_release=sae_release,
        sae_id=sae_id,
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    config = {
        "model": model_name,
        "hook_point": hook_point,
        "sae_release": sae_release,
        "sae_id": sae_id,
        "top_k_features": top_k_features,
        "operation": operation,
        "factor": factor,
        "patch_mode": patch_mode,
        "token_position": token_position,
        "device": device,
        "result_type": "real_sae_decoded_intervention",
    }
    write_config(config, out_path / "config.json")

    if model_adapter is None or sae_adapter is None:
        compatibility = verify_sae_compatibility(
            model_name=model_name,
            hook_point=hook_point,
            sae_release=sae_release,
            sae_id=sae_id,
            device=device,
        )
    else:
        compatibility = _verify_with_loaded_adapters(
            model_adapter=model_adapter,
            sae_adapter=sae_adapter,
            model_name=model_name,
            hook_point=hook_point,
            sae_release=sae_release,
            sae_id=sae_id,
        )
    write_config(compatibility.model_dump(), out_path / "compatibility.json")
    if not compatibility.compatible:
        _write_readme(
            out_dir=out_path,
            compatible=False,
            model_name=model_name,
            hook_point=hook_point,
            sae_release=sae_release,
            sae_id=sae_id,
            operation=operation,
            patch_mode=patch_mode,
            feature_ids=[],
            compatibility=compatibility,
            error=compatibility.error,
        )
        _remove_blocked_outputs(out_path)
        return SAEInterventionRun(
            out_dir=out_path,
            n_pairs=0,
            n_features=0,
            operation=operation,
            patch_mode=patch_mode,
            top_features=[],
            compatible=False,
        )

    if model_adapter is None:
        from self_ground.model import TransformerLensModelAdapter

        model_adapter = TransformerLensModelAdapter(model_name=model_name, device=device)
    if sae_adapter is None:
        from self_ground.sae import SAELensAdapter

        sae_adapter = SAELensAdapter.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device or model_adapter.device,
        )

    ranking_path = Path(ranking_dir) if ranking_dir is not None else None
    source = "ranking_dir" if ranking_path is not None else "computed"
    if ranking_path is None:
        ranking_path = out_path / "_computed_sae_ranking"
        run_activation_ranking(
            out_dir=ranking_path,
            pairs_path=pairs_path,
            per_family=per_family,
            seed=seed,
            model_name=model_name,
            hook_point=hook_point,
            feature_source="sae",
            pooling="final_token",
            top_k_features=top_k_features,
            device=device,
            sae_release=sae_release,
            sae_id=sae_id,
            model_adapter=model_adapter,
            sae_adapter=sae_adapter,
        )

    feature_ids = _read_top_sae_features(ranking_path, top_k_features)
    selected_features = {
        "source": source,
        "ranking_dir": str(ranking_path),
        "feature_ids": feature_ids,
        "top_k_features": top_k_features,
    }
    write_config(selected_features, out_path / "selected_features.json")

    pairs = _load_or_generate_pairs(
        ranking_dir=ranking_path,
        pairs_path=pairs_path,
        per_family=per_family,
        seed=seed,
    )
    positive = positive_tokens or DEFAULT_POSITIVE_TOKENS
    negative = negative_tokens or DEFAULT_NEGATIVE_TOKENS
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        texts = _condition_texts(pair)
        baseline = condition_dict(
            model_adapter.logit_contrast(texts, positive=positive, negative=negative)
            .detach()
            .cpu()
            .tolist()
        )
        patched_logits = run_sae_decoded_intervention_logits(
            model_adapter,
            sae_adapter,
            texts,
            hook_point,
            feature_ids,
            operation,
            factor=factor,
            token_position=token_position,
            patch_mode=patch_mode,
        )
        patched = condition_dict(
            contrast_from_logits(
                model_adapter=model_adapter,
                logits=patched_logits,
                positive_tokens=positive,
                negative_tokens=negative,
            )
        )
        delta = delta_dict(baseline, patched)
        metrics = delta_metrics(delta)
        rows.append(
            {
                "pair_id": pair.id,
                "template_family": pair.template_family,
                "feature_ids": feature_ids,
                "operation": operation,
                "patch_mode": patch_mode,
                "hook_point": hook_point,
                "sae_release": sae_release,
                "sae_id": sae_id,
                "positive_tokens": positive,
                "negative_tokens": negative,
                "baseline": baseline,
                "patched": patched,
                "delta": delta,
                **metrics,
            }
        )

    write_jsonl(rows, out_path / "intervention_results.jsonl")
    _write_summary(rows, out_path / "summary.csv")
    _write_readme(
        out_dir=out_path,
        compatible=True,
        model_name=model_name,
        hook_point=hook_point,
        sae_release=sae_release,
        sae_id=sae_id,
        operation=operation,
        patch_mode=patch_mode,
        feature_ids=feature_ids,
        compatibility=compatibility,
    )
    return SAEInterventionRun(
        out_dir=out_path,
        n_pairs=len(pairs),
        n_features=len(feature_ids),
        operation=operation,
        patch_mode=patch_mode,
        top_features=feature_ids,
        compatible=True,
    )
