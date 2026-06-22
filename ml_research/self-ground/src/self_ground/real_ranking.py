from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from self_ground.activations import (
    CONDITIONS,
    POOLING_MODES,
    FeatureActivations,
    PairFeatureActivations,
    RankedFeature,
    condition_text,
    flatten_pair_conditions,
    pool_feature_values,
    rank_candidate_features,
    residual_activations_to_features,
)
from self_ground.data import MinimalPair
from self_ground.engine_boundary import SAE_LENS_BACKEND, TRANSFORMER_LENS_BACKEND
from self_ground.io import (
    read_minimal_pairs,
    write_config,
    write_feature_rankings_csv,
    write_jsonl,
)
from self_ground.negation import generate_negation_pairs
from self_ground.sae_compat import SAECompatibilityResult, verify_sae_compatibility
from self_ground.task_source import (
    count_tasks_by_family,
    load_task_file_with_minimum,
    write_task_source_artifacts,
)


@dataclass(frozen=True)
class ActivationRankingRun:
    out_dir: Path
    n_pairs: int
    n_features: int
    feature_source: str
    top_features: list[str]


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _load_or_generate_pairs(
    *,
    pairs_path: str | Path | None,
    per_family: int,
    seed: int,
) -> list[MinimalPair]:
    if pairs_path is not None:
        return read_minimal_pairs(pairs_path)
    return generate_negation_pairs(per_family=per_family, seed=seed)


def _pairs_from_behavioral_tasks(task_file: str | Path, *, per_family: int) -> list[MinimalPair]:
    tasks, _ = load_task_file_with_minimum(task_file=task_file, min_per_family=per_family)
    pairs: list[MinimalPair] = []
    for task in tasks:
        pairs.append(
            MinimalPair(
                id=task.id,
                domain="behavioral_task_bank",
                concept=task.concept,
                template_family=task.family,
                x_pos=task.prompt,
                x_neg=task.control_prompt,
                x_para=task.prompt,
                x_decoy=task.control_prompt,
                held_constant=["task_id", "family", "concept", "target_foil_tokens"],
                changed_variable="negation_scope_prompt_vs_matched_control_prompt",
                control_purity=1.0,
            )
        )
    return pairs


def _normalise_sae_feature_ids(feature_activations: FeatureActivations) -> FeatureActivations:
    values = _to_numpy(feature_activations.values)
    feature_ids = [f"sae_{idx}" for idx in range(values.shape[-1])]
    return FeatureActivations(values=values, feature_ids=feature_ids)


def feature_activations_from_real_activations(
    activations,
    *,
    feature_source: str,
    pooling: str,
    sae_adapter=None,
) -> FeatureActivations:
    if feature_source == "residual_dimensions":
        return residual_activations_to_features(activations, pooling=pooling)
    if feature_source == "sae":
        if sae_adapter is None:
            raise ValueError("feature_source='sae' requires sae_adapter")
        encoded = sae_adapter.encode(activations)
        encoded = _normalise_sae_feature_ids(encoded)
        pooled = pool_feature_values(encoded.values, pooling=pooling)
        return FeatureActivations(values=pooled, feature_ids=list(encoded.feature_ids))
    raise ValueError("feature_source must be 'residual_dimensions' or 'sae'")


def _pair_features_from_matrix(
    *,
    pairs: list[MinimalPair],
    feature_activations: FeatureActivations,
) -> PairFeatureActivations:
    _, pair_ids, template_families, conditions = flatten_pair_conditions(pairs)
    return PairFeatureActivations(
        pair_ids=pair_ids,
        template_families=template_families,
        conditions=conditions,
        values=feature_activations.values,
        feature_ids=list(feature_activations.feature_ids),
    )


def _top_condition_examples(
    *,
    pairs: list[MinimalPair],
    features: PairFeatureActivations,
    feature_id: str,
    condition: str,
    limit: int,
) -> list[dict[str, Any]]:
    feature_idx = features.feature_index(feature_id)
    pair_by_id = {pair.id: pair for pair in pairs}
    rows: list[dict[str, Any]] = []
    for idx, row_condition in enumerate(features.conditions):
        if row_condition != condition:
            continue
        pair_id = features.pair_ids[idx]
        pair = pair_by_id[pair_id]
        rows.append(
            {
                "pair_id": pair_id,
                "template_family": pair.template_family,
                "condition": condition,
                "text": condition_text(pair, condition),
                "activation": float(features.values[idx, feature_idx]),
            }
        )
    rows.sort(key=lambda row: (-float(row["activation"]), str(row["pair_id"])))
    return rows[:limit]


def build_top_examples(
    *,
    pairs: list[MinimalPair],
    features: PairFeatureActivations,
    rankings: list[RankedFeature],
    top_k_features: int,
    examples_per_condition: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ranking in rankings[:top_k_features]:
        row = {
            "feature_id": ranking.feature_id,
            "score": ranking.score,
        }
        for condition in CONDITIONS:
            key = {
                "x_pos": "top_pos_examples",
                "x_neg": "top_neg_examples",
                "x_para": "top_para_examples",
                "x_decoy": "top_decoy_examples",
            }[condition]
            row[key] = _top_condition_examples(
                pairs=pairs,
                features=features,
                feature_id=ranking.feature_id,
                condition=condition,
                limit=examples_per_condition,
            )
        rows.append(row)
    return rows


def _write_ranking_readme(
    *,
    out_dir: Path,
    model_name: str,
    hook_point: str,
    feature_source: str,
    pooling: str,
    n_pairs: int,
    top_k_features: int,
    compatibility: SAECompatibilityResult | None = None,
    task_source: dict[str, Any] | None = None,
) -> None:
    sae_metadata = ""
    if compatibility is not None:
        sae_metadata = f"""
SAE semantic compatibility:

- declared SAE model: `{compatibility.declared_model}`
- requested model: `{model_name}`
- declared SAE hook point: `{compatibility.declared_hook_point}`
- requested hook point: `{hook_point}`
- shape compatible: `{compatibility.shape_compatible}`
- metadata compatible: `{compatibility.metadata_compatible}`
- reconstruction compatible: `{compatibility.reconstruction_compatible}`
"""
    text = f"""# Real Activation Ranking

- model: `{model_name}`
- hook point: `{hook_point}`
- feature source: `{feature_source}`
- pooling: `{pooling}`
- pairs: `{n_pairs}`
- top-k features: `{top_k_features}`
- task source: `{(task_source or {}).get("task_source", "generated")}`
- task source id: `{(task_source or {}).get("task_source_id", "not set")}`

This run uses real TransformerLens activations. If `feature_source` is
`residual_dimensions`, features are residual stream dimensions named `resid_N`.
If `feature_source` is `sae`, features are encoded by SAELens and named `sae_N`.
{sae_metadata}

Ranking formula:

```text
score = mean(x_pos) + mean(x_para) - mean(x_neg) - mean(x_decoy)
```

This is activation contrast analysis, not a behavioral causal intervention.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def run_activation_ranking(
    *,
    out_dir: str | Path,
    pairs_path: str | Path | None = None,
    per_family: int = 15,
    seed: int = 7,
    model_name: str = "EleutherAI/pythia-70m",
    hook_point: str = "blocks.2.hook_resid_post",
    feature_source: str = "residual_dimensions",
    pooling: str = "final_token",
    top_k_features: int = 50,
    device: str | None = "cpu",
    sae_release: str | None = None,
    sae_id: str | None = None,
    task_source: str = "generated",
    task_file: str | Path | None = None,
    task_source_id: str | None = None,
    model_adapter=None,
    sae_adapter=None,
) -> ActivationRankingRun:
    if per_family < 1:
        raise ValueError("per_family must be >= 1")
    if top_k_features < 1:
        raise ValueError("top_k_features must be >= 1")
    if feature_source not in {"residual_dimensions", "sae"}:
        raise ValueError("feature_source must be 'residual_dimensions' or 'sae'")
    if pooling not in POOLING_MODES:
        raise ValueError("pooling must be 'final_token' or 'mean'")
    if feature_source == "sae" and (not sae_release or not sae_id):
        raise ValueError("feature_source='sae' requires --sae-release and --sae-id")
    if task_source not in {"generated", "file"}:
        raise ValueError("task_source must be 'generated' or 'file'")
    if task_source == "file" and task_file is None:
        raise ValueError("task_source='file' requires task_file")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_tasks = None
    if task_source == "file":
        source_tasks, _ = load_task_file_with_minimum(
            task_file=Path(task_file or ""),
            min_per_family=per_family,
        )
        pairs = _pairs_from_behavioral_tasks(Path(task_file or ""), per_family=per_family)
    else:
        pairs = _load_or_generate_pairs(
            pairs_path=pairs_path,
            per_family=per_family,
            seed=seed,
        )
    texts, _, _, _ = flatten_pair_conditions(pairs)

    if model_adapter is None:
        from self_ground.model import TransformerLensModelAdapter

        model_adapter = TransformerLensModelAdapter(model_name=model_name, device=device)

    if feature_source == "sae" and sae_adapter is None:
        from self_ground.sae import SAELensAdapter

        sae_adapter = SAELensAdapter.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device or getattr(model_adapter, "device", "cpu"),
        )

    sae_compatibility: SAECompatibilityResult | None = None
    if feature_source == "sae":
        sae_compatibility = verify_sae_compatibility(
            model_name=model_name,
            hook_point=hook_point,
            sae_release=sae_release or "",
            sae_id=sae_id or "",
            device=device,
            model_adapter=model_adapter,
            sae_adapter=sae_adapter,
        )
        if not sae_compatibility.compatible:
            raise ValueError(
                "SAE metadata/shape/reconstruction compatibility failed: "
                f"{sae_compatibility.error}"
            )

    activations = model_adapter.get_activations(texts, hook_point=hook_point)
    activation_shape = list(_to_numpy(activations).shape)
    if not activation_shape or activation_shape[0] != len(texts):
        raise ValueError(
            "activation batch count mismatch: "
            f"expected {len(texts)} rows, got shape {activation_shape}"
        )
    feature_activations = feature_activations_from_real_activations(
        activations,
        feature_source=feature_source,
        pooling=pooling,
        sae_adapter=sae_adapter,
    )
    pair_features = _pair_features_from_matrix(
        pairs=pairs,
        feature_activations=feature_activations,
    )
    rankings = rank_candidate_features(pair_features)
    if not rankings:
        raise ValueError("ranking output is empty")
    top_examples = build_top_examples(
        pairs=pairs,
        features=pair_features,
        rankings=rankings,
        top_k_features=top_k_features,
    )

    metadata = {
        "model": model_name,
        "hook_point": hook_point,
        "activation_shape": activation_shape,
        "feature_source": feature_source,
        "pooling": pooling,
        "sae_release": sae_release if feature_source == "sae" else None,
        "sae_id": sae_id if feature_source == "sae" else None,
        "n_pairs": len(pairs),
        "n_conditions": len(texts),
        "n_features": len(feature_activations.feature_ids),
        "engine_backend": TRANSFORMER_LENS_BACKEND,
        "sae_backend": SAE_LENS_BACKEND if feature_source == "sae" else None,
        "claim_eligible": False,
        "task_source": task_source,
        "task_file": str(task_file) if task_file else None,
        "task_source_id": task_source_id,
    }
    task_source_payload = None
    if source_tasks is not None:
        task_source_payload = write_task_source_artifacts(
            out_dir=out_dir,
            task_source=task_source,
            task_file=task_file,
            task_source_id=task_source_id,
            task_bank_calibration_dir=None,
            tasks=source_tasks,
            min_per_family=per_family,
        )
        metadata["calibrated_task_count_by_family"] = count_tasks_by_family(source_tasks)
    if sae_compatibility is not None:
        metadata.update(
            {
                "shape_compatible": sae_compatibility.shape_compatible,
                "metadata_compatible": sae_compatibility.metadata_compatible,
                "reconstruction_compatible": sae_compatibility.reconstruction_compatible,
                "declared_model": sae_compatibility.declared_model,
                "declared_hook_point": sae_compatibility.declared_hook_point,
                "declared_hook_layer": sae_compatibility.declared_hook_layer,
                "declared_hook_type": sae_compatibility.declared_hook_type,
                "requested_hook_layer": sae_compatibility.requested_hook_layer,
                "requested_hook_type": sae_compatibility.requested_hook_type,
                "reconstruction_mse": sae_compatibility.reconstruction_mse,
                "reconstruction_l2_relative": (
                    sae_compatibility.reconstruction_l2_relative
                ),
                "reconstruction_max_abs_error": (
                    sae_compatibility.reconstruction_max_abs_error
                ),
                "metadata_report": sae_compatibility.metadata_report,
            }
        )
    config = {
        **metadata,
        "pairs_path": str(pairs_path) if pairs_path is not None else None,
        "task_source": task_source,
        "task_file": str(task_file) if task_file else None,
        "task_source_id": task_source_id,
        "per_family": per_family,
        "seed": seed,
        "top_k_features": top_k_features,
        "device": device,
        "sae_release": sae_release,
        "sae_id": sae_id,
    }

    write_config(config, out_dir / "config.json")
    write_jsonl(pairs, out_dir / "pairs.jsonl")
    write_config(metadata, out_dir / "activation_metadata.json")
    write_feature_rankings_csv(rankings, out_dir / "feature_rankings.csv")
    write_jsonl(top_examples, out_dir / "top_examples.jsonl")
    _write_ranking_readme(
        out_dir=out_dir,
        model_name=model_name,
        hook_point=hook_point,
        feature_source=feature_source,
        pooling=pooling,
        n_pairs=len(pairs),
        top_k_features=top_k_features,
        compatibility=sae_compatibility,
        task_source=task_source_payload,
    )

    return ActivationRankingRun(
        out_dir=out_dir,
        n_pairs=len(pairs),
        n_features=len(feature_activations.feature_ids),
        feature_source=feature_source,
        top_features=[ranking.feature_id for ranking in rankings[:top_k_features]],
    )
