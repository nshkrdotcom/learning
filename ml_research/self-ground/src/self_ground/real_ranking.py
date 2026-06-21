from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from self_ground.activations import (
    CONDITIONS,
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
from self_ground.io import (
    read_minimal_pairs,
    write_config,
    write_feature_rankings_csv,
    write_jsonl,
)
from self_ground.negation import generate_negation_pairs


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
) -> None:
    text = f"""# Real Activation Ranking

- model: `{model_name}`
- hook point: `{hook_point}`
- feature source: `{feature_source}`
- pooling: `{pooling}`
- pairs: `{n_pairs}`
- top-k features: `{top_k_features}`

This run uses real TransformerLens activations. If `feature_source` is
`residual_dimensions`, features are residual stream dimensions named `resid_N`.
If `feature_source` is `sae`, features are encoded by SAELens and named `sae_N`.

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
    model_adapter=None,
    sae_adapter=None,
) -> ActivationRankingRun:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        if not sae_release or not sae_id:
            raise ValueError("feature_source='sae' requires --sae-release and --sae-id")
        from self_ground.sae import SAELensAdapter

        sae_adapter = SAELensAdapter.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device or getattr(model_adapter, "device", "cpu"),
        )

    activations = model_adapter.get_activations(texts, hook_point=hook_point)
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
    top_examples = build_top_examples(
        pairs=pairs,
        features=pair_features,
        rankings=rankings,
        top_k_features=top_k_features,
    )

    metadata = {
        "model": model_name,
        "hook_point": hook_point,
        "activation_shape": list(_to_numpy(activations).shape),
        "feature_source": feature_source,
        "pooling": pooling,
        "n_pairs": len(pairs),
        "n_conditions": len(texts),
        "n_features": len(feature_activations.feature_ids),
    }
    config = {
        **metadata,
        "pairs_path": str(pairs_path) if pairs_path is not None else None,
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
    )

    return ActivationRankingRun(
        out_dir=out_dir,
        n_pairs=len(pairs),
        n_features=len(feature_activations.feature_ids),
        feature_source=feature_source,
        top_features=[ranking.feature_id for ranking in rankings[:top_k_features]],
    )
