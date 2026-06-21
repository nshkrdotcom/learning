from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from self_ground.activations import collect_pair_feature_activations, rank_candidate_features
from self_ground.data import ExperimentResult
from self_ground.interventions import evaluate_feature_intervention
from self_ground.io import (
    read_minimal_pairs,
    write_config,
    write_feature_rankings_csv,
    write_jsonl,
    write_summary_csv,
)
from self_ground.metrics import MetricWeights


@dataclass(frozen=True)
class RunResult:
    out_dir: Path
    n_pairs: int
    top_features: list[str]


def _build_real_adapters(
    *,
    model_name: str,
    sae_release: str | None,
    sae_id: str | None,
    device: str | None,
):
    if not sae_release or not sae_id:
        raise ValueError("real runs require --sae-release and --sae-id")
    from self_ground.model import TransformerLensModelAdapter
    from self_ground.sae import SAELensAdapter

    model = TransformerLensModelAdapter(model_name=model_name, device=device)
    sae = SAELensAdapter.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device or model.device,
    )
    return model, sae


def _write_run_readme(
    *,
    out_dir: Path,
    model_name: str,
    sae_release: str | None,
    sae_id: str | None,
    n_pairs: int,
    layer: str,
    top_k_features: int,
) -> None:
    sae_label = f"{sae_release} / {sae_id}" if sae_release and sae_id else "injected test adapter"
    text = f"""# SELF-GROUND Negation Run

- model: `{model_name}`
- SAE: `{sae_label}`
- pairs: `{n_pairs}`
- layer: `{layer}`
- top-k features: `{top_k_features}`

Metric definitions:

- ranking = mean(x_pos) + mean(x_para) - mean(x_neg) - mean(x_decoy)
- necessity = (delta_ablate_pos + delta_ablate_para) - (delta_ablate_neg + delta_ablate_decoy)
- sufficiency = amplify/patch target shift + rescue after ablation - collateral shift
- specificity = delta_pos + delta_para - delta_neg - delta_decoy
- cleanliness = weighted necessity + sufficiency + specificity - collateral - mechanism size penalty

Limitations:

This v0 writes real SAE feature-space deltas. It does not yet reinject decoded activations into
TransformerLens, so intervention rows are proxy causal evidence rather than completed behavioral
interventions.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def run_negation_experiment(
    *,
    pairs_path: str | Path,
    out_dir: str | Path,
    model_name: str = "gpt2-small",
    layer: str = "blocks.8.hook_resid_post",
    top_k_features: int = 20,
    sae_release: str | None = None,
    sae_id: str | None = None,
    device: str | None = None,
    model_adapter=None,
    sae_adapter=None,
    weights: MetricWeights | None = None,
) -> RunResult:
    pairs_path = Path(pairs_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = read_minimal_pairs(pairs_path)
    if model_adapter is None or sae_adapter is None:
        model_adapter, sae_adapter = _build_real_adapters(
            model_name=model_name,
            sae_release=sae_release,
            sae_id=sae_id,
            device=device,
        )

    config = {
        "model": model_name,
        "sae_release": sae_release,
        "sae_id": sae_id,
        "layer": layer,
        "top_k_features": top_k_features,
        "n_pairs": len(pairs),
        "intervention_mode": "sae_feature_space_proxy",
    }
    write_config(config, out_dir / "config.json")
    write_jsonl(pairs, out_dir / "pairs.jsonl")

    features = collect_pair_feature_activations(
        pairs,
        model=model_adapter,
        sae=sae_adapter,
        layer=layer,
    )
    rankings = rank_candidate_features(features)
    top_rankings = rankings[:top_k_features]
    write_feature_rankings_csv(rankings, out_dir / "feature_rankings.csv")

    results: list[ExperimentResult] = []
    for ranking in top_rankings:
        for pair in pairs:
            effect = evaluate_feature_intervention(
                features,
                pair.id,
                ranking.feature_id,
                sae_adapter,
                weights=weights,
            )
            results.append(
                ExperimentResult(
                    pair_id=pair.id,
                    feature_id=ranking.feature_id,
                    template_family=pair.template_family,
                    metrics=effect,
                    metadata={
                        "layer": layer,
                        "ranking_score": ranking.score,
                        "intervention_mode": "sae_feature_space_proxy",
                    },
                )
            )

    write_jsonl(results, out_dir / "intervention_results.jsonl")
    write_summary_csv(results, out_dir / "summary.csv")
    _write_run_readme(
        out_dir=out_dir,
        model_name=model_name,
        sae_release=sae_release,
        sae_id=sae_id,
        n_pairs=len(pairs),
        layer=layer,
        top_k_features=top_k_features,
    )

    return RunResult(
        out_dir=out_dir,
        n_pairs=len(pairs),
        top_features=[ranking.feature_id for ranking in top_rankings],
    )
