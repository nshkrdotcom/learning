from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from self_ground.activations import CONDITIONS, condition_text
from self_ground.data import MinimalPair
from self_ground.io import read_minimal_pairs, write_config, write_jsonl
from self_ground.negation import generate_negation_pairs
from self_ground.real_ranking import run_activation_ranking
from self_ground.residual_intervention import run_residual_intervention_logits

DEFAULT_POSITIVE_TOKENS = [" not", " no", " never"]
DEFAULT_NEGATIVE_TOKENS = [" often", " always", " sometimes"]


@dataclass(frozen=True)
class ResidualInterventionRun:
    out_dir: Path
    n_pairs: int
    n_features: int
    operation: str
    top_features: list[str]


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


def _read_top_residual_features(ranking_dir: Path, top_k_features: int) -> list[str]:
    ranking_path = ranking_dir / "feature_rankings.csv"
    if not ranking_path.exists():
        raise ValueError(f"ranking file does not exist: {ranking_path}")

    with ranking_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"ranking file is empty: {ranking_path}")

    feature_ids = [row["feature_id"] for row in rows[:top_k_features]]
    invalid = [feature_id for feature_id in feature_ids if not feature_id.startswith("resid_")]
    if invalid:
        raise ValueError(
            "Phase 1 residual intervention requires residual feature ids; "
            f"got invalid ids: {invalid}"
        )
    return feature_ids


def _contrast_from_logits(
    *,
    model_adapter,
    logits: torch.Tensor,
    positive_tokens: list[str],
    negative_tokens: list[str],
) -> list[float]:
    pos_ids = model_adapter.token_ids_for_strings(positive_tokens)
    neg_ids = model_adapter.token_ids_for_strings(negative_tokens)
    final_logits = logits[:, -1, :]
    pos = final_logits[:, pos_ids].mean(dim=-1)
    neg = final_logits[:, neg_ids].mean(dim=-1)
    return (pos - neg).detach().cpu().tolist()


def _mean_abs(values: list[float]) -> float:
    return sum(abs(value) for value in values) / len(values)


def _condition_texts(pair: MinimalPair) -> list[str]:
    return [condition_text(pair, condition) for condition in CONDITIONS]


def _write_intervention_summary(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "feature_set",
        "operation",
        "n_pairs",
        "negation_specific_delta_mean",
        "control_delta_mean",
        "specificity_score_mean",
    ]
    if rows:
        n_pairs = len(rows)
        feature_set = "+".join(rows[0]["feature_ids"])
        operation = rows[0]["operation"]
        summary = {
            "feature_set": feature_set,
            "operation": operation,
            "n_pairs": n_pairs,
            "negation_specific_delta_mean": sum(
                row["negation_specific_delta"] for row in rows
            )
            / n_pairs,
            "control_delta_mean": sum(row["control_delta"] for row in rows) / n_pairs,
            "specificity_score_mean": sum(row["specificity_score"] for row in rows)
            / n_pairs,
        }
    else:
        summary = {
            "feature_set": "",
            "operation": "",
            "n_pairs": 0,
            "negation_specific_delta_mean": 0.0,
            "control_delta_mean": 0.0,
            "specificity_score_mean": 0.0,
        }

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)


def _write_run_readme(
    *,
    out_dir: Path,
    model_name: str,
    hook_point: str,
    feature_ids: list[str],
    operation: str,
    n_pairs: int,
) -> None:
    text = f"""# Real Residual Intervention

- model: `{model_name}`
- hook point: `{hook_point}`
- operation: `{operation}`
- feature ids: `{", ".join(feature_ids)}`
- pairs: `{n_pairs}`

This run performs a real TransformerLens residual intervention. It patches raw
residual stream dimensions at the selected hook point, reruns the model, and
measures logit-contrast changes.

It is not an SAE decoded intervention. Raw residual dimensions are
basis-dependent and are not directly interpretable as sparse features.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _validate_inputs(
    *,
    per_family: int,
    top_k_features: int,
    operation: str,
    factor: float,
) -> None:
    if per_family < 1:
        raise ValueError("per_family must be >= 1")
    if top_k_features < 1:
        raise ValueError("top_k_features must be >= 1")
    if operation not in {"zero", "amplify"}:
        raise ValueError("operation must be 'zero' or 'amplify'")
    if operation == "amplify" and factor == 1.0:
        raise ValueError("operation='amplify' requires factor != 1.0")


def run_real_residual_intervention(
    *,
    out_dir: str | Path,
    ranking_dir: str | Path | None = None,
    pairs_path: str | Path | None = None,
    per_family: int = 15,
    seed: int = 7,
    model_name: str = "EleutherAI/pythia-70m",
    hook_point: str = "blocks.2.hook_resid_post",
    top_k_features: int = 5,
    operation: Literal["zero", "amplify"] = "zero",
    factor: float = 0.0,
    device: str | None = "cpu",
    positive_tokens: list[str] | None = None,
    negative_tokens: list[str] | None = None,
    model_adapter=None,
) -> ResidualInterventionRun:
    _validate_inputs(
        per_family=per_family,
        top_k_features=top_k_features,
        operation=operation,
        factor=factor,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = Path(ranking_dir) if ranking_dir is not None else None
    source = "ranking_dir" if ranking_path is not None else "computed"

    if ranking_path is None:
        computed_ranking_dir = out_dir / "_computed_activation_ranking"
        run_activation_ranking(
            out_dir=computed_ranking_dir,
            pairs_path=pairs_path,
            per_family=per_family,
            seed=seed,
            model_name=model_name,
            hook_point=hook_point,
            feature_source="residual_dimensions",
            pooling="final_token",
            top_k_features=top_k_features,
            device=device,
            model_adapter=model_adapter,
        )
        ranking_path = computed_ranking_dir

    feature_ids = _read_top_residual_features(ranking_path, top_k_features)
    pairs = _load_or_generate_pairs(
        ranking_dir=ranking_path,
        pairs_path=pairs_path,
        per_family=per_family,
        seed=seed,
    )
    positive = positive_tokens or DEFAULT_POSITIVE_TOKENS
    negative = negative_tokens or DEFAULT_NEGATIVE_TOKENS

    if model_adapter is None:
        from self_ground.model import TransformerLensModelAdapter

        model_adapter = TransformerLensModelAdapter(model_name=model_name, device=device)

    result_rows: list[dict[str, Any]] = []
    for pair in pairs:
        texts = _condition_texts(pair)
        baseline_values = model_adapter.logit_contrast(
            texts,
            positive=positive,
            negative=negative,
        ).detach().cpu().tolist()
        patched_logits = run_residual_intervention_logits(
            model_adapter,
            texts,
            hook_point,
            feature_ids,
            operation,
            factor=factor,
            token_position=-1,
        )
        patched_values = _contrast_from_logits(
            model_adapter=model_adapter,
            logits=patched_logits,
            positive_tokens=positive,
            negative_tokens=negative,
        )

        baseline = dict(zip(CONDITIONS, baseline_values, strict=True))
        patched = dict(zip(CONDITIONS, patched_values, strict=True))
        delta = {
            condition: patched[condition] - baseline[condition]
            for condition in CONDITIONS
        }
        negation_specific_delta = _mean_abs([delta["x_pos"], delta["x_para"]])
        control_delta = _mean_abs([delta["x_neg"], delta["x_decoy"]])
        specificity_score = negation_specific_delta - control_delta

        result_rows.append(
            {
                "pair_id": pair.id,
                "template_family": pair.template_family,
                "feature_ids": feature_ids,
                "operation": operation,
                "hook_point": hook_point,
                "positive_tokens": positive,
                "negative_tokens": negative,
                "baseline": baseline,
                "patched": patched,
                "delta": delta,
                "negation_specific_delta": negation_specific_delta,
                "control_delta": control_delta,
                "specificity_score": specificity_score,
            }
        )

    config = {
        "model": model_name,
        "hook_point": hook_point,
        "top_k_features": top_k_features,
        "operation": operation,
        "factor": factor,
        "device": device,
        "positive_tokens": positive,
        "negative_tokens": negative,
        "ranking_dir": str(ranking_path),
        "pairs_path": str(pairs_path) if pairs_path is not None else None,
        "per_family": per_family,
        "seed": seed,
        "n_pairs": len(pairs),
        "result_type": "real_residual_intervention",
    }
    selected_features = {
        "source": source,
        "ranking_dir": str(ranking_path),
        "feature_ids": feature_ids,
        "top_k_features": top_k_features,
    }

    write_config(config, out_dir / "config.json")
    write_config(selected_features, out_dir / "selected_features.json")
    write_jsonl(result_rows, out_dir / "intervention_results.jsonl")
    _write_intervention_summary(result_rows, out_dir / "summary.csv")
    _write_run_readme(
        out_dir=out_dir,
        model_name=model_name,
        hook_point=hook_point,
        feature_ids=feature_ids,
        operation=operation,
        n_pairs=len(pairs),
    )

    return ResidualInterventionRun(
        out_dir=out_dir,
        n_pairs=len(pairs),
        n_features=len(feature_ids),
        operation=operation,
        top_features=feature_ids,
    )
