from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from local_mi_lab.activations import resid_post_hook_name, resolve_position_index
from local_mi_lab.config import selected_layers
from local_mi_lab.models import n_layers
from local_mi_lab.plots import plot_patching_heatmap
from local_mi_lab.tokens import token_id_for_single_token


def patching_effect_size(clean_score: float, corrupt_score: float, patched_score: float) -> float:
    denominator = clean_score - corrupt_score
    if abs(denominator) < 1e-12:
        return 0.0
    return (patched_score - corrupt_score) / denominator


def target_logit_score(model: Any, prompt: str, target_token_id: int) -> float:
    tokens = model.to_tokens(prompt)
    with torch.inference_mode():
        logits = model(tokens)
    return float(logits[0, -1, target_token_id].detach().cpu())


def run_resid_post_patching(
    model: Any,
    clean_prompt: str,
    corrupt_prompt: str,
    target_token: str,
    config: dict[str, Any],
    output_dir: Path,
    full_sweep: bool = False,
) -> pd.DataFrame:
    target_id = token_id_for_single_token(model.tokenizer, target_token)
    if full_sweep:
        layers = list(range(n_layers(model)))
        positions: list[str | int] = ["all"]
    else:
        layers = selected_layers(config, n_layers(model))
        positions = (
            (config.get("task") or {})
            .get("patching", {})
            .get("positions", (config.get("activations") or {}).get("token_positions", ["final"]))
        )

    clean_tokens = model.to_tokens(clean_prompt)
    corrupt_tokens = model.to_tokens(corrupt_prompt)
    clean_score = target_logit_score(model, clean_prompt, target_id)
    corrupt_score = target_logit_score(model, corrupt_prompt, target_id)

    selected_positions: list[int]
    if positions == ["all"]:
        selected_positions = list(range(corrupt_tokens.shape[1]))
    else:
        selected_positions = [
            resolve_position_index(position, corrupt_tokens.shape[1]) for position in positions
        ]

    rows: list[dict[str, Any]] = []
    for layer in tqdm(layers, desc="Activation patching"):
        hook_name = resid_post_hook_name(layer)
        with torch.inference_mode():
            _, clean_cache = model.run_with_cache(clean_tokens, names_filter=[hook_name])
        clean_acts = clean_cache[hook_name].detach()
        for position_index in selected_positions:
            if position_index >= clean_acts.shape[1]:
                continue

            def patch_hook(
                corrupt_act: torch.Tensor,
                hook: Any,
                *,
                clean_acts_for_hook: torch.Tensor = clean_acts,
                position_for_hook: int = position_index,
            ) -> torch.Tensor:
                del hook
                patched = corrupt_act.clone()
                patched[:, position_for_hook, :] = clean_acts_for_hook[:, position_for_hook, :]
                return patched

            with torch.inference_mode():
                patched_logits = model.run_with_hooks(
                    corrupt_tokens,
                    fwd_hooks=[(hook_name, patch_hook)],
                )
            patched_score = float(patched_logits[0, -1, target_id].detach().cpu())
            rows.append(
                {
                    "clean_prompt": clean_prompt,
                    "corrupt_prompt": corrupt_prompt,
                    "target_token": target_token,
                    "target_token_id": target_id,
                    "metric": "target_logit",
                    "patched_layer": layer,
                    "patched_component": "resid_post",
                    "patched_position": position_index,
                    "baseline_clean_score": clean_score,
                    "baseline_corrupt_score": corrupt_score,
                    "patched_score": patched_score,
                    "effect_size": patching_effect_size(clean_score, corrupt_score, patched_score),
                    "full_sweep": full_sweep,
                }
            )

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "patching_results.csv", index=False)
    heatmap = df.pivot_table(
        index="patched_layer",
        columns="patched_position",
        values="effect_size",
        aggfunc="mean",
    )
    heatmap.to_csv(output_dir / "patching_heatmap.csv")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_patching_heatmap(heatmap, figures_dir / "patching_heatmap.png")
    notes = [
        "# Activation Patching Notes",
        "",
        f"Clean prompt: `{clean_prompt}`",
        f"Corrupt prompt: `{corrupt_prompt}`",
        f"Target token: `{target_token}`",
        "Metric: target logit at the final position.",
        f"Full sweep: {full_sweep}",
        "",
        "This is causal intervention evidence for this prompt pair and metric only.",
        "This is not a broad model claim.",
    ]
    (output_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    (output_dir / "patching_metadata.json").write_text(
        json.dumps(
            {
                "clean_prompt": clean_prompt,
                "corrupt_prompt": corrupt_prompt,
                "target_token": target_token,
                "metric": "target_logit",
                "full_sweep": full_sweep,
                "exploratory": full_sweep,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return df
