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


def attn_out_hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_attn_out"


def component_hook_name(component: str, layer: int) -> str:
    if component == "resid_post":
        return resid_post_hook_name(layer)
    if component == "attn_out":
        return attn_out_hook_name(layer)
    raise ValueError(f"Unsupported patching component {component!r}")


def patching_effect_size(
    clean_score: float,
    corrupt_score: float,
    patched_score: float,
) -> float | None:
    denominator = clean_score - corrupt_score
    if abs(denominator) < 1e-12:
        return None
    return (patched_score - corrupt_score) / denominator


def patching_effect(
    clean_score: float,
    corrupt_score: float,
    patched_score: float,
) -> dict[str, float | str | None]:
    effect_size = patching_effect_size(clean_score, corrupt_score, patched_score)
    if effect_size is None:
        return {"effect_size": None, "effect_size_status": "denominator_zero"}
    return {"effect_size": effect_size, "effect_size_status": "ok"}


def validate_clean_corrupt_token_lengths(
    clean_seq_len: int,
    corrupt_seq_len: int,
    allow_length_mismatch: bool,
) -> None:
    if clean_seq_len == corrupt_seq_len:
        return
    if allow_length_mismatch:
        return
    raise ValueError(
        "clean_prompt and corrupt_prompt must tokenize to the same length unless "
        "--allow-length-mismatch is passed. "
        f"Got clean length {clean_seq_len}, corrupt length {corrupt_seq_len}."
    )


def resolve_patch_positions(
    positions: list[str | int],
    clean_seq_len: int,
    corrupt_seq_len: int,
    allow_length_mismatch: bool,
) -> list[int]:
    validate_clean_corrupt_token_lengths(clean_seq_len, corrupt_seq_len, allow_length_mismatch)
    if positions == ["all"]:
        return list(range(min(clean_seq_len, corrupt_seq_len)))

    selected_positions: list[int] = []
    for position in positions:
        position_index = resolve_position_index(position, corrupt_seq_len)
        if position_index < 0 or position_index >= clean_seq_len:
            raise ValueError(
                f"Patch position {position!r} resolves to {position_index}, which is not valid "
                f"for clean sequence length {clean_seq_len}."
            )
        selected_positions.append(position_index)
    return selected_positions


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
    component: str = "resid_post",
    allow_length_mismatch: bool = False,
) -> pd.DataFrame:
    return run_component_patching(
        model=model,
        clean_prompt=clean_prompt,
        corrupt_prompt=corrupt_prompt,
        target_token=target_token,
        config=config,
        output_dir=output_dir,
        full_sweep=full_sweep,
        component=component,
        allow_length_mismatch=allow_length_mismatch,
    )


def run_component_patching(
    model: Any,
    clean_prompt: str,
    corrupt_prompt: str,
    target_token: str,
    config: dict[str, Any],
    output_dir: Path,
    full_sweep: bool = False,
    component: str = "resid_post",
    allow_length_mismatch: bool = False,
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
    clean_seq_len = int(clean_tokens.shape[1])
    corrupt_seq_len = int(corrupt_tokens.shape[1])
    selected_positions = resolve_patch_positions(
        positions,
        clean_seq_len=clean_seq_len,
        corrupt_seq_len=corrupt_seq_len,
        allow_length_mismatch=allow_length_mismatch,
    )
    clean_score = target_logit_score(model, clean_prompt, target_id)
    corrupt_score = target_logit_score(model, corrupt_prompt, target_id)

    rows: list[dict[str, Any]] = []
    for layer in tqdm(layers, desc="Activation patching"):
        hook_name = component_hook_name(component, layer)
        with torch.inference_mode():
            _, clean_cache = model.run_with_cache(clean_tokens, names_filter=[hook_name])
        clean_acts = clean_cache[hook_name].detach()
        for position_index in selected_positions:
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
                    "patched_component": component,
                    "patched_position": position_index,
                    "baseline_clean_score": clean_score,
                    "baseline_corrupt_score": corrupt_score,
                    "patched_score": patched_score,
                    **patching_effect(clean_score, corrupt_score, patched_score),
                    "full_sweep": full_sweep,
                    "allow_length_mismatch": allow_length_mismatch,
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
        f"Patched component: `{component}`",
        f"Full sweep: {full_sweep}",
        f"Allow length mismatch: {allow_length_mismatch}",
        "",
        "This is causal intervention evidence for this prompt pair and metric only.",
        "It is not a full IOI replication or a broad circuit claim.",
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
                "component": component,
                "full_sweep": full_sweep,
                "exploratory": full_sweep,
                "allow_length_mismatch": allow_length_mismatch,
                "clean_seq_len": clean_seq_len,
                "corrupt_seq_len": corrupt_seq_len,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return df
