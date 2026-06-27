from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from local_mi_lab.activations import resid_post_hook_name
from local_mi_lab.config import selected_layers
from local_mi_lab.metrics import expected_token_stats
from local_mi_lab.models import n_layers
from local_mi_lab.plots import plot_logit_lens
from local_mi_lab.tokens import decode_token, token_id_for_single_token, top_token_ids
from local_mi_lab.types import PromptRecord


def residual_to_logits(model: Any, resid: torch.Tensor) -> torch.Tensor:
    resid_for_lens = resid.unsqueeze(1)
    if hasattr(model, "ln_final"):
        resid_for_lens = model.ln_final(resid_for_lens)
    logits = model.unembed(resid_for_lens)
    return logits[:, 0, :]


def compute_logit_lens(
    model: Any,
    records: list[PromptRecord],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    layers = selected_layers(config, n_layers(model))
    hook_names = [resid_post_hook_name(layer) for layer in layers]
    rows: list[dict[str, Any]] = []

    for record in tqdm(records, desc="Running logit lens"):
        target_id = token_id_for_single_token(model.tokenizer, record.expected_next_token)
        tokens = model.to_tokens(record.prompt)
        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)
        for layer in layers:
            resid = cache[resid_post_hook_name(layer)][:, -1, :].detach()
            lens_logits = residual_to_logits(model, resid)[0]
            stats = expected_token_stats(lens_logits, target_id)
            top_id = top_token_ids(lens_logits, k=1)[0]
            rows.append(
                {
                    "example_id": record.example_id,
                    "layer": layer,
                    "expected_token": record.expected_next_token,
                    "expected_token_id": target_id,
                    "expected_logit": stats["target_logit"],
                    "expected_probability": stats["target_probability"],
                    "expected_rank": stats["target_rank"],
                    "top_token": decode_token(model.tokenizer, top_id),
                    "top_token_id": top_id,
                }
            )

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "logit_lens_by_layer.csv", index=False)
    summary_df = (
        df.groupby("layer", as_index=False)
        .agg(
            mean_expected_logit=("expected_logit", "mean"),
            mean_expected_probability=("expected_probability", "mean"),
            median_expected_rank=("expected_rank", "median"),
        )
        .sort_values("layer")
    )
    summary = {
        "model": config["model"]["name"],
        "n_examples": len(records),
        "layers": layers,
        "best_layer_by_mean_probability": int(
            summary_df.sort_values("mean_expected_probability", ascending=False).iloc[0]["layer"]
        )
        if not summary_df.empty
        else None,
        "by_layer": summary_df.to_dict(orient="records"),
        "interpretation_note": "Logit lens is descriptive and is not causal mechanism evidence.",
    }
    (output_dir / "logit_lens_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_logit_lens(summary_df, figures_dir / "logit_lens_expected_token.png")
    return df, summary
