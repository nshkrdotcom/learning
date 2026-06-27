from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from local_mi_lab.head_hooks import HeadPatchSite, resolve_head_patch_site
from local_mi_lab.induction_metrics import (
    logit_diff_score,
    normalized_effect_size,
    probability_score,
    rank_score,
    resolve_induction_token_ids,
    target_logit_score,
)
from local_mi_lab.plots import plot_head_specific_gaps, plot_head_specific_patching_by_family
from local_mi_lab.prompts import read_prompts_csv
from local_mi_lab.types import PromptRecord

DEFAULT_HEAD_PATCHING_FAMILIES = [
    "positive_repeat_sequence",
    "distractor_repeat_control",
    "random_expected_token_control",
    "same_token_frequency_control",
]

RESULT_COLUMNS = [
    "run_id",
    "seed",
    "example_id",
    "family",
    "control_family",
    "should_show_induction_behavior",
    "layer",
    "head",
    "hook_name",
    "head_specific_patch",
    "actual_patch_scope",
    "intervention",
    "intervention_status",
    "position_label",
    "position_status",
    "patch_position",
    "mean_ablation_source_n_examples",
    "clean_prompt",
    "corrupt_prompt",
    "true_expected_next_token",
    "wrong_or_control_token",
    "metric",
    "clean_score",
    "corrupt_score",
    "patched_score",
    "effect_size",
    "effect_size_status",
]


def parse_head_spec(spec: str) -> tuple[int, int]:
    normalized = spec.strip().upper()
    if not normalized.startswith("L") or "H" not in normalized:
        raise ValueError(f"Head spec must look like L0H1, got {spec!r}")
    layer_text, head_text = normalized[1:].split("H", maxsplit=1)
    return int(layer_text), int(head_text)


def parse_heads(heads: str) -> list[tuple[int, int]]:
    parsed = [parse_head_spec(head) for head in heads.split(",") if head.strip()]
    return sorted(dict.fromkeys(parsed))


def heads_from_candidate_csv(
    path: str | Path,
    candidate_source: str | None = None,
    max_heads: int | None = None,
) -> list[tuple[int, int]]:
    rows = pd.read_csv(path)
    if candidate_source:
        rows = rows[rows["source"] == candidate_source]
    heads = [
        (int(row.layer), int(row.head))
        for row in rows.itertuples()
        if not pd.isna(row.head)
    ]
    deduped = sorted(dict.fromkeys(heads))
    return deduped[:max_heads] if max_heads is not None else deduped


def selected_records(
    records: list[PromptRecord],
    families: list[str],
    examples_per_family: int,
    seed: int,
) -> list[PromptRecord]:
    rng = random.Random(seed)
    selected: list[PromptRecord] = []
    for family in families:
        family_records = [record for record in records if record.family == family]
        family_records = sorted(family_records, key=lambda record: record.family_index or 0)
        rng.shuffle(family_records)
        selected.extend(sorted(family_records[:examples_per_family], key=lambda r: r.family_index or 0))
    return selected


def build_head_patching_jobs(
    records: list[PromptRecord],
    heads: list[tuple[int, int]],
    families: list[str],
    examples_per_family: int,
    seed: int,
    position_label: str,
) -> list[dict[str, Any]]:
    by_id = {record.example_id: record for record in records}
    jobs: list[dict[str, Any]] = []
    for layer, head in heads:
        for record in selected_records(records, families, examples_per_family, seed):
            positive = by_id.get(record.paired_positive_example_id)
            if positive is None:
                raise ValueError(f"Missing paired positive {record.paired_positive_example_id!r}")
            clean_prompt, corrupt_prompt = clean_corrupt_prompts(record, positive)
            jobs.append(
                {
                    "record": record,
                    "layer": layer,
                    "head": head,
                    "clean_prompt": clean_prompt,
                    "corrupt_prompt": corrupt_prompt,
                    "position_label": position_label,
                }
            )
    return jobs


def clean_corrupt_prompts(record: PromptRecord, positive: PromptRecord) -> tuple[str, str]:
    if record.is_positive_induction_example:
        return record.prompt, record.control_prompt
    if record.family == "positive_repeat_sequence":
        return positive.prompt, positive.control_prompt
    if record.family == "random_expected_token_control":
        return positive.prompt, record.prompt
    return positive.prompt, record.prompt


def apply_head_intervention(
    corrupt_act: torch.Tensor,
    clean_act: torch.Tensor | None,
    *,
    site: HeadPatchSite,
    head: int,
    clean_position: int,
    corrupt_position: int,
    intervention: str,
    mean_act: torch.Tensor | None = None,
) -> torch.Tensor:
    patched = corrupt_act.clone()
    if not site.head_specific_possible:
        if intervention == "head_clean_to_corrupt_patch" and clean_act is not None:
            patched[:, corrupt_position, :] = clean_act[:, clean_position, :]
        elif intervention == "head_zero_ablation":
            patched[:, corrupt_position, :] = 0
        elif intervention == "head_mean_ablation" and mean_act is not None:
            patched[:, corrupt_position, :] = mean_act
        elif intervention == "head_mean_ablation":
            raise ValueError("mean_act is required for head_mean_ablation")
        else:
            raise ValueError(f"Unsupported intervention {intervention!r}")
        return patched
    if intervention == "head_clean_to_corrupt_patch":
        if clean_act is None:
            raise ValueError("clean_act is required for clean-to-corrupt patching")
        patched[:, corrupt_position, head, :] = clean_act[:, clean_position, head, :]
    elif intervention == "head_zero_ablation":
        patched[:, corrupt_position, head, :] = 0
    elif intervention == "head_mean_ablation":
        if mean_act is None:
            raise ValueError("mean_act is required for head_mean_ablation")
        patched[:, corrupt_position, head, :] = mean_act
    else:
        raise ValueError(f"Unsupported intervention {intervention!r}")
    return patched


def run_head_specific_patching(
    model: Any,
    run_dir: str | Path,
    heads: list[tuple[int, int]],
    *,
    seed: int,
    families: list[str] | None = None,
    examples_per_family: int = 8,
    metric: str = "true_vs_control_logit_diff",
    intervention: str = "head_clean_to_corrupt_patch",
    position_label: str = "final",
) -> dict[str, Any]:
    root = Path(run_dir)
    records = read_prompts_csv(root / "prompts.csv")
    selected_families = families or DEFAULT_HEAD_PATCHING_FAMILIES
    jobs = build_head_patching_jobs(
        records,
        heads=heads,
        families=selected_families,
        examples_per_family=examples_per_family,
        seed=seed,
        position_label=position_label,
    )
    site_by_layer = {
        layer: HeadPatchSite(**resolve_head_patch_site(model, layer))
        for layer in sorted({layer for layer, _ in heads})
    }
    mean_activations: dict[tuple[int, int, str, int], tuple[torch.Tensor, int]] = {}
    if intervention == "head_mean_ablation":
        mean_activations = compute_mean_ablation_activations(model, jobs, site_by_layer)
        for job in jobs:
            key = job.get("_mean_ablation_key")
            if key is not None:
                job["mean_act"], job["mean_ablation_source_n_examples"] = mean_activations.get(
                    key,
                    (None, 0),
                )
    rows = [
        _run_head_job(
            model,
            job,
            run_id=root.name,
            seed=seed,
            site=site_by_layer[int(job["layer"])],
            metric=metric,
            intervention=intervention,
        )
        for job in tqdm(jobs, desc="Head-specific patching")
    ]
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    by_family = aggregate_head_patching_by_family(results)
    by_head = aggregate_head_patching_by_head(results)
    summary = head_patching_summary(results, by_family, by_head, selected_families)
    write_head_patching_artifacts(root, results, by_family, by_head, summary)
    return summary


def aggregate_head_patching_by_family(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    valid = results.copy()
    _ensure_result_defaults(valid)
    valid["effect_size_numeric"] = pd.to_numeric(valid["effect_size"], errors="coerce")
    grouped = valid.groupby(
        [
            "seed",
            "family",
            "layer",
            "head",
            "hook_name",
            "head_specific_patch",
            "actual_patch_scope",
            "intervention",
            "intervention_status",
            "metric",
            "position_label",
        ],
        as_index=False,
    )
    return grouped.agg(
        n_examples=("example_id", "nunique"),
        mean_effect_size=("effect_size_numeric", "mean"),
        median_effect_size=("effect_size_numeric", "median"),
        n_denominator_zero=("effect_size_status", lambda values: int((values == "denominator_zero").sum())),
        fraction_positive_effect=(
            "effect_size_numeric",
            lambda values: float((values.dropna() > 0).mean()) if len(values.dropna()) else 0.0,
        ),
        should_show_induction_behavior=("should_show_induction_behavior", "first"),
    )


def aggregate_head_patching_by_head(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    results = results.copy()
    _ensure_result_defaults(results)
    by_family = aggregate_head_patching_by_family(results)
    rows: list[dict[str, Any]] = []
    group_cols = [
        "seed",
        "layer",
        "head",
        "hook_name",
        "head_specific_patch",
        "actual_patch_scope",
        "intervention",
        "intervention_status",
        "metric",
        "position_label",
    ]
    for key, subset in by_family.groupby(group_cols):
        key_dict = dict(zip(group_cols, key, strict=True))
        positive = subset[subset["should_show_induction_behavior"]]
        controls = subset[~subset["should_show_induction_behavior"]]
        positive_mean = _safe_float(positive["mean_effect_size"].mean()) if not positive.empty else None
        max_control_row = (
            controls.sort_values("mean_effect_size", ascending=False).iloc[0]
            if not controls.empty and controls["mean_effect_size"].notna().any()
            else None
        )
        max_control_mean = (
            _safe_float(max_control_row["mean_effect_size"]) if max_control_row is not None else None
        )
        gap = (
            positive_mean - max_control_mean
            if positive_mean is not None and max_control_mean is not None
            else None
        )
        head_results = results.copy()
        for column in group_cols:
            head_results = head_results[head_results[column] == key_dict[column]]
        rows.append(
            {
                **key_dict,
                "positive_mean_effect_size": positive_mean,
                "max_control_mean_effect_size": max_control_mean,
                "positive_minus_control_effect_gap": gap,
                "hardest_control_family": (
                    str(max_control_row["family"]) if max_control_row is not None else ""
                ),
                "n_positive_examples": int(positive["n_examples"].sum()) if not positive.empty else 0,
                "n_control_examples": int(controls["n_examples"].sum()) if not controls.empty else 0,
                "specificity_status": classify_head_specificity(
                    head_results,
                    bool(key_dict["head_specific_patch"]),
                    positive_mean,
                    max_control_mean,
                    gap,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "positive_minus_control_effect_gap",
        ascending=False,
        na_position="last",
    )


def classify_head_specificity(
    head_results: pd.DataFrame,
    head_specific_patch: bool,
    positive_mean: float | None,
    max_control_mean: float | None,
    gap: float | None,
) -> str:
    if not head_specific_patch:
        return "not_head_specific"
    if head_results.empty:
        return "insufficient_examples"
    denominator_zero = int((head_results["effect_size_status"] == "denominator_zero").sum())
    if denominator_zero >= max(1, len(head_results) // 2):
        return "denominator_problem"
    if positive_mean is None or max_control_mean is None:
        return "insufficient_examples"
    if positive_mean <= 0:
        return "no_positive_effect"
    if max_control_mean >= positive_mean:
        return "nonspecific_moves_controls"
    if gap is not None and gap > 0:
        return "head_specific_positive_candidate"
    return "insufficient_examples"


def head_patching_summary(
    results: pd.DataFrame,
    by_family: pd.DataFrame,
    by_head: pd.DataFrame,
    families: list[str],
) -> dict[str, Any]:
    positive = by_family[by_family["should_show_induction_behavior"]] if not by_family.empty else pd.DataFrame()
    controls = by_family[~by_family["should_show_induction_behavior"]] if not by_family.empty else pd.DataFrame()
    return {
        "n_result_rows": int(len(results)),
        "families": families,
        "n_heads": int(results[["layer", "head"]].drop_duplicates().shape[0]) if not results.empty else 0,
        "head_specific_patch": bool(results["head_specific_patch"].all()) if not results.empty else False,
        "actual_patch_scopes": (
            sorted(str(scope) for scope in results["actual_patch_scope"].dropna().unique())
            if not results.empty
            else []
        ),
        "positive_mean_effect_size": (
            _safe_float(positive["mean_effect_size"].mean()) if not positive.empty else None
        ),
        "max_control_mean_effect_size": (
            _safe_float(controls["mean_effect_size"].max()) if not controls.empty else None
        ),
        "best_positive_minus_control_effect_gap": (
            _safe_float(by_head["positive_minus_control_effect_gap"].max())
            if not by_head.empty
            else None
        ),
        "specificity_status_counts": (
            by_head["specificity_status"].value_counts().to_dict() if not by_head.empty else {}
        ),
        "interpretation_note": (
            "Head-specific patching tests selected heads under selected prompts, controls, "
            "position, intervention, and metric. It is not a circuit claim."
        ),
    }


def write_head_patching_artifacts(
    run_dir: Path,
    results: pd.DataFrame,
    by_family: pd.DataFrame,
    by_head: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    results.to_csv(run_dir / "head_specific_patching_results.csv", index=False)
    by_family.to_csv(run_dir / "head_specific_patching_by_family.csv", index=False)
    by_head.to_csv(run_dir / "head_specific_patching_by_head.csv", index=False)
    (run_dir / "head_specific_patching_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "head_specific_patching_notes.md").write_text(_notes(summary), encoding="utf-8")
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    plot_head_specific_patching_by_family(by_family, figures / "head_specific_patching_by_family.png")
    plot_head_specific_gaps(by_head, figures / "head_specific_patching_head_gaps.png")


def _run_head_job(
    model: Any,
    job: dict[str, Any],
    *,
    run_id: str,
    seed: int,
    site: HeadPatchSite,
    metric: str,
    intervention: str,
) -> dict[str, Any]:
    record: PromptRecord = job["record"]
    token_ids = resolve_induction_token_ids(model.tokenizer, record)
    true_id = int(token_ids["true_token_id"])
    wrong_id = int(token_ids["wrong_or_control_token_id"])
    clean_tokens = model.to_tokens(job["clean_prompt"])
    corrupt_tokens = model.to_tokens(job["corrupt_prompt"])
    clean_position, clean_position_status = resolve_position_index_for_record(
        record,
        job["position_label"],
        int(clean_tokens.shape[1]),
    )
    corrupt_position, corrupt_position_status = resolve_position_index_for_record(
        record,
        job["position_label"],
        int(corrupt_tokens.shape[1]),
    )
    position_status = (
        "ok"
        if clean_position_status == "ok" and corrupt_position_status == "ok"
        else "unavailable_for_family"
    )
    if clean_position is None or corrupt_position is None:
        return _skipped_position_row(
            job,
            record=record,
            run_id=run_id,
            seed=seed,
            site=site,
            metric=metric,
            intervention=intervention,
            position_status=position_status,
        )
    with torch.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=[site.hook_name])
        corrupt_logits = model(corrupt_tokens)
    clean_act = _cache_get(clean_cache, site.hook_name)
    clean_score = score_logits(clean_logits, metric, true_id, wrong_id)
    corrupt_score = score_logits(corrupt_logits, metric, true_id, wrong_id)

    def patch_hook(corrupt_act: torch.Tensor, hook: Any) -> torch.Tensor:
        del hook
        return apply_head_intervention(
            corrupt_act,
            clean_act,
            site=site,
            head=int(job["head"]),
            clean_position=clean_position,
            corrupt_position=corrupt_position,
            intervention=intervention,
            mean_act=job.get("mean_act"),
        )

    with torch.inference_mode():
        patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(site.hook_name, patch_hook)])
    patched_score = score_logits(patched_logits, metric, true_id, wrong_id)
    effect = normalized_effect_size(clean_score, corrupt_score, patched_score)
    return {
        "run_id": run_id,
        "seed": seed,
        "example_id": record.example_id,
        "family": record.family,
        "control_family": record.control_family,
        "should_show_induction_behavior": record.should_show_induction_behavior,
        "layer": int(job["layer"]),
        "head": int(job["head"]),
        "hook_name": site.hook_name,
        "head_specific_patch": site.head_specific_possible,
        "actual_patch_scope": site.actual_patch_scope,
        "intervention": intervention,
        "intervention_status": "ok",
        "position_label": job["position_label"],
        "position_status": position_status,
        "patch_position": corrupt_position,
        "mean_ablation_source_n_examples": int(job.get("mean_ablation_source_n_examples") or 0),
        "clean_prompt": job["clean_prompt"],
        "corrupt_prompt": job["corrupt_prompt"],
        "true_expected_next_token": record.true_expected_next_token,
        "wrong_or_control_token": record.wrong_or_control_token,
        "metric": metric,
        "clean_score": clean_score,
        "corrupt_score": corrupt_score,
        "patched_score": patched_score,
        **effect,
    }


def score_logits(logits: Any, metric: str, true_id: int, wrong_id: int) -> float:
    if metric == "target_logit":
        return target_logit_score(logits, true_id)
    if metric == "true_vs_control_logit_diff":
        return logit_diff_score(logits, true_id, wrong_id)
    if metric == "probability_gap":
        return probability_score(logits, true_id) - probability_score(logits, wrong_id)
    if metric == "rank_delta":
        return float(rank_score(logits, wrong_id) - rank_score(logits, true_id))
    raise ValueError(f"Unsupported metric {metric!r}")


def position_index_for_record(record: PromptRecord, position_label: str, seq_len: int) -> int:
    position, status = resolve_position_index_for_record(record, position_label, seq_len)
    if position is None:
        raise ValueError(f"Record {record.example_id} has no source position metadata")
    if status != "ok":
        raise ValueError(f"Position {position_label!r} unavailable for {record.example_id}")
    return position


def resolve_position_index_for_record(
    record: PromptRecord,
    position_label: str,
    seq_len: int,
) -> tuple[int | None, str]:
    if position_label == "final":
        return seq_len - 1, "ok"
    if position_label in {"source", "previous_occurrence"}:
        if record.expected_source_position_hint is None:
            return None, "unavailable_for_family"
        bos_offset = max(seq_len - len(record.prompt_tokens_text), 0)
        return int(record.expected_source_position_hint) + bos_offset, "ok"
    if position_label == "all_prompt_positions":
        raise ValueError("all_prompt_positions is supported only by the sweep job expander")
    raise ValueError(f"Unsupported position label {position_label!r}")


def mean_ablation_key(job: dict[str, Any], seq_len: int) -> tuple[int, int, str, int] | None:
    record: PromptRecord = job["record"]
    position, status = resolve_position_index_for_record(record, job["position_label"], seq_len)
    if position is None or status != "ok":
        return None
    return int(job["layer"]), int(job["head"]), str(job["position_label"]), int(position)


def compute_mean_ablation_activations(
    model: Any,
    jobs: list[dict[str, Any]],
    site_by_layer: dict[int, HeadPatchSite],
) -> dict[tuple[int, int, str, int], tuple[torch.Tensor, int]]:
    values: dict[tuple[int, int, str, int], list[torch.Tensor]] = {}
    for job in jobs:
        tokens = model.to_tokens(job["clean_prompt"])
        key = mean_ablation_key(job, int(tokens.shape[1]))
        job["_mean_ablation_key"] = key
        if key is None:
            continue
        layer, head, _position_label, clean_position = key
        site = site_by_layer[layer]
        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=[site.hook_name])
        act = _cache_get(cache, site.hook_name)
        if site.head_specific_possible:
            vector = act[:, clean_position, head, :].mean(dim=0)
        else:
            vector = act[:, clean_position, :].mean(dim=0)
        values.setdefault(key, []).append(vector)
    return {
        key: (torch.stack(vectors, dim=0).mean(dim=0), len(vectors))
        for key, vectors in values.items()
    }


def _ensure_result_defaults(results: pd.DataFrame) -> None:
    defaults: dict[str, Any] = {
        "intervention_status": "ok",
        "position_status": "ok",
        "mean_ablation_source_n_examples": 0,
    }
    for column, default in defaults.items():
        if column not in results.columns:
            results[column] = default


def _skipped_position_row(
    job: dict[str, Any],
    *,
    record: PromptRecord,
    run_id: str,
    seed: int,
    site: HeadPatchSite,
    metric: str,
    intervention: str,
    position_status: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": seed,
        "example_id": record.example_id,
        "family": record.family,
        "control_family": record.control_family,
        "should_show_induction_behavior": record.should_show_induction_behavior,
        "layer": int(job["layer"]),
        "head": int(job["head"]),
        "hook_name": site.hook_name,
        "head_specific_patch": site.head_specific_possible,
        "actual_patch_scope": site.actual_patch_scope,
        "intervention": intervention,
        "intervention_status": "skipped_position_unavailable",
        "position_label": job["position_label"],
        "position_status": position_status,
        "patch_position": None,
        "mean_ablation_source_n_examples": 0,
        "clean_prompt": job["clean_prompt"],
        "corrupt_prompt": job["corrupt_prompt"],
        "true_expected_next_token": record.true_expected_next_token,
        "wrong_or_control_token": record.wrong_or_control_token,
        "metric": metric,
        "clean_score": None,
        "corrupt_score": None,
        "patched_score": None,
        "effect_size": None,
        "effect_size_status": "position_unavailable",
    }


def _cache_get(cache: Any, hook_name: str) -> Any:
    cache_dict = getattr(cache, "cache_dict", None)
    if isinstance(cache_dict, dict) and hook_name in cache_dict:
        return cache_dict[hook_name]
    return cache[hook_name]


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _notes(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Head-Specific Patching Notes",
            "",
            "This artifact tests selected heads under selected controls and metrics.",
            f"Head-specific patching: {summary['head_specific_patch']}",
            f"Patch scopes: {summary['actual_patch_scopes']}",
            f"Heads patched: {summary['n_heads']}",
            f"Positive mean effect size: {summary['positive_mean_effect_size']}",
            f"Max control mean effect size: {summary['max_control_mean_effect_size']}",
            f"Best positive-minus-control effect gap: {summary['best_positive_minus_control_effect_gap']}",
            f"Specificity statuses: {summary['specificity_status_counts']}",
            "",
            "This is not an induction-head discovery or a circuit claim.",
        ]
    ) + "\n"
