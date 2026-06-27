from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from local_mi_lab.attention import ATTENTION_LIMITATION
from local_mi_lab.paths import relative_files, resolve_repo_path

MANDATORY_LANGUAGE = [
    "This is not a mechanism claim unless causal intervention evidence supports it.",
    "This is not a broad model claim.",
    "This is a local MI practice run.",
]


def read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows_if_present(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def generate_run_summary(run_dir: str | Path) -> str:
    root = resolve_repo_path(run_dir)
    config = read_json_if_present(root / "capability_report.json")
    baseline = read_json_if_present(root / "baseline_metrics.json")
    activation_manifest = read_json_if_present(root / "activations" / "manifest.json")
    activation_summary = read_json_if_present(root / "activation_summary.json")
    logit_lens = read_json_if_present(root / "logit_lens_summary.json")
    attention = read_json_if_present(root / "attention_summary.json")
    patching_metadata = read_json_if_present(root / "patching_metadata.json")
    controlled_patching = read_json_if_present(root / "controlled_patching_summary.json")
    files = relative_files(root) if root.exists() else []

    lines = [
        "# Run Summary",
        "",
        "## Question",
        "",
        _question_text(root),
        "",
        "## Model and resources",
        "",
        _model_resources_text(config, baseline, activation_manifest),
        "",
        "## Files present",
        "",
        *_files_text(files),
        "",
        "## Baseline behavior",
        "",
        _baseline_text(baseline),
        "",
        "## Baseline behavior by family",
        "",
        _baseline_family_text(root, baseline),
        "",
        "## Activation cache",
        "",
        _activation_text(activation_manifest or activation_summary),
        "",
        "## Logit lens",
        "",
        _logit_lens_text(logit_lens),
        "",
        "## Logit lens by family",
        "",
        _logit_lens_family_text(root, logit_lens),
        "",
        "## Attention patterns",
        "",
        _attention_text(attention),
        "",
        "## Attention controls",
        "",
        _attention_controls_text(root, attention),
        "",
        "## Activation patching",
        "",
        _patching_text(root, patching_metadata),
        "",
        "## Controlled patching",
        "",
        _controlled_patching_text(controlled_patching),
        "",
        "## Controlled patching by family",
        "",
        _controlled_patching_family_text(root),
        "",
        "## Candidate specificity",
        "",
        _candidate_specificity_text(root, controlled_patching),
        "",
        "## False positives and controls",
        "",
        _false_positive_text(attention),
        "",
        "## False-positive lesson",
        "",
        _controlled_false_positive_lesson(controlled_patching),
        "",
        "## What this shows",
        "",
        "This is a local MI practice run. It shows only the artifacts present in this run directory.",
        "",
        "## What this suggests",
        "",
        _suggests_text(root, baseline, attention),
        "",
        "## What causal patching changes",
        "",
        _what_causal_patching_changes(controlled_patching),
        "",
        "## What causal patching does not show",
        "",
        _what_causal_patching_does_not_show(),
        "",
        "## What this does not show",
        "",
        *MANDATORY_LANGUAGE[:2],
        "A head that attends to previous occurrences on positive prompts but also attends strongly on controls is not a specific induction-head candidate.",
        "A positive-minus-control gap is more informative than raw positive attention alone.",
        "These controls are still simple practice controls, not a publication-quality induction-head benchmark.",
        "",
        "## Next step",
        "",
        _next_step_text(root),
        "",
        MANDATORY_LANGUAGE[2],
    ]
    return "\n".join(lines) + "\n"


def write_run_summary(run_dir: str | Path) -> Path:
    root = resolve_repo_path(run_dir)
    summary = generate_run_summary(root)
    output = root / "summary.md"
    output.write_text(summary, encoding="utf-8")
    return output


def _question_text(root: Path) -> str:
    if (root / "prompts.csv").exists():
        return "Does the model show a simple expected-token behavior on the prompt set?"
    if (root / "capability_report.json").exists():
        return "Can the configured model and local environment run the first-pass workflow?"
    return "Missing: no question artifact was found."


def _model_resources_text(
    capability: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
    activation: dict[str, Any] | None,
) -> str:
    if capability:
        status = capability.get("status", "unknown")
        model = capability.get("model", "unknown")
        return f"Capability report present for `{model}` with status `{status}`."
    if baseline:
        return f"Baseline metrics present for `{baseline.get('model', 'unknown')}`."
    if activation:
        return f"Activation manifest present for `{activation.get('model', 'unknown')}`."
    return "Missing: no model/resource artifact was found."


def _files_text(files: list[str]) -> list[str]:
    if not files:
        return ["Missing: run directory has no files."]
    return [f"- `{file}`" for file in files]


def _baseline_text(baseline: dict[str, Any] | None) -> str:
    if baseline is None:
        return "Missing: `baseline_metrics.json` was not found."
    return (
        f"Examples: {baseline.get('n_examples')}. "
        f"Mean expected probability: {baseline.get('mean_expected_probability')}. "
        f"Median expected rank: {baseline.get('median_expected_rank')}. "
        f"Mean probability diff versus control: {baseline.get('mean_probability_diff_vs_control')}."
    )


def _baseline_family_text(root: Path, baseline: dict[str, Any] | None) -> str:
    if not (root / "baseline_by_family.csv").exists():
        return "Missing: `baseline_by_family.csv` was not found."
    if not baseline or "positive_vs_control_gap" not in baseline:
        return "Baseline family table present. Controlled positive-vs-control summary is missing."
    gap = baseline["positive_vs_control_gap"]
    hardest = baseline.get("hardest_control_family") or {}
    return (
        "Baseline family table present. "
        f"Positive mean expected probability: {gap.get('positive_mean_expected_probability')}. "
        f"Max control mean expected probability: {gap.get('max_control_mean_expected_probability')}. "
        f"Gap: {gap.get('gap_mean_expected_probability')}. "
        f"Hardest control family: {hardest.get('family')}."
    )


def _activation_text(manifest: dict[str, Any] | None) -> str:
    if manifest is None:
        return "Missing: activation manifest was not found."
    layers = manifest.get("layers", [])
    files = manifest.get("files", [])
    return f"Selected activation cache present for layers {layers}; tensor files: {len(files)}."


def _logit_lens_text(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "Missing: `logit_lens_summary.json` was not found."
    return (
        "Logit lens summary present. "
        f"Best layer by mean probability: {summary.get('best_layer_by_mean_probability')}. "
        "This is descriptive, not causal evidence."
    )


def _logit_lens_family_text(root: Path, summary: dict[str, Any] | None) -> str:
    if not (root / "logit_lens_by_family.csv").exists():
        return "Missing: `logit_lens_by_family.csv` was not found."
    if summary is None:
        return "Logit-lens family table present, but `logit_lens_summary.json` is missing."
    best_positive = summary.get("best_positive_layer")
    hardest_control = summary.get("hardest_control_family_by_expected_probability")
    separates = summary.get("positive_separates_from_controls_descriptively")
    return (
        "Logit-lens-by-family table present. "
        f"Best positive layer: {best_positive}. "
        f"Hardest control family by expected-token probability: {hardest_control}. "
        f"Positive separates from controls descriptively: {separates}. "
        "Logit lens remains descriptive."
    )


def _attention_text(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return f"Missing: `attention_summary.json` was not found. {ATTENTION_LIMITATION}"
    top_heads = (
        summary.get("top_heads_on_positive_examples")
        or summary.get("top_heads_by_previous_occurrence_attention")
        or []
    )
    if not top_heads:
        return f"Attention summary present, but no top heads were recorded. {ATTENTION_LIMITATION}"
    formatted = []
    for head in top_heads[:5]:
        formatted.append(
            "L"
            f"{int(head['layer'])}H{int(head['head'])}="
            f"{float(head['mean_attention_to_previous_occurrence']):.3f}"
        )
    return (
        "Attention summary present. Top induction-like attention pattern candidates by "
        f"previous-occurrence attention: {', '.join(formatted)}. {ATTENTION_LIMITATION}"
    )


def _attention_controls_text(root: Path, summary: dict[str, Any] | None) -> str:
    if not (root / "attention_by_family.csv").exists():
        return f"Missing: `attention_by_family.csv` was not found. {ATTENTION_LIMITATION}"
    if summary is None:
        return "Attention family table present, but `attention_summary.json` is missing."
    gap_heads = summary.get("top_heads_by_positive_minus_control_gap") or []
    raw_heads = summary.get("top_heads_on_positive_examples") or []
    hardest = summary.get("hardest_control_family_by_attention")
    gap_text = _format_gap_heads(gap_heads[:5])
    raw_text = _format_raw_heads(raw_heads[:5])
    return (
        f"Raw positive heads: {raw_text}. "
        f"Top positive-minus-control gap heads: {gap_text}. "
        f"Hardest control family by attention: {hardest}. "
        f"{ATTENTION_LIMITATION}"
    )


def _patching_text(root: Path, metadata: dict[str, Any] | None) -> str:
    if not (root / "patching_results.csv").exists():
        return "Missing: `patching_results.csv` was not found."
    if metadata:
        return (
            f"Patching results present for metric `{metadata.get('metric')}`. "
            f"Exploratory full sweep: {metadata.get('exploratory')}."
        )
    return "Patching results present, but metadata is missing."


def _controlled_patching_text(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "Missing: `controlled_patching_summary.json` was not found."
    return (
        "Controlled patching summary present. Controlled patching asks whether causal effects "
        "separate positives from controls. "
        f"Candidates patched: {summary.get('n_candidates')}. "
        f"Positive mean effect size: {summary.get('positive_mean_effect_size')}. "
        f"Max control mean effect size: {summary.get('max_control_mean_effect_size')}. "
        f"Best positive-minus-control causal gap: {summary.get('best_positive_minus_control_effect_gap')}."
    )


def _controlled_patching_family_text(root: Path) -> str:
    rows = read_csv_rows_if_present(root / "controlled_patching_by_family.csv")
    if not rows:
        return "Missing: `controlled_patching_by_family.csv` was not found."
    families = sorted({row["family"] for row in rows})
    return f"Controlled patching family table present for families: {', '.join(families)}."


def _candidate_specificity_text(root: Path, summary: dict[str, Any] | None) -> str:
    if not (root / "controlled_patching_by_candidate.csv").exists():
        return "Missing: `controlled_patching_by_candidate.csv` was not found."
    counts = (summary or {}).get("specificity_status_counts", {})
    return (
        f"Candidate specificity statuses: {counts}. "
        "A candidate that moves controls as much as positives is nonspecific. "
        "A candidate with positive-minus-control causal gap is more interesting than a raw "
        "attention candidate, but still not a mechanism claim. "
        "Layer-level attn_out patching is not head-specific unless the artifact explicitly says "
        "head_specific_patch=true."
    )


def _false_positive_text(summary: dict[str, Any] | None) -> str:
    base = (
        "A head that attends to previous occurrences on positive prompts but also attends strongly "
        "on controls is not a specific induction-head candidate. A positive-minus-control gap is "
        "more informative than raw positive attention alone."
    )
    if summary is None:
        return base
    control_heads = summary.get("top_heads_on_controls") or []
    if not control_heads:
        return f"{base} No control-firing head summary was found."
    return f"{base} Strongest control-firing heads include: {_format_raw_heads(control_heads[:5])}."


def _controlled_false_positive_lesson(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "Controlled patching has not run yet, so the causal false-positive lesson is pending."
    counts = summary.get("specificity_status_counts", {})
    if counts.get("nonspecific_moves_controls", 0):
        return (
            "Controlled patching found at least one candidate where controls moved as much as "
            "positives. That is a nonspecific causal pattern, not an induction-head result."
        )
    if counts.get("positive_specific_candidate", 0):
        return (
            "Controlled patching found at least one positive-specific candidate. This is worth "
            "manual inspection and small replication, not a mechanism claim."
        )
    return "Controlled patching did not identify a clear positive-specific causal pattern."


def _suggests_text(
    root: Path,
    baseline: dict[str, Any] | None,
    attention: dict[str, Any] | None,
) -> str:
    if not (root / "baseline_by_family.csv").exists() or not (root / "attention_by_family.csv").exists():
        return "Controlled artifacts are incomplete, so no false-positive comparison should be inferred."
    gap = (baseline or {}).get("positive_vs_control_gap", {})
    attention_gap = (attention or {}).get("top_heads_by_positive_minus_control_gap", [])
    return (
        "The controlled run can suggest whether behavior and attention candidates separate "
        f"positives from controls. Baseline probability gap: {gap.get('gap_mean_expected_probability')}. "
        f"Top attention gap candidate: {_format_gap_heads(attention_gap[:1])}."
    )


def _what_causal_patching_changes(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "Causal patching has not been run for this controlled artifact set."
    return (
        "Causal patching changes the evidence type from descriptive attention/logit-lens patterns "
        "to an intervention on selected components. It only tests the selected examples, candidate "
        "sites, component scope, position, and metric."
    )


def _what_causal_patching_does_not_show() -> str:
    return (
        "Controlled patching does not identify a full circuit by itself. Layer-level attn_out "
        "patching is not head-specific unless `head_specific_patch=true` is recorded. A positive "
        "causal gap is still only a practice candidate until replicated and inspected manually."
    )


def _next_step_text(root: Path) -> str:
    if not (root / "prompts.csv").exists():
        return "Build induction_controls prompts."
    if not (root / "baseline_by_family.csv").exists():
        return "Run baseline behavior on controls."
    if not (root / "attention_by_family.csv").exists():
        return "Run attention-pattern controls."
    if not (root / "logit_lens_by_family.csv").exists():
        return "Run logit lens by family."
    if not (root / "controlled_patching_candidates.csv").exists():
        return "Run select_controlled_patching_candidates.py, then run_controlled_patching.py."
    if not (root / "controlled_patching_summary.json").exists():
        return "Run controlled patching on selected candidates."
    summary = read_json_if_present(root / "controlled_patching_summary.json") or {}
    counts = summary.get("specificity_status_counts", {})
    if counts.get("positive_specific_candidate", 0):
        return (
            "Inspect the candidate manually and run a smaller replication with a new seed before "
            "any stronger claim."
        )
    return "Write a learning note explaining the false-positive pattern before adding new tasks."


def _format_raw_heads(heads: list[dict[str, Any]]) -> str:
    if not heads:
        return "none"
    return ", ".join(
        "L"
        f"{int(head['layer'])}H{int(head['head'])}="
        f"{float(head['mean_attention_to_previous_occurrence']):.3f}"
        for head in heads
    )


def _format_gap_heads(heads: list[dict[str, Any]]) -> str:
    if not heads:
        return "none"
    return ", ".join(
        "L"
        f"{int(head['layer'])}H{int(head['head'])} gap="
        f"{float(head['positive_minus_control_attention_gap']):.3f}"
        for head in heads
    )
