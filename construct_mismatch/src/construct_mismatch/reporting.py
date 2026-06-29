from __future__ import annotations

from pathlib import Path

import pandas as pd

from construct_mismatch.datasets import CONSTRUCTS, DECOUPLING_AXES, artifact_path, dataset_file


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    if df.empty:
        return "_No artifact available._"
    present = [column for column in columns if column in df.columns]
    if not present:
        return "_No matching columns available._"
    table = df[present].head(max_rows).copy()
    lines = [
        "| " + " | ".join(present) + " |",
        "| " + " | ".join("---" for _ in present) + " |",
    ]
    for row in table.itertuples(index=False):
        values = [str(value).replace("\n", "<br>") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def dataset_examples(root: Path) -> str:
    lines: list[str] = []
    for construct in CONSTRUCTS:
        path = dataset_file(construct, "decoupling", root)
        if not path.exists():
            continue
        df = pd.read_json(path, lines=True)
        lines.append(f"### {construct}")
        for axis in DECOUPLING_AXES:
            row = df[df["decoupling_axis"] == axis].head(1)
            if row.empty:
                continue
            record = row.iloc[0]
            lines.append(f"- `{axis}`: {record['prompt']} -> {record['class_a_target']}/{record['class_b_target']}")
    return "\n".join(lines)


def summarize_matrix(matrix: pd.DataFrame) -> str:
    if matrix.empty:
        return "_Matrix was not generated._"
    pivot = matrix.pivot_table(
        index=["construct", "method"],
        columns="evaluation_axis",
        values="status",
        aggfunc="first",
    )
    pivot = pivot.reset_index()
    columns = [str(column) for column in pivot.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in pivot.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def generate_report(root: Path) -> Path:
    artifacts = artifact_path(root)
    behavior = read_csv_or_empty(artifacts / "behavior" / "behavior_summary.csv")
    behavior_examples = read_csv_or_empty(artifacts / "behavior" / "behavior_examples.csv")
    matrix = read_csv_or_empty(artifacts / "scoring" / "construct_mismatch_matrix.csv")
    classifications = read_csv_or_empty(artifacts / "scoring" / "object_classifications.csv")
    tokenization = read_csv_or_empty(artifacts / "tokenization" / "gpt2_small_target_tokens.csv")

    direction_tables = {
        construct: read_csv_or_empty(artifacts / "directions" / f"{construct}_direction_metrics.csv")
        for construct in CONSTRUCTS
    }
    probe_tables = {
        construct: read_csv_or_empty(artifacts / "probes" / f"{construct}_probe_metrics.csv")
        for construct in CONSTRUCTS
    }
    patching_tables = {
        construct: read_csv_or_empty(artifacts / "patching" / f"{construct}_top_sites.csv")
        for construct in CONSTRUCTS
    }

    lines = [
        "# Construct Mismatch Report",
        "",
        "## 1. Abstract",
        "This report tests whether direction, probe, and activation-patching methods appear to disagree because they operationalize different validity targets. The main artifact is a construct mismatch matrix over certainty/uncertainty and sentiment in GPT-2 Small.",
        "",
        "## 2. Hypothesis",
        "Method success and failure should be predictable from construct-validity mismatches. When lexical cues, semantic stance, target-token behavior, and causal intervention are decoupled, methods that look successful on ordinary examples should fail in different ways.",
        "",
        "## 3. Prior-art Positioning",
        "Sentiment analysis in GPT-style models is already well studied. Activation patching, linear probes, diff-in-means directions, sentiment steering, GPT-2 sentiment analysis, and the observation that probes can be predictive but non-causal are not claimed as novel here.",
        "",
        "## 4. Why This Is Not Merely a Method Benchmark",
        "The evaluation varies the operationalization of the construct: ordinary examples, lexical reversals, negation, quotation, contrast, format shifts, causal steering, and specificity. The comparison is about construct validity, not a single leaderboard score.",
        "",
        "## 5. Why Sentiment Is a Baseline and Certainty Is Core",
        "Sentiment is included as a familiar sanity check for the pipeline. Certainty/uncertainty is the stronger target construct because it is less saturated and more directly stresses the distinction between endorsed stance, lexical cue, and target-token behavior.",
        "",
        "## 6. Model and Hardware",
        "All experiments use GPT-2 Small through TransformerLens. Hardware is whatever device TransformerLens selected locally (`cuda`, `mps`, or `cpu`); no additional models are included.",
        "",
        "## 7. Tokenization Constraints",
        "GPT-2 leading spaces are part of the target tokens. Dataset construction validates target strings as single tokens before writing JSONL records.",
        "",
        markdown_table(tokenization, ["construct", "raw_string", "token_ids", "n_tokens", "usable_as_target"], 25),
        "",
        "## 8. Dataset Design",
        "Each construct has train, ordinary heldout, and decoupling splits. Records include paired class-A/class-B examples for patching where possible. The datasets are small and manually templated to prioritize interpretability over coverage.",
        "",
        "## 9. Dataset Examples by Decoupling Axis",
        dataset_examples(root),
        "",
        "## 10. Behavior Check Results",
        markdown_table(behavior, ["construct", "split", "decoupling_axis", "n", "accuracy", "mean_signed_logit_diff", "behavior_status"], 20),
        "",
        "Strong disagreements or weak margins:",
        markdown_table(behavior_examples, ["id", "construct", "decoupling_axis", "signed_logit_diff", "flag", "prompt"], 12),
        "",
        "## 11. Direction Results",
    ]
    for construct, table in direction_tables.items():
        lines.extend(
            [
                f"### {construct}",
                markdown_table(
                    table,
                    ["baseline_type", "split", "decoupling_axis", "layer", "accuracy", "mean_signed_projection"],
                    12,
                ),
            ]
        )
    lines.append("")
    lines.append("## 12. Probe Results")
    for construct, table in probe_tables.items():
        lines.extend(
            [
                f"### {construct}",
                markdown_table(
                    table,
                    ["split", "decoupling_axis", "layer", "accuracy", "direction_accuracy_reference"],
                    12,
                ),
            ]
        )
    lines.append("")
    lines.append("## 13. Patching Results")
    for construct, table in patching_tables.items():
        lines.extend(
            [
                f"### {construct}",
                markdown_table(
                    table,
                    ["pair_id", "decoupling_axis", "top_layer", "top_position", "top_recovery", "axis_top_site_stability"],
                    12,
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 14. Construct Mismatch Matrix",
            summarize_matrix(matrix),
            "",
            "Object classifications:",
            markdown_table(classifications, ["construct", "method", "object_classification"], 20),
            "",
            "## 15. Failure-mode Taxonomy",
            "- `ordinary_only_proxy`: ordinary heldout passes, but one or more decoupling axes fail.",
            "- `causal_but_nonspecific_handle`: steering moves the target logits but causes high KL collateral disruption.",
            "- `predictive_noncausal_detector`: a probe predicts class information without supporting a causal claim.",
            "- `prompt_local_dependency`: patching works for individual prompt pairs but top sites are unstable.",
            "- `no_reliable_object`: ordinary behavior or method signal is too weak to interpret.",
            "- `behavior_absent_or_weak`: GPT-2 Small did not show enough target behavior to support MI analysis.",
            "",
            "## 16. Ordinary Success That Failed Under Decoupling",
            "These cases are identified by matrix rows where `ordinary_heldout` is `pass` or `weak` and at least one decoupling axis is `fail`. They should be read as construct-validity failures, not necessarily method bugs.",
            "",
            "## 17. Causal Steering Worked but Was Nonspecific",
            "Direction steering is summarized separately from prediction. When target-logit movement appears with high KL divergence, it is classified as nonspecific causal control rather than a clean construct handle.",
            "",
            "## 18. Probes Detected Information That Steering Did Not Control",
            "Probe success is treated as predictive evidence only. The report does not infer causal control from probe accuracy, even when it exceeds direction accuracy.",
            "",
            "## 19. Patching Appeared Prompt-local",
            "Patching top sites are evaluated for stability across examples. Unstable top sites are treated as prompt-local dependencies rather than stable construct variables.",
            "",
            "## 20. Limitations",
            "The dataset is small, manually templated, and English-only. Patching runs on a small subset for speed. GPT-2 Small target-token behavior may not align with human labels on all decoupled examples, and weak behavior is not converted into success language.",
            "",
            "## 21. Recommended Next Experiment",
            "Keep GPT-2 Small fixed and improve the certainty dataset with a second manual pass focused on format-shift and nonlexical uncertainty. Then rerun the same matrix to test whether failures persist after reducing ambiguous examples.",
            "",
        ]
    )

    report_path = root / "reports" / "construct_mismatch_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
