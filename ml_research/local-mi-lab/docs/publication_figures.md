# Publication-Style Figures

This repo keeps regenerated experiment artifacts under `runs/` and `reports/`, but publication-style figures can be tracked separately when they summarize a completed practice result.

## Head-Specific Induction Causality v1

Regenerate the vector figures with:

```bash
uv run python scripts/make_publication_figures.py \
  --report-dir reports/head_specific_induction_causality_v1 \
  --output-dir figures/head_specific_induction_causality_v1
```

Inputs:

- `reports/head_specific_induction_causality_v1/head_specific_multiseed_by_head.csv`
- `reports/head_specific_induction_causality_v1/head_specific_multiseed_summary.json`
- `reports/head_specific_induction_causality_v1/run_manifest.json`
- `reports/head_specific_induction_causality_v1/replicated_head_L7H7_examples.csv`

Tracked outputs:

- `figures/head_specific_induction_causality_v1/figure_1_multiseed_candidate_gaps.svg`
- `figures/head_specific_induction_causality_v1/figure_2_seed_status_counts.svg`
- `figures/head_specific_induction_causality_v1/figure_3_candidate_group_outcomes.svg`
- `figures/head_specific_induction_causality_v1/figure_4_l7h7_example_effects.svg`
- matching `.pdf` files
- `figures/head_specific_induction_causality_v1/manifest.json`

These figures are for communication and inspection. They summarize local practice artifacts and do not establish an induction-head mechanism, a circuit, or a broad GPT-2 claim.
