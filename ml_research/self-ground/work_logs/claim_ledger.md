# Claim Ledger

## Claim: Negation-Scope SAE Feature Set Candidate

- current status: `insufficient_evidence`
- latest run: `runs/diagnostic_negation_ravel_eval_density_matched`
- latest report: `runs/diagnostic_negation_ravel_eval_density_matched/mechanism_report.json`
- latest inspection: `uv run python scripts/inspect_claim_run.py --run-dir runs/diagnostic_negation_ravel_eval_density_matched`

The latest diagnostic run is artifact-backed and uses the real
TransformerLens + SAELens decoded SAE path, but it does not support candidate
evidence:

- target delta is zero,
- matched-control delta is zero,
- specificity gap is zero,
- run scale is diagnostic (`cpu`, `per_family=2`, `top_k_features=2`),
- density matching is approximate and relaxed.

No broad mechanism discovery, monosemanticity, broad behavioral understanding,
or genuine introspection claim is supported.

## Next Promotion Requirement

Run E002 on a GPU-scale setting:

- `per_family=10`,
- `top_k_features=5`,
- `top-vs-random-density-and-bottom-active`,
- semantic SAE compatibility passing,
- density-matched controls present,
- no skipped rows,
- threshold checks passing in `mechanism_report.json`.
