# Phase 10 Completion Report

Status: complete.

This report closes the literal checklist gaps remaining after the Phase 9 release candidate.

## Closed Items

- `mwb ipython --resume <session-ref>` now starts a new captured session linked by `resumed_from_session_ref`.
- `ctx.note(...)` and `ctx.record(...)` are covered by IPython capture tests.
- `mwb sweep` now writes the full dry-run artifact set:
  - `sweep_config.json`
  - `run_manifest.json`
  - `verification_results.jsonl`
  - `intervention_receipts.jsonl`
  - `control_metrics.json`
  - `blocker_report.json`
- Sweep artifacts are explicitly non-claim-bearing and do not fabricate causal evidence.
- MechanismCard language tables cover all Phase 0 evidence tiers:
  - association
  - projection
  - causal necessity
  - causal sufficiency
  - mediation
  - generalization
  - mechanism
- MechanismCard generation writes `scientific_debt.json`.
- Draft Guard now supports `allowed`, `caveated`, `blocked`, `unknown_claim`, and `missing_card` statuses.
- SELF-GROUND E004 ingest validates comparison CSVs and forensics CSVs in addition to summary JSON.
- `mwb rebuild-index` rebuilds a separate SQLite index from file-backed `.mechanism` records.

## Real Completion Commands

```bash
uv run mwb ipython --execute "note = ctx.note('resume-source')"
uv run mwb ipython --resume <session-ref> --execute "note = ctx.record(ctx.note('resume-target'), name='resumed-note')"
uv run mwb sweep docs/fixtures/hypothesis_phase5.json --axis layer=0,1 --axis feature_selection_mode=top-absolute --axis operation=ablate --axis patch_mode=direct --axis amplification_factor=1.0 --axis control_family=negation_removed --dry-run
uv run mwb ingest self-ground /home/home/p/g/n/learning/ml_research/self-ground/runs/e004_specificity_rescue_matrix
uv run mwb card latest
uv run mwb next-probe latest
uv run mwb draft-check docs/fixture_draft.md
uv run mwb rebuild-index --output .mechanism/workbench.rebuilt.sqlite
uv run mwb doctor
uv run ruff check .
uv run pytest
MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_phase4_context.py -m integration
uv run mwb adapter conformance transformer-lens --model EleutherAI/pythia-70m-deduped --device cpu
uv run mwb adapter conformance saelens --model EleutherAI/pythia-70m-deduped --hook blocks.2.hook_resid_post --device cpu
```

## Results

- Completion test suite: `8 passed`.
- Full test suite: `50 passed, 1 skipped`.
- Ruff: passed.
- Doctor: passed with `status: ok`.
- SQLite rebuild: passed with `status: ok`.
- Real adapter integration: passed.
- TransformerLens conformance: passed.
- SAELens conformance: passed.
- Non-real-work scan: no matches.
- Overclaim scan: matches are expected guard-table, test, and fixture occurrences. The real E004 card remains association-tier and blocks mechanism wording.

## Evidence Boundary

The completion work does not upgrade SELF-GROUND E004 evidence. E004 remains `insufficient_evidence`, `control_leaky`, non-claim-bearing, and bounded by association-tier draft language.
