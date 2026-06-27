# Mechanistic Workbench

Mechanistic Workbench (`mwb`) is a local-first, IPython-native mechanistic interpretability workbench.

This repository contains the Phase 0 workbench loop:

```bash
uv run mwb init
uv run mwb ipython
```

The implementation tracks typed mechanistic objects, session provenance, local artifacts, backend identity, evidence tiers, diagnosis trees, materialized next probes, MechanismCards, and draft claim checks.

## Common Commands

```bash
uv sync
uv run mwb init --name self-ground
uv run mwb doctor
uv run mwb demo negation --model EleutherAI/pythia-70m-deduped --device cpu
uv run mwb ingest self-ground /home/home/p/g/n/learning/ml_research/self-ground/runs/e004_specificity_rescue_matrix
uv run mwb card latest
uv run mwb diagnose latest
uv run mwb next-probe latest
uv run mwb next-probe latest --materialize
uv run mwb run-probe .mechanism/runs/<run_ref>/probe.yaml
uv run mwb claim check docs/fixtures/claim_association.json
uv run mwb draft-check docs/fixture_draft.md
uv run mwb graph rebuild
uv run mwb ledger validate
uv run mwb hypothesis transition <hypothesis-ref> --to-state triaged
uv run mwb space check docs/fixtures/space_check_valid.json
uv run mwb compile hypothesis docs/fixtures/hypothesis_phase5.json
uv run mwb verify docs/fixtures/hypothesis_phase5.json --diagnostic-only --dry-run
uv run mwb bundle audit negation_phase3_calibrated
uv run mwb benchmark framework
uv run mwb policy check
uv run mwb rebuild-index --output .mechanism/workbench.rebuilt.sqlite
```

See `docs/USAGE.md`, `docs/EVIDENCE_GRAPH.md`, `docs/LEDGERS.md`, `docs/HYPOTHESIS_LIFECYCLE.md`, `docs/SPACE_TYPES.md`, `docs/STATIC_COMPILER.md`, `docs/CAUSAL_VERIFICATION.md`, `docs/EXAMPLE_GEOMETRY.md`, `docs/DIAGNOSIS_AND_PROBES.md`, `docs/REFERENCE_MECHANISMS.md`, `docs/CLAIM_GRAMMAR.md`, `docs/POLICY_PROFILES.md`, `docs/FUNDAMENTAL_FUNCTIONALITY_CHECKLIST.md`, `docs/world_class_buildout/README.md`, `docs/PHASE0_ACCEPTANCE_REPORT.md`, and `docs/PHASE10_COMPLETION_REPORT.md` for the completed workflow, evidence graph, research ledgers, hypothesis lifecycle, space type system, static compiler, causal verification, example geometry, diagnosis/probe workflows, reference benchmarks, claim grammar, policy profiles, source-traced buildout plan, and dogfood evidence boundary.
