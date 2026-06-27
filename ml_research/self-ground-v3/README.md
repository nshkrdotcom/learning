# Mechanistic Workbench

Mechanistic Workbench (`mwb`) is a local-first, IPython-native mechanistic interpretability workbench.

This repository contains the Phase 0 workbench loop:

```bash
uv run mwb init
uv run mwb ipython
```

The implementation tracks typed mechanistic objects, session provenance, local artifacts, backend identity, evidence tiers, next-probe recommendations, MechanismCards, and draft claim checks.

## Common Commands

```bash
uv sync
uv run mwb init --name self-ground
uv run mwb doctor
uv run mwb demo negation --model EleutherAI/pythia-70m-deduped --device cpu
uv run mwb ingest self-ground /home/home/p/g/n/learning/ml_research/self-ground/runs/e004_specificity_rescue_matrix
uv run mwb card latest
uv run mwb next-probe latest
uv run mwb draft-check docs/fixture_draft.md
uv run mwb rebuild-index --output .mechanism/workbench.rebuilt.sqlite
```

See `docs/USAGE.md`, `docs/FUNDAMENTAL_FUNCTIONALITY_CHECKLIST.md`, `docs/PHASE0_ACCEPTANCE_REPORT.md`, and `docs/PHASE10_COMPLETION_REPORT.md` for the completed Phase 0 workflow and dogfood evidence boundary.
