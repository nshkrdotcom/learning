# Phase 0 Acceptance Report

Status: accepted for local dogfood.

This report records the Phase 0 acceptance state for the local-first Mechanistic Workbench implementation in this repository.

## Accepted Surface

- Project initialization and health checks: `mwb init`, `mwb doctor`.
- IPython-native scratch sessions with captured cells and typed workbench objects.
- TransformerLens and SAELens adapter conformance with real backend loads.
- Built-in SELF-GROUND negation bundle loading and real activation capture.
- Hypothesis preflight, prediction-lock checks, dry-run verification, sweep planning, blocker diagnosis, and next-probe planning.
- MechanismCard generation, claim registry writes, and draft claim checks.
- SELF-GROUND E004 artifact ingestion into `.mechanism/runs`.
- Resume-aware IPython sessions, full dry-run sweep artifacts, scientific debt records, and SQLite rebuild checks.

## Real Dogfood Artifact

Source:

```text
/home/home/p/g/n/learning/ml_research/self-ground/runs/e004_specificity_rescue_matrix
```

Ingested run:

```text
run_self_ground_e004_specificity_rescue_matrix
```

Accepted interpretation:

- Status: `insufficient_evidence`
- Primary blocker: `control_leaky`
- Evidence tier: `association`
- Claim-bearing mechanism evidence: no

The E004 ingest preserves the artifact's own adjudication. It does not convert a control-leaky or insufficient result into a mechanism claim.

## Acceptance Commands

```bash
uv run mwb demo negation --model EleutherAI/pythia-70m-deduped --device cpu
uv run mwb ingest self-ground /home/home/p/g/n/learning/ml_research/self-ground/runs/e004_specificity_rescue_matrix
uv run mwb card latest
uv run mwb next-probe latest
uv run mwb draft-check docs/fixture_draft.md
uv run mwb rebuild-index --output .mechanism/workbench.rebuilt.sqlite
uv run mwb doctor
uv run ruff check .
uv run pytest
```

Observed result:

- Real model demo passed and captured activation shape `[2, 6, 512]`.
- Ingest passed and wrote a normalized workbench run.
- Card and next-probe generation passed through `latest`.
- Draft guard passed for allowed association-tier language.
- SQLite rebuild passed from file-backed `.mechanism` records.
- Doctor passed with `status: ok`.
- Ruff passed.
- Pytest passed with `50 passed, 1 skipped`.

## Release Boundary

Phase 0 is implementation-ready for local dogfood and reproducible evidence hygiene. It is not a claim that the E004 artifacts support a broad negation mechanism; the generated card explicitly blocks that wording.
