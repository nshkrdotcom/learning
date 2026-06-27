# Mechanistic Workbench Usage

This is the accepted Phase 0 local workflow for the `self-ground-v3` repository.

## Setup

```bash
uv sync
uv run mwb init --name self-ground
uv run mwb doctor
```

Runtime state is written under `.mechanism/`. That directory is intentionally ignored by Git.

## Scratch Work

Launch IPython with Workbench context:

```bash
uv run mwb ipython
```

Run a captured one-cell session:

```bash
uv run mwb ipython --execute "bundle = ctx.domains.negation.load('phase3_calibrated')"
uv run mwb inspect session latest
```

Resume from an existing captured session:

```bash
uv run mwb ipython --resume <session-ref> --execute "note = ctx.note('resumed work')"
```

`ctx.note(...)` creates a typed note object, and `ctx.record(obj, name=...)` returns a labeled typed object copy that is captured when bound in IPython.

## Sweep Artifacts

Dry-run sweeps write a full non-claim-bearing artifact set under `.mechanism/runs/<run_ref>/`:

```bash
uv run mwb sweep docs/fixtures/hypothesis_phase5.json \
  --axis layer=0,1 \
  --axis feature_selection_mode=top-absolute \
  --axis operation=ablate \
  --axis patch_mode=direct \
  --axis amplification_factor=1.0 \
  --axis control_family=negation_removed \
  --dry-run
```

The emitted files include `sweep_config.json`, `run_manifest.json`, `verification_results.jsonl`, `intervention_receipts.jsonl`, `control_metrics.json`, and `blocker_report.json`.

## Backend Checks

```bash
uv run mwb adapter conformance transformer-lens --model EleutherAI/pythia-70m-deduped --device cpu
uv run mwb adapter conformance saelens --model EleutherAI/pythia-70m-deduped --hook blocks.2.hook_resid_post --device cpu
```

## SELF-GROUND Dogfood

Run the built-in negation demo:

```bash
uv run mwb demo negation --model EleutherAI/pythia-70m-deduped --device cpu
```

Ingest the E004 artifact set:

```bash
uv run mwb ingest self-ground /home/home/p/g/n/learning/ml_research/self-ground/runs/e004_specificity_rescue_matrix
```

Inspect the latest generated evidence hygiene outputs:

```bash
uv run mwb card latest
uv run mwb next-probe latest
uv run mwb draft-check docs/fixture_draft.md
```

The accepted E004 posture is `insufficient_evidence` with `control_leaky` as the primary blocker. The workbench preserves that boundary in the generated card and draft guard.

## Evidence Graph

Rebuild graph edges from file-backed workbench records:

```bash
uv run mwb graph rebuild
```

Run focused provenance and evidence queries:

```bash
uv run mwb graph query claims-depending-on <unit-or-object-ref>
uv run mwb graph query controls-contradicting <run-ref>
uv run mwb graph query cells-producing <artifact-ref>
uv run mwb graph query debt-blocking <claim-ref>
```

Graph edges are written to `.mechanism/graph/evidence_edges.jsonl` and indexed in SQLite. See `docs/EVIDENCE_GRAPH.md` for the schema and claim boundary.

## Research Ledgers

Validate Git-visible ledgers and refresh their SQLite index rows:

```bash
uv run mwb ledger validate
```

Generate human-reviewable proposals from local artifacts:

```bash
uv run mwb ledger propose-run <run-ref>
uv run mwb ledger propose-claim <card-ref>
```

Committed ledgers live under `research/logs/`. Proposal files stay under `.mechanism/` until reviewed. See `docs/LEDGERS.md` for schemas and parser rules.

## Hypothesis Lifecycle

Record workflow state separately from evidence tier and claim status:

```bash
uv run mwb hypothesis transition <hypothesis-ref> --to-state triaged
```

Generate live alternative explanations from blocker reports:

```bash
uv run mwb hypothesis explain <run-ref>
```

Promotion to `claimable` requires `--approved-by` and `--decision-ref`. See `docs/HYPOTHESIS_LIFECYCLE.md` for states, transition rules, and alternative-explanation outputs.

## Space Types

Check tensor-space and mechanistic-unit compatibility before a patch, projection, or comparison:

```bash
uv run mwb space check docs/fixtures/space_check_valid.json
```

Space checks write `.mechanism/space_checks/latest_space_check.json` and block incompatible dictionaries, pre-LN/post-LN mismatches without transforms, wrong-hook patches, and invalid unit operations. See `docs/SPACE_TYPES.md`.

## Rebuild Check

Rebuild a separate SQLite index from file-backed `.mechanism` records:

```bash
uv run mwb rebuild-index --output .mechanism/workbench.rebuilt.sqlite
```

The canonical recovery alias from the source archive is also available:

```bash
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
```

## Quality Gate

```bash
uv run ruff check .
uv run pytest
MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_phase4_context.py -m integration
uv run mwb doctor
```

For the source-traced fundamental checklist, see `docs/FUNDAMENTAL_FUNCTIONALITY_CHECKLIST.md`.

For the mined world-class buildout docset, including findings, target architecture, phased TDD/RGR checklist, and QC/commit/push protocol, see `docs/world_class_buildout/README.md`.
