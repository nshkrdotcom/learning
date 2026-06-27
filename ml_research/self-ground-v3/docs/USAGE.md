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
uv run mwb diagnose latest
uv run mwb next-probe latest
uv run mwb next-probe latest --materialize
uv run mwb draft-check docs/fixture_draft.md
```

The accepted E004 posture is `insufficient_evidence` with `control_leaky` as the primary blocker. The workbench preserves that boundary in the generated card and draft guard.

## Diagnosis And Probes

Write a provenance-preserving diagnosis tree:

```bash
uv run mwb diagnose latest
```

Materialize the deterministic next probe:

```bash
uv run mwb next-probe latest --materialize
```

Run an implemented probe through the local workflow runner:

```bash
uv run mwb run-probe .mechanism/runs/<run_ref>/probe.yaml
```

Diagnosis writes `diagnosis_tree.json`, `diagnosis_tree.yaml`, and `diagnosis_tree.md`. Probe materialization writes `probe.json`, `probe.yaml`, and `probe.md`. Unsupported probe kinds are recorded as blocked materialized probes with no command and are rejected by `run-probe`. See `docs/DIAGNOSIS_AND_PROBES.md`.

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

## Claim Grammar

Check a single paper-facing claim fixture:

```bash
uv run mwb claim check docs/fixtures/claim_association.json
```

Draft Guard runs typed claim grammar before the phrase fallback:

```bash
uv run mwb draft-check docs/fixture_draft.md
```

Claim checks write `.mechanism/claims/<claim_ref>_grammar_report.json` and block stronger language when evidence requirements, blockers, or unresolved scientific debt do not support the requested claim type. See `docs/CLAIM_GRAMMAR.md`.

## Policy Profiles

Evaluate the default research-taste profile:

```bash
uv run mwb policy check
```

Evaluate a named profile:

```bash
uv run mwb policy check --profile strict
uv run mwb policy check --profile exploratory
```

Policy profiles control claim ceilings and research-taste gates such as zero-ablation ceilings, paired noising/denoising requirements, resample-ablation requirements, and generalization-before-mechanism wording. See `docs/POLICY_PROFILES.md`.

## Space Types

Check tensor-space and mechanistic-unit compatibility before a patch, projection, or comparison:

```bash
uv run mwb space check docs/fixtures/space_check_valid.json
```

Space checks write `.mechanism/space_checks/latest_space_check.json` and block incompatible dictionaries, pre-LN/post-LN mismatches without transforms, wrong-hook patches, and invalid unit operations. See `docs/SPACE_TYPES.md`.

## Static Compiler

Run static algebraic plausibility checks before claim-bearing verification:

```bash
uv run mwb compile hypothesis docs/fixtures/hypothesis_phase5.json
```

The compiler writes `.mechanism/static_compiler/latest_static_compile.json`, indexes top-level reports plus per-check rows, and blocks claim-bearing verification when the gate is `FAIL`. See `docs/STATIC_COMPILER.md`.

## Causal Verification

Run exact verification operations and write receipts:

```bash
uv run mwb verify docs/fixtures/hypothesis_phase5.json --diagnostic-only
```

Dry-run verification remains non-claim-bearing:

```bash
uv run mwb verify docs/fixtures/hypothesis_phase5.json --diagnostic-only --dry-run
```

Verification writes `.mechanism/runs/<run_ref>/verification_run.json`, `intervention_receipts.jsonl`, `verification_results.jsonl`, and `telemetry.jsonl`. Claim-bearing runs require a prediction lock and a passing static compiler gate. See `docs/CAUSAL_VERIFICATION.md`.

## Example Geometry

Audit target/control bundle geometry:

```bash
uv run mwb bundle audit negation_phase3_calibrated
uv run mwb bundle rebalance --dry-run
```

The audit checks token validity, role balance, contaminated controls, and baseline margins, then proposes heldout/control-balance improvements. See `docs/EXAMPLE_GEOMETRY.md`.

## Reference Mechanisms

Run the built-in framework benchmark suite against deterministic known-ground-truth tasks:

```bash
uv run mwb benchmark framework
```

The benchmark writes `.mechanism/benchmarks/latest_framework_benchmark.json`, indexes `benchmark_reports` and `reference_tasks`, and checks planted mechanism recovery, false-positive blocking, synthetic SAE split detection, synthetic SAE absorption detection, and calibration fields. See `docs/REFERENCE_MECHANISMS.md`.

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
