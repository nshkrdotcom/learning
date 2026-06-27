# Implementation Plan

## Strategy

Build outward from the current Phase 0 workbench without replacing it:

1. harden the evidence graph and flat-file ledger layer;
2. model investigation lifecycle separately from claims;
3. deepen static preflight into a mechanistic compiler;
4. add exact causal verification operations with real adapters;
5. build deterministic diagnosis and probe materialization;
6. add reference mechanism tasks;
7. only then expand adapters and platform integrations.

Every implementation phase uses:

```text
failing characterization test
minimal implementation
regression test
real or source-backed dogfood fixture
docs update
QC-green
commit
push
```

## Buildout Streams

### Stream A: Evidence Core

Goal:

Make the provenance graph queryable and reconstructable.

Deliverables:

- typed graph edge model,
- graph query CLI,
- graph rebuild from `.mechanism` files,
- contradiction/support edges,
- claim/debt graph views,
- exported graph summaries.

Why first:

Most later features need reliable joins over cells, objects, artifacts, runs, controls, claims, and debt.

### Stream B: Git-Native Research Ledgers

Goal:

Add human-approved, committed research records.

Deliverables:

- `research/logs/claim_ledger.md`,
- `research/logs/run_ledger.csv`,
- `research/logs/decision_log.md`,
- `research/logs/research_log.md`,
- ledger parsers,
- ledger validators,
- proposal generation from local runs,
- human-approval workflow.

Why second:

The mined tracker docs identify the claim ledger as the center of gravity. The workbench needs this while preserving IPython-first exploration.

### Stream C: Hypothesis Lifecycle

Goal:

Represent research search state before claims.

Deliverables:

- HypothesisState model,
- transition commands,
- state transition receipts,
- alternative-explanation records,
- promotion gate from hypothesis to claim candidate,
- retirement/abandonment records.

Why:

Current evidence tier/status is too paper-facing. Researchers need to know what is live, killed, or still confounded.

### Stream D: Static Mechanistic Compiler

Goal:

Kill structurally bad hypotheses before GPU sweeps.

Deliverables:

- TensorRef and expanded TensorSpace compatibility,
- static check registry,
- decoder-unembed implementation over real model matrices,
- dictionary neighbor geometry,
- DLA/logit-lens checks,
- OV/QK check scaffolding,
- attribution proxy checks where real backend support exists,
- plausibility gate.

Why:

This is the compute-saving differentiator repeatedly identified by the mined framework notes.

### Stream E: Exact Verification Engine

Goal:

Turn dry-run planning into real causal verification receipts.

Deliverables:

- resample ablation,
- noising/denoising,
- feature amplification,
- direct patch,
- controlled target/control metrics,
- telemetry and drift checks,
- intervention receipts,
- PredictionLock gate,
- claim-bearing evidence gate.

Why:

MechanismCard evidence tiers cannot become robust without real interventions beyond ingested external artifacts.

### Stream F: Example Geometry

Goal:

Make task/control quality measurable before claims.

Deliverables:

- bundle audit CLI,
- token validity checks,
- baseline margin checks,
- role separability metrics,
- contaminated-control detection,
- rebalance recommendations,
- heldout bundle materialization.

Why:

SELF-GROUND failures show task/control geometry is a first-order scientific risk.

### Stream G: Diagnosis And Probe Synthesis

Goal:

Turn failures into ranked, materializable next experiments.

Deliverables:

- AlternativeExplanation planner,
- deterministic probe templates,
- `mwb diagnose`,
- `mwb next-probe --materialize`,
- `mwb run-probe`,
- probe provenance refs.

Why:

The workbench becomes more than a linter when it teaches researchers how to falsify themselves faster.

### Stream H: Reference Mechanism Suite

Goal:

Test the framework against known mechanisms and false positives.

Deliverables:

- toy residual-direction fixtures,
- synthetic SAE dictionary fixtures,
- Tracr/RASP task integration if dependency accepted,
- IOI/path-patching reference fixture,
- negative-control confound suite,
- benchmark report.

Why:

Known-ground-truth tests are required before trusting generalized mechanism verification.

### Stream I: Rich Claim Grammar

Goal:

Make Draft Guard evidence-aware by claim type.

Deliverables:

- claim grammar parser,
- sentence intent classifier constrained to deterministic patterns,
- evidence requirement resolver,
- caveat/debt renderer,
- blocked-language and missing-evidence explanations,
- claim update proposals.

Why:

Phrase matching catches obvious overclaims, but claim grammar can explain exactly what evidence is missing.

### Stream J: Adapter Expansion

Goal:

Add backends only after core evidence behavior is solid.

Deliverables:

- nnsight/nnterp adapter,
- pyvene adapter,
- Neuronpedia read-only adapter,
- DVC/git-annex/Git LFS artifact pointer support,
- Inspect/lm-eval interop later.

Why last:

Adapters without evidence graph, claim grammar, and reference conformance just create a fragile wrapper pile.

## Cross-Cutting Requirements

### TDD/RGR

Each feature begins with at least one failing test that encodes source-mined behavior.

Accepted test sources:

- existing `.mechanism` dogfood artifacts,
- SELF-GROUND E001-E004 history,
- canonical 20260625 docs,
- known-ground-truth reference tasks,
- real TransformerLens/SAELens integration commands.

Rejected acceptance sources:

- smoke-only tests,
- synthetic-only "happy path" fixtures,
- mocked backend support presented as real support,
- prose-only done criteria.

### Documentation

Every phase updates:

- user-facing usage docs,
- implementation checklist/ledger,
- relevant schema docs,
- evidence boundary notes.

### Quality Gates

Minimum every phase:

```bash
uv sync
uv run ruff check .
uv run pytest
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
git status --short --branch
```

Integration phases additionally run real adapter checks.

### Commit And Push

Every phase ends with:

```bash
git add <scoped files>
git commit -m "phase N: <specific outcome>"
git push
```

The ledger records:

- source docs read,
- tests added,
- commands run,
- observed results,
- commit hash,
- push result,
- residual risk.
