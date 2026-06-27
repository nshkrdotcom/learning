# Phased TDD/RGR Checklist

This checklist begins after the current completed Phase 0/10/11 baseline. The source-mining docset itself was recorded as repo Phase 12, so completed buildout phases after it are recorded in the repo ledger with the next available repo phase number.

Every phase must be implemented with TDD/RGR, docs updates, QC-green, commit, and push.

## Source Legend

When a phase references a short source name, resolve it through this legend:

- `CANONICAL`: `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625`
- `TRACKER`: `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260624/ml_research/mechinterp_tracker`
- `MI_DOCS`: `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs`

For example, `0005.md` means `CANONICAL/0005.md`; `0430_revised_v6.md` means `TRACKER/0430_revised_v6.md`; and `mechinterp_framework/0020_gpt.md` means `MI_DOCS/mechinterp_framework/0020_gpt.md`.

## Phase 12: Evidence Graph Query Core

Status: complete in repo Phase 13 commit `e20d307`, pushed yes.

### Required Reading

- `0003.md` Evidence Core and persistence sections.
- `0004.md` object registration and recovery sections.
- `mechinterp_framework/0020_gpt.md` "Evidence as a typed causal graph".
- `0430_revised_v6.md` SQLite-never-canonical and run ledger sections.

### TDD/RGR

- [x] Add failing tests for typed edges: `supports`, `contradicts`, `depends_on`, `derived_from`, `tested_by`, `confounded_by`, `fails_on`, `generalizes_to`, `cited_by`.
- [x] Add failing tests for graph rebuild from file-backed records.
- [x] Add failing tests for graph query CLI:
  - [x] claims depending on a unit,
  - [x] controls contradicting a run,
  - [x] cells producing an artifact,
  - [x] debt blocking a claim.

### Implementation

- [x] Add `EvidenceEdge` domain object.
- [x] Add `EvidenceGraphService`.
- [x] Add `mwb graph query`.
- [x] Add `mwb graph rebuild`.
- [x] Persist graph edges in JSONL and SQLite.
- [x] Keep SQLite rebuildable.

### Docs

- [x] Update `docs/USAGE.md`.
- [x] Add graph schema docs.
- [x] Update phase ledger.

### QC Gate

```bash
uv sync
uv run ruff check .
uv run pytest
uv run mwb graph rebuild
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
git status --short --branch
```

### Commit / Push

```bash
git add .
git commit -m "phase 13: add evidence graph query core"
git push
```

## Phase 13: Git-Native Research Ledgers

### Required Reading

- `0430_revised_v6.md` claim ledger, run ledger, decision log, research log.
- `0431_selfground_refactor.md`.
- `0432_selfground_refactor.md`.
- `mechinterp_tracker/0300_research_landscape_for_git_native_research_integrity_systems.md`.

### TDD/RGR

- [ ] Add failing parser tests for `research/logs/claim_ledger.md`.
- [ ] Add failing parser tests for `research/logs/run_ledger.csv`.
- [ ] Add failing parser tests for `research/logs/decision_log.md`.
- [ ] Add failing tests for run-to-ledger proposal generation.
- [ ] Add failing tests that SQLite rebuild does not lose ledger state.

### Implementation

- [ ] Add `research/` scaffold.
- [ ] Add claim ledger schema.
- [ ] Add run ledger schema.
- [ ] Add decision log schema.
- [ ] Add research log schema.
- [ ] Add `mwb ledger validate`.
- [ ] Add `mwb ledger propose-run <run-ref>`.
- [ ] Add `mwb ledger propose-claim <card-ref>`.
- [ ] Add human-reviewable proposal files.

### Docs

- [ ] Add `docs/LEDGERS.md`.
- [ ] Add templates under `research/`.
- [ ] Update README/USAGE.
- [ ] Update phase ledger.

### QC Gate

```bash
uv sync
uv run ruff check .
uv run pytest
uv run mwb ledger validate
uv run mwb doctor
git status --short --branch
```

### Commit / Push

```bash
git add .
git commit -m "phase 13: add git-native research ledgers"
git push
```

## Phase 14: Hypothesis Lifecycle And Alternative Explanations

### Required Reading

- `0005.md` research modes and canonical workflow.
- `mechinterp_framework/0020_gpt.md` hypothesis state machine and alternative-explanation engine.
- `0020_critique_claude.md` claim taxonomy critique.

### TDD/RGR

- [ ] Add failing tests for workflow state separate from evidence tier.
- [ ] Add failing tests for valid/invalid hypothesis transitions.
- [ ] Add failing tests for live alternative explanations from blocker metrics.
- [ ] Add failing tests for human approval requirement on claim promotion.

### Implementation

- [ ] Add `HypothesisState`.
- [ ] Add `AlternativeExplanation`.
- [ ] Add transition receipts.
- [ ] Add `mwb hypothesis transition`.
- [ ] Add `mwb hypothesis explain`.
- [ ] Add claim promotion proposal, not automatic promotion.

### Docs

- [ ] Add lifecycle docs.
- [ ] Update MechanismCard docs to reference lifecycle state.
- [ ] Update phase ledger.

### QC Gate

```bash
uv sync
uv run ruff check .
uv run pytest
uv run mwb hypothesis explain <fixture-hypothesis>
uv run mwb doctor
git status --short --branch
```

### Commit / Push

```bash
git add .
git commit -m "phase 14: add hypothesis lifecycle and alternatives"
git push
```

## Phase 15: Mechanistic Space Type System

### Required Reading

- `0003.md` TensorSpace and MechanisticUnitRef.
- `mechinterp_framework/0020_gpt.md` space-typed tensors.
- `mechinterp_framework/0010_claude.md` static compiler.

### TDD/RGR

- [ ] Add failing tests for incompatible SAE dictionary comparisons.
- [ ] Add failing tests for pre-LN/post-LN projection mismatch.
- [ ] Add failing tests for wrong-hook patch target.
- [ ] Add failing tests for explicit transform provenance.
- [ ] Add failing tests for MechanisticUnit valid/invalid operation registry.

### Implementation

- [ ] Add `TensorRef`.
- [ ] Expand `TensorSpace`.
- [ ] Add `SpaceCompatibilityReport`.
- [ ] Add transform registry.
- [ ] Add `MechanisticUnitRegistry`.
- [ ] Add `mwb space check`.

### Docs

- [ ] Add `docs/SPACE_TYPES.md`.
- [ ] Update adapter docs.
- [ ] Update phase ledger.

### QC Gate

```bash
uv sync
uv run ruff check .
uv run pytest
uv run mwb space check <fixture>
uv run mwb doctor
git status --short --branch
```

### Commit / Push

```bash
git add .
git commit -m "phase 15: add mechanistic space type system"
git push
```

## Phase 16: Static Mechanistic Compiler

### Required Reading

- `0005.md` static preflight.
- `mechinterp_framework/0010_claude.md` compiler/static algebra.
- `mechinterp_framework/0020_gpt.md` static plausibility requirements.

### TDD/RGR

- [ ] Add failing tests for real decoder-unembedding cosine calculation.
- [ ] Add failing tests for dictionary neighbor interference.
- [ ] Add failing tests for activation density warnings.
- [ ] Add failing tests for plausibility gate aggregation.
- [ ] Add failing tests that failed static gate blocks claim-bearing verification.

### Implementation

- [ ] Add `StaticCheckResult`.
- [ ] Add compiler check registry.
- [ ] Implement real decoder/unembed projection over TransformerLens model identity.
- [ ] Implement dictionary neighbor geometry for SAELens dictionaries where available.
- [ ] Implement plausibility gate.
- [ ] Add `mwb compile hypothesis`.

### Docs

- [ ] Add `docs/STATIC_COMPILER.md`.
- [ ] Update preflight docs.
- [ ] Update phase ledger.

### QC Gate

```bash
uv sync
uv run ruff check .
uv run pytest
MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_static_compiler_integration.py -m integration
uv run mwb compile hypothesis docs/fixtures/hypothesis_phase5.json
uv run mwb doctor
git status --short --branch
```

### Commit / Push

```bash
git add .
git commit -m "phase 16: add static mechanistic compiler"
git push
```

## Phase 17: Exact Causal Verification Operations

### Required Reading

- `0005.md` causal verification.
- `mechinterp_framework/0020_gpt.md` causal engine and research taste policies.
- `0010_mechinterp_tracker_gpt.md` activation patching, steering, SAE requirements.

### TDD/RGR

- [ ] Add failing tests for resample ablation receipts.
- [ ] Add failing tests for noising and denoising distinction.
- [ ] Add failing tests for feature amplification.
- [ ] Add failing tests for telemetry drift checks.
- [ ] Add failing tests that zero ablation has a lower claim ceiling unless policy allows it.
- [ ] Add failing real integration test on Pythia-70M small bundle.

### Implementation

- [ ] Implement resample ablation through TransformerLens/SAELens path.
- [ ] Implement noising/denoising receipts.
- [ ] Implement feature amplification receipts.
- [ ] Implement KL/norm drift telemetry.
- [ ] Write verification artifacts.
- [ ] Enforce PredictionLock for claim-bearing exact runs.

### Docs

- [ ] Add `docs/CAUSAL_VERIFICATION.md`.
- [ ] Update MechanismCard evidence examples.
- [ ] Update phase ledger.

### QC Gate

```bash
uv sync
uv run ruff check .
uv run pytest
MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_causal_verification_integration.py -m integration
uv run mwb verify docs/fixtures/hypothesis_phase5.json --diagnostic-only --dry-run
uv run mwb doctor
git status --short --branch
```

### Commit / Push

```bash
git add .
git commit -m "phase 17: add exact causal verification operations"
git push
```

## Phase 18: Example Geometry And Control Audits

### Required Reading

- `0005.md` controls and blockers.
- `mechinterp_framework/0020_gpt.md` first-class example geometry.
- SELF-GROUND E004 comparison/forensics artifacts.

### TDD/RGR

- [ ] Add failing tests for token validity audit.
- [ ] Add failing tests for role balance.
- [ ] Add failing tests for contaminated controls.
- [ ] Add failing tests for baseline margin checks.
- [ ] Add failing tests for heldout/control bundle proposal generation.

### Implementation

- [ ] Add `ExampleGeometryReport`.
- [ ] Add `ControlContaminationReport`.
- [ ] Add `mwb bundle audit`.
- [ ] Add `mwb bundle rebalance --dry-run`.
- [ ] Add ingest links from SELF-GROUND forensics to bundle audit outputs.

### Docs

- [ ] Add `docs/EXAMPLE_GEOMETRY.md`.
- [ ] Update bundle docs.
- [ ] Update phase ledger.

### QC Gate

```bash
uv sync
uv run ruff check .
uv run pytest
uv run mwb bundle audit negation_phase3_calibrated
uv run mwb doctor
git status --short --branch
```

### Commit / Push

```bash
git add .
git commit -m "phase 18: add example geometry audits"
git push
```

## Phase 19: Diagnosis Tree And Probe Materialization

### Required Reading

- `0005.md` next-probe planning.
- `mechinterp_framework/0020_gpt.md` mechanistic debugger and probe synthesis.
- `0430_revised_v6.md` scientific debt and negative evidence.

### TDD/RGR

- [ ] Add failing tests for diagnosis tree generation from blocker reports.
- [ ] Add failing tests for deterministic probe templates.
- [ ] Add failing tests for materialized `probe.yaml` provenance.
- [ ] Add failing tests that unsupported probe commands are not emitted.

### Implementation

- [ ] Add `DiagnosisTree`.
- [ ] Add probe template registry.
- [ ] Add `mwb diagnose`.
- [ ] Add `mwb next-probe --materialize`.
- [ ] Add `mwb run-probe <probe-yaml>` for implemented probes only.

### Docs

- [ ] Add `docs/DIAGNOSIS_AND_PROBES.md`.
- [ ] Update next-probe docs.
- [ ] Update phase ledger.

### QC Gate

```bash
uv sync
uv run ruff check .
uv run pytest
uv run mwb diagnose latest
uv run mwb next-probe latest --materialize
uv run mwb doctor
git status --short --branch
```

### Commit / Push

```bash
git add .
git commit -m "phase 19: add diagnosis tree and probe materialization"
git push
```

## Phase 20: Reference Mechanism Suite

### Required Reading

- `mechinterp_framework/0020_gpt.md` reference tasks with known ground truth.
- `mechinterp_framework/0010_claude.md` Tracr and calibration loop.
- `BEST_EVALS_github.md` eval registry quality patterns.

### TDD/RGR

- [ ] Add failing tests for toy known mechanism classification.
- [ ] Add failing tests for tempting false-positive confound blocking.
- [ ] Add failing tests for synthetic SAE split/absorption detection.
- [ ] Add failing tests for reference task report generation.

### Implementation

- [ ] Add reference task registry.
- [ ] Add small toy model fixtures or deterministic generated fixtures.
- [ ] Add negative controls.
- [ ] Add `mwb benchmark framework`.
- [ ] Add benchmark report artifacts.

### Docs

- [ ] Add `docs/REFERENCE_MECHANISMS.md`.
- [ ] Add benchmark contribution guide.
- [ ] Update phase ledger.

### QC Gate

```bash
uv sync
uv run ruff check .
uv run pytest
uv run mwb benchmark framework
uv run mwb doctor
git status --short --branch
```

### Commit / Push

```bash
git add .
git commit -m "phase 20: add reference mechanism suite"
git push
```

## Phase 21: Rich Claim Grammar

### Required Reading

- `0005.md` evidence tiers.
- `mechinterp_framework/0020_gpt.md` richer claim grammar.
- `0430_revised_v6.md` Draft Guard deterministic policy.

### TDD/RGR

- [ ] Add failing tests for observation claim requirements.
- [ ] Add failing tests for static claim requirements.
- [ ] Add failing tests for necessity/sufficiency/mediation/generalization/mechanism claim requirements.
- [ ] Add failing tests for required caveats and unresolved debt.
- [ ] Add failing tests for inline override visibility.

### Implementation

- [ ] Add claim grammar model.
- [ ] Add deterministic claim-intent matcher.
- [ ] Add evidence requirement resolver.
- [ ] Add `mwb claim check`.
- [ ] Upgrade `mwb draft-check` to use grammar before phrase fallback.

### Docs

- [ ] Add `docs/CLAIM_GRAMMAR.md`.
- [ ] Update Draft Guard docs.
- [ ] Update phase ledger.

### QC Gate

```bash
uv sync
uv run ruff check .
uv run pytest
uv run mwb draft-check docs/fixture_draft.md
uv run mwb claim check <fixture-claim>
uv run mwb doctor
git status --short --branch
```

### Commit / Push

```bash
git add .
git commit -m "phase 21: add rich claim grammar"
git push
```

## Phase 22: Policy Profiles And Research Taste

### Required Reading

- `mechinterp_framework/0020_gpt.md` research taste policies.
- `0005.md` evidence tiers and blockers.
- `0430_revised_v6.md` scientific debt policy.

### TDD/RGR

- [ ] Add failing tests for policy profiles changing claim ceilings.
- [ ] Add failing tests for zero-ablation ceiling.
- [ ] Add failing tests for required noising/denoising policy.
- [ ] Add failing tests for generalization-before-mechanism policy.

### Implementation

- [ ] Add policy profile schema.
- [ ] Add default strict profile.
- [ ] Add project config policy selection.
- [ ] Apply policy to verification, cards, and draft guard.

### Docs

- [ ] Add `docs/POLICY_PROFILES.md`.
- [ ] Update project config docs.
- [ ] Update phase ledger.

### QC Gate

```bash
uv sync
uv run ruff check .
uv run pytest
uv run mwb doctor
git status --short --branch
```

### Commit / Push

```bash
git add .
git commit -m "phase 22: add policy profiles"
git push
```

## Phase 23: Adapter Expansion With Conformance

### Required Reading

- `0006.md` adapter strategy.
- `0010_mechinterp_tracker_gpt.md` technique coverage.
- `mech.md` ecosystem survey.

### TDD/RGR

- [ ] Add failing conformance manifest tests before each adapter.
- [ ] Add failing diagnostic-only tests for missing optional backends.
- [ ] Add real integration tests only where backend is installed and configured.
- [ ] Add tests that unsupported adapters cannot be claim-bearing.

### Implementation

- [ ] Add nnsight/nnterp adapter when dependency is available.
- [ ] Add pyvene adapter when dependency is available.
- [ ] Add Neuronpedia read-only metadata adapter.
- [ ] Add DVC/git-annex/Git LFS artifact pointer support.
- [ ] Keep optional deps optional.

### Docs

- [ ] Add adapter guide.
- [ ] Add conformance matrix.
- [ ] Update phase ledger.

### QC Gate

```bash
uv sync
uv run ruff check .
uv run pytest
uv run mwb adapter conformance transformer-lens --model EleutherAI/pythia-70m-deduped --device cpu
uv run mwb adapter conformance saelens --model EleutherAI/pythia-70m-deduped --hook blocks.2.hook_resid_post --device cpu
uv run mwb doctor
git status --short --branch
```

### Commit / Push

```bash
git add .
git commit -m "phase 23: add adapter expansion conformance"
git push
```

## Phase 24: Release Hardening

### Required Reading

- Full world-class buildout docset.
- Current phase ledger.
- `BEST_EVALS_github.md` quality/review patterns.

### TDD/RGR

- [ ] Add regression tests for all previously fixed false positives/negatives.
- [ ] Add compatibility tests for reading old `.mechanism` state.
- [ ] Add docs-link tests if docs tooling exists.
- [ ] Add command help snapshot tests for public CLI.

### Implementation

- [ ] Run full QC.
- [ ] Run real integration gates.
- [ ] Run scan for fake/dummy/mock/smoke/placeholder.
- [ ] Run overclaim language scan.
- [ ] Rebuild SQLite and graph.
- [ ] Validate docs against runtime commands.
- [ ] Generate release report.

### Docs

- [ ] Add release report.
- [ ] Update README.
- [ ] Update phase ledger.

### QC Gate

```bash
uv sync
uv run ruff check .
uv run pytest
MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_phase4_context.py -m integration
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
rg -n "fake|dummy|mock|simulated|placeholder|smoke" src tests docs README.md pyproject.toml
rg -n "implements|mechanism for|proves|isolated.*circuit|strong_candidate_evidence" src tests docs README.md pyproject.toml
git status --short --branch
```

### Commit / Push

```bash
git add .
git commit -m "phase 24: harden world-class buildout release"
git push
```
