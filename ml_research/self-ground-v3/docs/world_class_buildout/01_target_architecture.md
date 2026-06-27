# Target Architecture

## Architecture Thesis

Mechanistic Workbench should become a local-first mechanistic research operating system:

```text
IPython exploration
  -> typed mechanistic objects
  -> evidence graph
  -> hypothesis state machine
  -> static mechanistic compiler
  -> exact causal verification
  -> alternative-explanation diagnosis
  -> deterministic probe synthesis
  -> MechanismCards
  -> claim grammar and draft guard
```

It must not become:

- a new model runtime,
- a universal execution abstraction,
- a hosted dashboard-first tracker,
- an AI reviewer,
- a claim oracle,
- a raw tensor warehouse.

## Layer Model

### Presentation

Surfaces:

- IPython shell and extension,
- CLI,
- Markdown reports,
- MechanismCards,
- draft checks,
- future notebook/Jupyter and TUI surfaces.

Rules:

- Presentation may display and invoke application services.
- Presentation may not define scientific truth.
- Display helpers must not create or upgrade evidence.

### Application

Services:

- ProjectManager,
- SessionManager,
- RunContext,
- EvidenceGraphService,
- HypothesisLifecycleService,
- StaticCompilerCoordinator,
- VerificationCoordinator,
- AlternativeExplanationPlanner,
- ProbeTemplateService,
- LedgerSyncService,
- DraftGuardService.

Rules:

- Application coordinates workflows.
- Application calls infrastructure and adapters.
- Backend-specific semantics remain in adapters or typed domain rules.

### Scientific Domain

Core objects:

- TensorSpace,
- TensorRef,
- MechanisticUnitRef,
- ExampleBundle,
- ControlBundle,
- Hypothesis,
- HypothesisState,
- PredictionLock,
- InterventionSpec,
- PreflightReport,
- VerificationRun,
- BlockerReport,
- AlternativeExplanation,
- ProbePlan,
- MechanismCard,
- Claim,
- ScientificDebtRecord,
- DecisionRecord.

Rules:

- Domain objects are pure, serializable, ref-stable, and usable outside IPython.
- Evidence tier, workflow state, and claim status are distinct.
- No claim-bearing object may lack source refs.

### Evidence Core

Services:

- EvidenceGraph,
- ArtifactLedger,
- RunLedger,
- ClaimLedger,
- DecisionLog,
- ScientificDebtLedger,
- LineageQueryService,
- ConflictResolver,
- RebuildIndex.

Rules:

- Files are canonical.
- SQLite is rebuildable.
- Every graph edge is typed.
- Negative and contradictory evidence is first-class.

### Infrastructure

Components:

- `.mechanism/` runtime workspace,
- optional `research/` committed ledger workspace,
- SQLite index,
- JSONL event logs,
- file-backed artifacts,
- Git state capture,
- hashing,
- redaction policy,
- adapter registry.

Rules:

- Runtime caches and local large artifacts are ignored by Git.
- Human-approved ledgers and summaries are Git-visible.
- Large binary storage is delegated to DVC, git-annex, Git LFS, or external pointers.

### External Adapters

P0:

- TransformerLens,
- SAELens,
- Git,
- filesystem,
- SQLite,
- IPython.

P1 after core buildout:

- nnsight / nnterp,
- Neuronpedia read-only,
- pyvene,
- W&B / MLflow / DVC artifact interop.

P2 after repeated dogfood:

- OpenInterpretability,
- circuit-tracer,
- AutoCircuit,
- ACDC / EAP,
- CLT-Forge,
- SAEBench,
- RAVEL,
- Tracr,
- Inspect / lm-eval.

Rules:

- Adapter support requires manifest, version manifest, real integration path, conformance tests, known limitations, and failure modes.
- No adapter may silently degrade into random tensors, fixtures, or backend substitutions.

## Core Subsystems

### 1. Mechanistic Space Type System

Purpose:

Prevent mathematically invalid operations over tensors, directions, features, and patches.

Required capabilities:

- distinguish residual, SAE feature, decoder, logit, OV, QK, attention, attribution, gradient, and activation spaces;
- record hook, layer, token-position semantics, basis, normalization context, dtype, and shape;
- reject invalid projections or patches unless an explicit transform exists;
- attach TensorSpace/TensorRef identity to ActivationSet, FeatureRanking, InterventionSpec, and static compiler outputs.

Acceptance:

- invalid pre-LN/post-LN projection fails;
- incompatible SAE dictionary comparison fails;
- wrong-hook patch fails;
- valid folded transform records transform provenance.

### 2. Mechanistic Unit Registry

Purpose:

Make internal objects addressable and operation-aware.

Unit types:

- SAE feature,
- attention head,
- attention edge,
- MLP neuron,
- residual direction,
- circuit node,
- circuit edge,
- transcoder feature,
- crosscoder feature,
- prompt-token role.

Required capabilities:

- stable unit URI/ref,
- native read/write space,
- valid operations,
- invalid operations,
- external aliases,
- artifact and claim dependency queries.

### 3. Evidence Graph

Purpose:

Turn provenance into queryable scientific structure.

Nodes:

- cell,
- object,
- artifact,
- tensor space,
- unit,
- example bundle,
- control bundle,
- static result,
- patch result,
- telemetry warning,
- alternative explanation,
- run,
- claim,
- draft sentence,
- decision,
- scientific debt.

Edges:

- created_in_cell,
- derived_from,
- parent,
- supports,
- contradicts,
- depends_on,
- same_space_as,
- transforms_to,
- tested_by,
- confounded_by,
- fails_on,
- generalizes_to,
- cited_by.

Required query examples:

- show every claim depending on a feature;
- show controls contradicting a mechanism;
- show cells that produced a figure;
- show claims relying on zero ablation;
- show hypotheses killed by static algebra.

### 4. Hypothesis Lifecycle

Purpose:

Track investigation state before claims exist.

States:

```text
noticed
triaged
structurally_plausible
cheap_proxy_supported
exact_patch_supported
control_clean
generalized
claimable
structurally_impossible
proxy_false_positive
control_leaky
self_repair_confounded
off_manifold
task_artifact
dictionary_artifact
abandoned
retired
```

Rules:

- Hypothesis state is not evidence tier.
- Claim status is not workflow state.
- Transitions require evidence refs.
- Human approval is required for promotion to paper-facing claim.

### 5. Static Mechanistic Compiler

Purpose:

Kill bad hypotheses cheaply before expensive verification.

Checks:

- TensorSpace compatibility,
- decoder-to-unembedding projection,
- DLA/logit-lens alignment,
- OV/QK composition where applicable,
- dictionary geometry,
- neighbor-feature interference,
- activation density,
- off-manifold risk,
- attribution-patching proxy,
- direct-vs-routing plausibility,
- backend capability.

Output:

- `PreflightReport`,
- `StaticCheckResult`,
- `plausibility_gate: pass | weak | fail`,
- maximum allowed evidence ceiling if weak/fail.

Rules:

- Static evidence alone cannot produce causal claims.
- Failed static checks block claim-bearing verification unless downgraded to diagnostic.

### 6. Exact Causal Verification

Purpose:

Run real interventions through real adapters and write receipts.

Operations:

- resample ablation,
- mean/zero ablation as diagnostic with lower claim ceiling,
- noising,
- denoising,
- feature amplification,
- direct patch,
- path patch,
- mediation decomposition,
- self-repair scan.

Required outputs:

- `verification_results.jsonl`,
- `control_metrics.json`,
- `intervention_receipts.jsonl`,
- `telemetry.jsonl`,
- `blocker_report.json`,
- `mechanism_card.json`,
- `next_probe.yaml`,
- `diagnosis_tree.json`,
- `probe.yaml`.

Rules:

- no fake adapter path;
- no fixture-only acceptance;
- no hidden failed controls;
- PredictionLock required for claim-bearing runs.

### 7. Alternative Explanation Planner

Purpose:

Keep live confounds explicit and generate discriminating tests.

Core explanations:

- control_leaky,
- generic_high_activity_feature,
- sentiment_not_negation,
- token_position_artifact,
- decoder_neighbor_interference,
- downstream_self_repair,
- off_manifold_intervention,
- task_artifact,
- dictionary_artifact,
- insufficient_power.

Each explanation records:

- evidence_for,
- evidence_against,
- blockers,
- next_test,
- materializable_probe_template.

### 8. Probe Synthesis

Purpose:

Turn diagnoses into deterministic next experiments.

Probe examples:

- role-stratified control activation audit,
- decoder-neighbor cosine scan,
- neighbor cluster intervention,
- polarity-only bundle,
- position-resolved activation heatmap,
- downstream backup scan,
- zero/mean/resample ablation comparison,
- lower amplification-factor sweep.

Rules:

- Templates are deterministic.
- Commands are emitted only when backed by implemented CLI/adapters.
- Generated probes carry source diagnosis refs.

### 9. Example Geometry

Purpose:

Treat examples and controls as objects with measurable geometry.

Audits:

- single-token validity,
- lexical overlap,
- semantic similarity,
- token length,
- baseline margin,
- target/foil token stability,
- syntactic template balance,
- position semantics,
- role separability,
- embedding cluster overlap,
- activation cluster overlap,
- contaminated controls.

Outputs:

- `example_geometry_report.json`,
- `control_contamination_report.json`,
- rebalance recommendations,
- materialized heldout/control bundle proposals.

### 10. Claim Grammar

Purpose:

Move beyond phrase-blocking while staying deterministic.

Claim types:

- observation,
- static projection,
- causal necessity,
- causal sufficiency,
- mediation,
- generalization,
- mechanism.

Each type defines:

- required evidence refs,
- allowed verbs,
- blocked verbs,
- required caveats,
- debt visibility,
- maximum scope.

Mechanism wording remains rare and expensive.

### 11. Reference Mechanism Suite

Purpose:

Test the framework against known ground truth.

Fixtures:

- Tracr/RASP compiled circuits,
- toy transformers with planted features,
- synthetic SAE dictionaries with splitting/absorption,
- known induction-head/IOI examples,
- negative-control confounds.

Acceptance:

- known true mechanisms are classified correctly;
- tempting false positives are blocked;
- static compiler failures avoid unnecessary exact sweeps;
- exact verification receipts reproduce expected signs and magnitudes.

## Canonical File Layout Extensions

Current `.mechanism/` remains valid. Add Git-visible research state:

```text
research/
  experiments/
  logs/
    claim_ledger.md
    decision_log.md
    research_log.md
    run_ledger.csv
  paper/
  bundles/
  reference_tasks/
```

Add local runtime details:

```text
.mechanism/
  graph/
  probes/
  static_checks/
  example_geometry/
  benchmarks/
```

Rules:

- `research/` is human-approved and Git-visible.
- `.mechanism/runs/` remains local/generated unless explicitly exported.
- Rebuild commands must regenerate SQLite graph state from files.

## Authority Boundary

The workbench may recommend. It may block unsafe claim language. It may mark a run non-claim-bearing. It may not declare scientific truth by itself.

Human review is required for:

- hypothesis -> candidate claim,
- exploratory run -> evidence run,
- dirty candidate -> clean candidate,
- single-domain evidence -> generalized claim,
- claim sentence -> paper-facing statement.
