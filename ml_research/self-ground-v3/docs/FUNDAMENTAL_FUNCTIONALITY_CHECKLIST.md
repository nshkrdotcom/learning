# Fundamental Functionality Checklist

This checklist is derived from the canonical 20260625 archive:

- `0001.md` archive charter and decisions
- `0002.md` product PRD and Phase 0 dogfood spec
- `0003.md` system architecture and domain model
- `0004.md` IPython session capture spec
- `0005.md` scientific workflow, verification, next-probe, and claim grammar
- `0006.md` adapter strategy, provenance archive, and Phase 2 boundary

It intentionally excludes P1/P2 integrations and platform work that the archive says should wait until the local Phase 0 loop proves value.

## Definition Of Fundamental

A feature is fundamental only if it is necessary for the local-first IPython research loop to preserve provenance, prevent evidence confusion, or bound claims against real dogfood artifacts.

Fundamental items must be:

- covered by tests or real command transcripts,
- documented in the repo,
- QC-green before commit,
- committed and pushed manually,
- non-claim-bearing unless the evidence gate requirements are met.

## Core Product Loop

- [x] `mwb init` creates the local `.mechanism/` workspace.
  - Sources: `0001` north star, `0002` Phase 0 scope, `0003` project layout.
  - Evidence: `tests/test_phase0_project.py`, `mwb doctor`.
- [x] `mwb ipython` opens an IPython session with `ctx` injected.
  - Sources: `0001` product north star, `0002` first-run experience, `0004` load extension.
  - Evidence: `tests/test_phase2_ipython.py`.
- [x] `mwb ipython --resume <session-ref>` records `resumed_from_session_ref`.
  - Source: `0004` session resume.
  - Evidence: `tests/test_phase10_checklist_completion.py`.
- [x] `ctx` factories assign stable refs before objects reach user namespace.
  - Sources: `0003` ref system/domain objects, `0004` ctx factory requirement.
  - Evidence: `tests/test_phase1_domain.py`, `tests/test_phase4_context.py`.

## IPython Capture

- [x] Capture cell source hash, timing, execution index, status, and exceptions.
  - Source: `0004` core claim and cell records.
  - Evidence: `tests/test_phase2_ipython.py`.
- [x] Capture newly bound typed workbench objects automatically.
  - Source: `0004` automatic capture requirement.
  - Evidence: `tests/test_phase2_ipython.py`.
- [x] Capture aliases, deleted bindings, and mutation/fingerprint changes.
  - Source: `0004` namespace snapshot and rebinding rules.
  - Evidence: `tests/test_phase2_ipython.py`.
- [x] Capture bounded stdout and stderr summaries for executed cells.
  - Source: `0004` stdout, stderr, and exceptions.
  - Evidence: `tests/test_phase11_fundamental_review.py`.
- [x] Persist cell-to-object and parent-to-object lineage edges.
  - Sources: `0003` evidence core, `0004` object registration flow.
  - Evidence: `tests/test_phase11_fundamental_review.py`.
- [x] Provide explicit recording escape hatches: `ctx.record(...)`, `ctx.note(...)`, and artifact registration.
  - Source: `0004` explicit recording.
  - Evidence: `tests/test_phase10_checklist_completion.py`.

## Durable Identity And Persistence

- [x] Stable refs exist for projects, sessions, cells, objects, artifacts, runs, hypotheses, locks, reports, cards, and claims.
  - Source: `0003` ref system.
  - Evidence: `tests/test_phase1_domain.py`.
- [x] Domain objects are serializable and fingerprintable.
  - Source: `0003` scientific domain layer.
  - Evidence: `tests/test_phase1_domain.py`.
- [x] Artifacts are file-backed, hash-linked, and SQLite-indexed.
  - Source: `0003` filesystem stores.
  - Evidence: `tests/test_phase1_domain.py`.
- [x] SQLite is an operational index, not the only evidence store.
  - Source: `0003` persistence model.
  - Evidence: JSON/JSONL files under `.mechanism/`, `mwb rebuild-index`, `mwb repair-index`.
- [x] SQLite can be rebuilt from file-backed records.
  - Sources: `0003` JSONL replay, `0004` recovery behavior.
  - Evidence: `tests/test_phase10_checklist_completion.py`, `tests/test_phase11_fundamental_review.py`.

## P0 Real Backend Adapters

- [x] TransformerLens adapter has manifest, version manifest, model identity, hook TensorSpace mapping, and activation capture.
  - Sources: `0003` external adapter layer, `0006` P0 conformance.
  - Evidence: `tests/test_phase3_adapters.py`, real `mwb adapter conformance transformer-lens`.
- [x] SAELens adapter has manifest, version manifest, dictionary identity, hook compatibility, and feature refs.
  - Sources: `0003` external adapter layer, `0006` P0 conformance.
  - Evidence: `tests/test_phase3_adapters.py`, real `mwb adapter conformance saelens`.
- [x] Cross-adapter model/SAE/capture/ranking path works on real Pythia-70M artifacts.
  - Source: `0006` cross-adapter conformance.
  - Evidence: `MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_phase4_context.py -m integration`.
- [x] Missing identity or failed conformance cannot be upgraded into claim-bearing evidence.
  - Sources: `0005` prediction/evidence gates, `0006` claim-bearing gate.
  - Evidence: `tests/test_phase3_adapters.py`, `tests/test_phase5_workflow.py`.

## Scientific Workflow

- [x] Scratch mode supports model load, SAE load, bundle load, activation capture, feature ranking, notes, and artifact registration without a hypothesis.
  - Sources: `0002` Phase 0 scope, `0005` scratch mode.
  - Evidence: `tests/test_phase4_context.py`.
- [x] ExampleBundle and ControlBundle are first-class typed objects.
  - Sources: `0003` ExampleBundle/ControlBundle, `0005` hypothesis inputs.
  - Evidence: `tests/test_phase4_context.py`.
- [x] Hypotheses capture subject units, bundles, expected effects, required controls, and alternatives.
  - Source: `0005` create hypothesis.
  - Evidence: `tests/test_phase5_workflow.py`.
- [x] Prediction locks record spec hash, expected direction, control expectations, git state, and environment.
  - Source: `0005` prediction lock.
  - Evidence: `tests/test_phase5_workflow.py`.
- [x] Preflight blocks structurally bad claim-bearing runs and allows diagnostic-only output.
  - Source: `0005` static preflight.
  - Evidence: `tests/test_phase5_workflow.py`.
- [x] Verification skeleton requires a prediction lock for claim-bearing runs.
  - Sources: `0005` causal verification, `0006` claim-bearing gate.
  - Evidence: `tests/test_phase5_workflow.py`.
- [x] Sweep planning writes non-claim-bearing run artifacts and receipts.
  - Source: `0005` verification outputs.
  - Evidence: `tests/test_phase10_checklist_completion.py`.

## Failure Diagnosis And Next Probe

- [x] Blocker taxonomy includes the core blocking states needed for Phase 0 dogfood.
  - Source: `0005` blocker taxonomy.
  - Evidence: `tests/test_phase6_next_probe.py`.
- [x] `mwb next-probe` derives recommendations from blockers, metrics, tried axes, available axes, and backend capabilities.
  - Source: `0005` next-probe planning.
  - Evidence: `tests/test_phase6_next_probe.py`.
- [x] `mwb diagnose` writes a provenance-preserving diagnosis tree from blocker reports, metrics, and scientific debt.
  - Sources: `0005` next-probe planning, `0430_revised_v6` scientific debt.
  - Evidence: `tests/test_phase20_diagnosis_probes.py`.
- [x] `mwb next-probe --materialize` writes deterministic `probe.yaml` with source JSON paths.
  - Sources: `0005` materialized next-probe requirements, `mechinterp_framework/0020_gpt`.
  - Evidence: `tests/test_phase20_diagnosis_probes.py`.
- [x] `mwb run-probe` executes only implemented probe kinds through workbench workflows.
  - Source: `mechinterp_framework/0020_gpt` probe synthesis and execution boundary.
  - Evidence: `tests/test_phase20_diagnosis_probes.py`.
- [x] Artifact-incomplete input does not produce fabricated scientific commands.
  - Source: `0005` next-probe constraints.
  - Evidence: `tests/test_phase6_next_probe.py`.

## Reference Mechanism Benchmarks

- [x] `mwb benchmark framework` runs deterministic known-ground-truth reference tasks.
  - Sources: `mechinterp_framework/0020_gpt` reference tasks, `BEST_EVALS_github` eval registry patterns.
  - Evidence: `tests/test_phase21_reference_mechanisms.py`.
- [x] Planted toy mechanisms must be recovered from exact-effect scores and empirical nulls.
  - Sources: `mechinterp_framework/0020_gpt` known toy mechanisms, `mechinterp_framework/0010_claude` Tracr ground truth.
  - Evidence: `tests/test_phase21_reference_mechanisms.py`.
- [x] Tempting false-positive confounds are blocked instead of reported as mechanisms.
  - Source: `mechinterp_framework/0020_gpt` negative controls where tempting features are confounds.
  - Evidence: `tests/test_phase21_reference_mechanisms.py`.
- [x] Synthetic SAE split and absorption artifacts are detected.
  - Sources: `mechinterp_framework/0020_gpt` synthetic SAE dictionaries, `mechinterp_framework/0010_claude` SAE quality metrics.
  - Evidence: `tests/test_phase21_reference_mechanisms.py`.

## MechanismCards, Claim Grammar, And Draft Guard

- [x] `mwb card` generates structured JSON and Markdown from run artifacts.
  - Source: `0005` MechanismCard.
  - Evidence: `tests/test_phase7_cards_draftguard.py`.
- [x] Evidence tiers exist for association, projection, causal necessity, causal sufficiency, mediation, generalization, and mechanism.
  - Source: `0005` evidence tiers.
  - Evidence: `tests/test_phase10_checklist_completion.py`.
- [x] Mechanism wording is blocked unless mechanism-tier requirements are met.
  - Sources: `0005` mechanism tier, `0006` claim-bearing gate.
  - Evidence: `tests/test_phase7_cards_draftguard.py`, `tests/test_phase10_checklist_completion.py`.
- [x] Scientific debt records are emitted for unresolved blockers and missing stronger evidence.
  - Sources: `0005` alternatives/confounds, `0006` MechLedger reuse boundary.
  - Evidence: `tests/test_phase10_checklist_completion.py`.
- [x] Draft Guard supports `allowed`, `caveated`, `blocked`, `unknown_claim`, and `missing_card`.
  - Source: `0005` draft guard.
  - Evidence: `tests/test_phase10_checklist_completion.py`.

## SELF-GROUND Dogfood

- [x] Real demo loads `EleutherAI/pythia-70m-deduped` and captures activations.
  - Source: `0002` Phase 0 dogfood target.
  - Evidence: `docs/PHASE0_ACCEPTANCE_REPORT.md`.
- [x] E004 ingest validates summary JSON, comparison CSVs, forensics CSVs, and backend metadata.
  - Sources: `0002` dogfood pain, `0005` next-probe from real run artifacts.
  - Evidence: `tests/test_phase8_self_ground_ingest.py`, `tests/test_phase10_checklist_completion.py`.
- [x] E004 output remains `insufficient_evidence`, `control_leaky`, and non-claim-bearing.
  - Sources: `0005` claim grammar, `0006` real evidence versus fixtures.
  - Evidence: `docs/PHASE0_ACCEPTANCE_REPORT.md`, `docs/PHASE10_COMPLETION_REPORT.md`.

## Fundamental Non-Goals

These are intentionally deferred because the canonical archive excludes them from Phase 0:

- [ ] Dashboard-first UI.
- [ ] Cloud sync or multi-user collaboration.
- [ ] Distributed workers or NSHKR service split.
- [ ] nnsight/nnterp parity.
- [ ] pyvene hard dependency.
- [ ] Neuronpedia write integration.
- [ ] OpenInterpretability, circuit-tracer, CLT-Forge, SAEBench, RAVEL, Tracr, Inspect, or lm-eval interop.
- [ ] Filesystem watching for arbitrary artifacts.
- [ ] Raw tensor/model-weight capture by default.

## Fundamental QC Gate

Run this gate before any commit that changes fundamental behavior:

```bash
uv sync
uv run ruff check .
uv run pytest
MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_phase4_context.py -m integration
uv run mwb adapter conformance transformer-lens --model EleutherAI/pythia-70m-deduped --device cpu
uv run mwb adapter conformance saelens --model EleutherAI/pythia-70m-deduped --hook blocks.2.hook_resid_post --device cpu
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
```

Every gate failure is either fixed with RGR or recorded as explicit non-fundamental debt before commit.
