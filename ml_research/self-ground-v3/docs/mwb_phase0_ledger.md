# Mechanistic Workbench Phase 0 Ledger

## Phase -1: Repo Preflight And Baseline

Status: complete
Commit: `f8f4e36`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/README.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/00_context_and_decisions.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0001.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/MECHINTERP_CLAUDE.md`

Preflight:
- Requested target `/home/home/p/g/n/learning/ml_research/self-ground-v3` was absent.
- The implementation repo was created at the requested path.
- A local bare Git remote was created at `../self-ground-v3-origin.git` so phase pushes are real Git pushes.
- Package layout decision: new `src/mwb` package following the implementation docset.

Commands planned:

```bash
uv sync
uv run ruff check .
uv run pytest
git status --short --branch
```

Observed result:
- `uv sync`: passed; created `.venv` with Python 3.12.2 and installed `mechanistic-workbench==0.1.0`.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `1 passed`.
- `git status --short --branch`: clean except the initial files staged for the Phase -1 commit.

Known residual risk:
- This repo was created during implementation because the requested path did not exist locally.

## Phase 0: Package Skeleton, CLI, And Project Init

Status: complete
Commit: `345ad93`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0002.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0003.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/02_architecture_contracts.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/06_qc_commit_push_protocol.md`

Implemented:
- `mwb` package and Typer CLI entrypoint.
- `uv run mwb --help`.
- `uv run mwb init`.
- `.mechanism/` layout creation.
- Project discovery and `project.toml` validation.
- Append-only `events.jsonl`.
- Git state snapshot without committing.
- SQLite schema initialization.
- Read-only `uv run mwb doctor`.
- Tests for init idempotency, event append, SQLite schema, Git dirty state, and CLI init/doctor.

Commands run:

```bash
uv run mwb --help
uv run mwb init --name self-ground
uv run mwb doctor
uv run ruff check .
uv run pytest
```

Observed result:
- `uv run mwb --help`: passed.
- `uv run mwb init --name self-ground`: passed.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `6 passed`.

Known residual risk:
- `.mechanism/` is runtime project state and is intentionally ignored by Git in this implementation repo.

## Phase 1: Domain Refs, Typed Objects, And Artifact Registry

Status: complete
Commit: `a9a66d7`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0003.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0004.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/02_architecture_contracts.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/03_ipython_capture_impl.md`

Implemented:
- Deterministic `stable_ref`.
- Workbench object protocol and pydantic-backed typed objects.
- Domain models for sessions, cells, model identity, dictionary identity, TensorSpace, units, bundles, hypotheses, locks, interventions, reports, plans, cards, and claims.
- Canonical JSON serialization and type-based deserialization.
- Artifact registry with sha256, byte count, MIME type, role, parent refs, creator ref, and SQLite indexing.
- Generic SQLite payload insert/fetch helpers.

Commands run:

```bash
uv run mwb init --name self-ground
uv run mwb doctor
uv run ruff check .
uv run pytest
```

Observed result:
- `uv run mwb init --name self-ground`: passed.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `12 passed`.

Known residual risk:
- Phase 1 defines domain identity and artifact indexing but does not yet capture IPython cells; that starts in Phase 2.

## Phase 2: Session Manager And IPython Extension Crux

Status: complete
Commit: `b8c547b`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0004.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/MECHINTERP_CLAUDE.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/03_ipython_capture_impl.md`

Implemented:
- `uv run mwb ipython`.
- Repeatable `--execute` mode that runs real IPython cells through installed Workbench hooks.
- `%load_ext mwb.ipython` extension entrypoint.
- `ctx`, `mwb`, `display_card`, `display_run`, `display_features`, and `display_graph` injection.
- Session creation and close with `.mechanism/sessions/sess_*/session.json`.
- `cells.jsonl`, `namespace_objects.jsonl`, and exception capture.
- Namespace diffing for object registration, alias binding, alias deletion, and mutation.
- SQLite indexing for sessions, cells, objects, and object versions.
- `mwb inspect session latest`.

Commands run:

```bash
uv run mwb init --name self-ground
uv run mwb ipython --execute "obj = ctx.objects.create('Note', metadata={'source': 'phase2-qc'})"
uv run mwb inspect session latest
uv run mwb doctor
uv run ruff check .
uv run pytest
```

Observed result:
- `uv run mwb ipython --execute ...`: passed and wrote a closed IPython session.
- `uv run mwb inspect session latest`: passed and reported `surface: ipython`.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `17 passed`.

Known residual risk:
- Phase 2 validates generic typed-object capture. Real TransformerLens/SAELens adapter objects start in Phase 3.

## Phase 3: P0 Adapter Contracts And Conformance

Status: complete
Commit: `5bc4325`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0006.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0003.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/05_adapters_and_dogfood.md`

Implemented:
- P0 dependencies promoted to main dependencies: TransformerLens, SAELens, and Torch.
- `AdapterCapabilityManifest`.
- `BackendVersionManifest`.
- `AdapterConformanceResult`.
- Claim-bearing evidence gate over adapter conformance and required refs.
- `TransformerLensAdapter` with real model load, identity extraction, hook TensorSpace mapping, forward pass, and activation capture.
- `SAELensAdapter` with real SAE load, dictionary identity extraction, hook compatibility, and feature ref round-trip.
- `mwb adapter conformance transformer-lens`.
- `mwb adapter conformance saelens`.
- Tests for manifests, version manifests, TensorSpace mapping, evidence-gate blocking, and CLI dry-run output.

Commands run:

```bash
uv sync
uv run mwb init --name self-ground
uv run mwb adapter conformance transformer-lens --model EleutherAI/pythia-70m-deduped --device cpu
uv run mwb adapter conformance saelens --model EleutherAI/pythia-70m-deduped --hook blocks.2.hook_resid_post --device cpu
uv run mwb doctor
uv run ruff check .
uv run pytest
```

Observed result:
- `uv sync`: passed and installed `transformer-lens==3.3.0`, `sae-lens==6.44.4`, and `torch==2.12.1`.
- TransformerLens conformance: passed; loaded `EleutherAI/pythia-70m-deduped`, captured `blocks.0.hook_resid_post`, activation shape `[1, 5, 512]`.
- SAELens conformance: passed; loaded `pythia-70m-deduped-res-sm` / `blocks.2.hook_resid_post`, validated hook compatibility, created feature ref.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `23 passed`.

Known residual risk:
- Adapter conformance writes runtime reports under `.mechanism/adapters/`; `.mechanism/` remains ignored and is regenerated by commands.

## Phase 4: Bundles, Context APIs, And Scratch Workflow

Status: complete
Commit: `2c99823`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0002.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0003.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0005.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/04_scientific_workflow_impl.md`

Implemented:
- `ctx.models.load_tl(...)` using the real TransformerLens adapter.
- `ctx.saes.load(...)` using the real SAELens adapter.
- `ctx.domains.negation.load("phase3_calibrated")`.
- Built-in `negation_phase3_calibrated.yaml` bundle resource.
- `DomainBundle`, `ActivationSet`, and `FeatureRanking` typed objects.
- `ctx.capture(model, bundle).at(hook)` using real TransformerLens activation cache.
- `ctx.features.rank(sae, acts, contrast=...)` using real SAELens encoding and top-k feature scoring.
- `ctx.artifact.register(...)`.
- `mwb demo negation --dry-run`.
- IPython capture of loaded domain bundles.

Commands run:

```bash
uv run mwb demo negation --dry-run --model EleutherAI/pythia-70m-deduped
MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_phase4_context.py -m integration
uv run mwb doctor
uv run ruff check .
uv run pytest
```

Observed result:
- Demo dry-run: passed and validated the built-in negation bundle with four control families.
- Real integration test: passed; loaded TransformerLens model, loaded SAELens SAE, captured activations, encoded features, and produced top-k feature rankings.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `26 passed, 1 skipped`.

Known residual risk:
- Default pytest skips the real adapter workflow unless `MWB_RUN_REAL_ADAPTER_TESTS=1`; the command above was run manually for this phase.

## Phase 5: Hypotheses, Preflight, Prediction Locks, And Verification Skeleton

Status: complete
Commit: `c37350b`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0005.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0003.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/04_scientific_workflow_impl.md`

Implemented:
- `ctx.hypotheses.create(...)`.
- `ctx.predictions.lock(...)` with hypothesis fingerprint, Git state, environment, expected direction, and expected controls.
- `mwb preflight <hypothesis-json>`.
- TensorSpace compatibility, model/SAE hook compatibility, control bundle, and decoder-unembedding projection preflight checks.
- Preflight pass/warn/fail posture.
- `mwb verify <hypothesis-json>`.
- Prediction lock gate for claim-bearing verification.
- Diagnostic-only dry-run verification output.
- Explicit fixture-only hypothesis JSON for reproducible QC.

Commands run:

```bash
uv run mwb init --name self-ground
uv run mwb preflight docs/fixtures/hypothesis_phase5.json
uv run mwb verify docs/fixtures/hypothesis_phase5.json --diagnostic-only --dry-run
uv run mwb doctor
uv run ruff check .
uv run pytest
```

Observed result:
- Preflight fixture: passed with `decoder_unembed_projection` score `0.04`.
- Diagnostic verify fixture: passed with `evidence_posture=diagnostic_only`.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `31 passed, 1 skipped`.

Known residual risk:
- `docs/fixtures/hypothesis_phase5.json` is explicitly `fixture_only` and `claim_bearing=false`; it must not be used as claim-bearing evidence.

## Phase 6: Causal Verification, Sweep, Blockers, And Next-Probe

Status: complete
Commit: `8666bde`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0005.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/04_scientific_workflow_impl.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/05_adapters_and_dogfood.md`

Implemented:
- `mwb sweep`.
- Repeatable `--axis name=value[,value...]` parsing with cross-product matrix size.
- Canonical blocker diagnosis for artifact incompleteness, control leakage, family gap failure, and specificity gap failure.
- `mwb next-probe`.
- Deterministic next-probe plans from run manifests and control metrics.
- `next_probe.yaml` and `next_probe.md` emission.
- No recommendation command when required fields are missing.
- Control-leaky fixture run with generated next-probe outputs.

Commands run:

```bash
uv run mwb init --name self-ground
uv run mwb sweep docs/fixtures/hypothesis_phase5.json --axis layer=0,1 --axis patch_mode=direct --dry-run
uv run mwb next-probe docs/fixtures/runs/control_leaky
uv run mwb doctor
uv run ruff check .
uv run pytest
```

Observed result:
- Sweep dry-run: passed with cross-product matrix semantics.
- Next-probe fixture: passed and wrote `next_probe.yaml` plus `next_probe.md`.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `36 passed, 1 skipped`.

Known residual risk:
- Phase 6 plans dry-run sweeps and next probes; full causal execution remains bounded by adapter-backed verification and claim gates.

## Phase 7: MechanismCards, Claim Grammar, Draft Guard, And Scientific Debt

Status: complete
Commit: `7af7f20`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0005.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0006.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/04_scientific_workflow_impl.md`

Implemented:
- `mwb card <run-path>`.
- MechanismCard JSON and Markdown generation from run artifacts.
- Evidence tier calculation.
- Allowed and blocked language from evidence tier and blockers.
- Claim refs linked to generated cards.
- `.mechanism/cards` and `.mechanism/claims` local card/claim registry writes.
- `mwb draft-check <draft-path>`.
- Draft Guard `[CLAIM:...]` scanning over Markdown lines.
- Blocked, allowed, and missing-card draft statuses.
- Fixture allowed draft plus generated fixture MechanismCard artifacts.

Commands run:

```bash
uv run mwb init --name self-ground
uv run mwb card docs/fixtures/runs/control_leaky
uv run mwb draft-check docs/fixture_draft.md
uv run mwb doctor
uv run ruff check .
uv run pytest
```

Observed result:
- Card fixture: passed and wrote `mechanism_card.json` plus `mechanism_card.md`.
- Draft fixture: passed because it uses association-tier language.
- Blocked mechanism wording: covered by tests and exits non-zero.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `39 passed, 1 skipped`.

Known residual risk:
- Draft Guard is deterministic string matching over claim-tagged lines; it intentionally does not attempt full natural-language entailment.

## Phase 8: SELF-GROUND Dogfood Ingest And Acceptance

Status: complete
Commit: `1947d55`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/05_adapters_and_dogfood.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/06_qc_commit_push_protocol.md`
- `/home/home/p/g/n/learning/ml_research/self-ground/runs/e004_specificity_rescue_matrix/matrix_run_summary.json`
- `/home/home/p/g/n/learning/ml_research/self-ground/runs/e004_specificity_rescue_matrix/comparison/comparison.json`
- `/home/home/p/g/n/learning/ml_research/self-ground/runs/e004_specificity_rescue_matrix/comparison/matrix_summary.json`
- `/home/home/p/g/n/learning/ml_research/self-ground/runs/e004_specificity_rescue_matrix/comparison/claim_adjudication.md`

Implemented:
- `mwb ingest self-ground <run-dir>`.
- Validation of the SELF-GROUND E004 artifact shape and required summary columns.
- Translation of E004 best-run metrics into Workbench `control_metrics.json`.
- Run manifest generation under `.mechanism/runs/run_self_ground_e004_specificity_rescue_matrix`.
- Blocker report, next-probe plan, and MechanismCard generation during ingest.
- `latest` resolution for `mwb card latest` and `mwb next-probe latest`.
- `next_probe.json` emission alongside YAML and Markdown.
- Draft Guard regression fix: blocked-language matching ignores `[CLAIM:...]` marker text.
- Real E004 draft fixture using allowed association-tier language.
- Phase 0 acceptance report.

Commands run:

```bash
uv run mwb demo negation --model EleutherAI/pythia-70m-deduped --device cpu
uv run mwb ingest self-ground /home/home/p/g/n/learning/ml_research/self-ground/runs/e004_specificity_rescue_matrix
uv run mwb card latest
uv run mwb next-probe latest
uv run mwb draft-check docs/fixture_draft.md
uv run mwb doctor
uv run ruff check .
uv run pytest
```

Observed result:
- Real demo: passed; loaded `EleutherAI/pythia-70m-deduped` and captured activation shape `[2, 6, 512]`.
- SELF-GROUND ingest: passed with `run_ref=run_self_ground_e004_specificity_rescue_matrix`, `status=insufficient_evidence`, and `primary_blocker=control_leaky`.
- `mwb card latest`: passed and blocked mechanism/specificity language for the real E004 claim.
- `mwb next-probe latest`: passed and recommended the smallest untried adjacent layer axis.
- `mwb draft-check docs/fixture_draft.md`: passed with `status=allowed`.
- `mwb doctor`: passed with `status: ok`.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `42 passed, 1 skipped`.

Known residual risk:
- The ingested E004 result remains `insufficient_evidence`; the workbench preserves that status and does not upgrade it into claim-bearing mechanism evidence.

## Phase 9: Hardening And Release Candidate

Status: complete
Commit: `b265464`
Pushed: yes

Required reading completed:
- `docs/PHASE0_ACCEPTANCE_REPORT.md`
- `docs/USAGE.md`
- `docs/PHASE9_HARDENING_REPORT.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/06_qc_commit_push_protocol.md`

Implemented:
- Release-candidate hardening report.
- Usage guide with setup, scratch, backend, dogfood, and QC commands.
- README command surface update.
- Full dependency, lint, unit, integration, adapter, doctor, scan, commit, and push gate.

Commands run:

```bash
uv sync
uv run ruff check .
uv run pytest
uv run mwb init --name self-ground
uv run mwb doctor
MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_phase4_context.py -m integration
uv run mwb adapter conformance transformer-lens --model EleutherAI/pythia-70m-deduped --device cpu
uv run mwb adapter conformance saelens --model EleutherAI/pythia-70m-deduped --hook blocks.2.hook_resid_post --device cpu
rg -n --glob '!docs/PHASE9_HARDENING_REPORT.md' --glob '!docs/PHASE10_COMPLETION_REPORT.md' --glob '!docs/mwb_phase0_ledger.md' "fake|dummy|mock|simulated|placeholder|smoke" src tests docs README.md pyproject.toml
rg -n --glob '!docs/PHASE9_HARDENING_REPORT.md' --glob '!docs/PHASE10_COMPLETION_REPORT.md' --glob '!docs/mwb_phase0_ledger.md' "implements|mechanism for|proves|isolated.*circuit|strong_candidate_evidence" src tests docs README.md pyproject.toml
```

Observed result:
- `uv sync`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `42 passed, 1 skipped`.
- `uv run mwb init --name self-ground`: passed.
- `uv run mwb doctor`: passed with `status: ok`.
- Real adapter integration test: passed, `1 passed, 3 deselected`.
- TransformerLens conformance: passed with real model load and activation capture.
- SAELens conformance: passed with real SAE load and feature ref round-trip.
- Non-real-work scan: no matches.
- Overclaiming scan: expected matches only in blocked-language declarations, tests, and fixture cards that assert blocking behavior.

Known residual risk:
- Upstream TransformerLens/SAELens deprecation warnings were observed in integration tests. They do not currently break the Phase 0 API, but should be watched before upgrading either package family.

## Phase 10: Literal Checklist Completion

Status: complete
Commit: `f2de310`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260626/mi_impl/01_phased_checklist.md`
- `docs/PHASE0_ACCEPTANCE_REPORT.md`
- `docs/PHASE9_HARDENING_REPORT.md`
- `docs/USAGE.md`

Implemented:
- `mwb ipython --resume <session-ref>` for scripted and interactive launches.
- IPython capture tests for `ctx.note(...)` and `ctx.record(...)`.
- Full dry-run sweep artifact emission:
  - `sweep_config.json`
  - `run_manifest.json`
  - `verification_results.jsonl`
  - `intervention_receipts.jsonl`
  - `control_metrics.json`
  - `blocker_report.json`
- Complete evidence-tier language table for association, projection, causal necessity, causal sufficiency, mediation, generalization, and mechanism.
- `scientific_debt.json` records generated from MechanismCards.
- Draft Guard statuses for `allowed`, `caveated`, `blocked`, `unknown_claim`, and `missing_card`.
- SELF-GROUND E004 comparison CSV and forensics CSV validation.
- `mwb rebuild-index` for file-backed SQLite rebuild/read compatibility checks.
- Phase 10 completion report and usage/acceptance doc updates.

Commands run:

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

Observed result:
- Resume session: passed and wrote `resumed_from_session_ref`.
- Sweep dry-run: passed and wrote the full non-claim-bearing artifact set.
- SELF-GROUND ingest: passed with full comparison and forensics validation.
- Card/latest, next-probe/latest, and draft guard: passed.
- SQLite rebuild: passed with `status: ok`.
- `mwb doctor`: passed with `status: ok`.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `50 passed, 1 skipped`.
- Real adapter integration test: passed, `1 passed, 3 deselected`.
- TransformerLens conformance: passed.
- SAELens conformance: passed.
- Non-real-work scan: no matches.
- Overclaiming scan: expected matches only in blocked/allowed language tables, tests, and fixture cards; E004 remains association-tier and mechanism-blocked.

Known residual risk:
- The workbench still does not claim E004 proves a mechanism. The completed implementation enforces that boundary through blocker reports, MechanismCards, draft guard, and scientific debt records.

## Phase 11: Canonical Archive Fundamental Review

Status: complete
Commit: `be3f691`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0001.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0002.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0003.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0004.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0005.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0006.md`

Implemented:
- Source-traced `docs/FUNDAMENTAL_FUNCTIONALITY_CHECKLIST.md`.
- Bounded stdout/stderr capture for IPython cells.
- Canonical `mwb repair-index` alias for SQLite recovery.
- SQLite lineage edges for cell-to-object and parent-to-object provenance.
- Rebuild-index lineage reconstruction from file-backed namespace logs.
- RGR coverage for stdout/stderr capture, repair-index, and lineage edges.
- README and usage doc links to the fundamental checklist and repair alias.

Commands run:

```bash
uv run pytest tests/test_phase11_fundamental_review.py
uv run ruff check .
uv run pytest
```

Observed result:
- Phase 11 RGR tests first failed on missing stream refs, missing `repair-index`, and missing lineage edges.
- Stream capture implementation passed.
- `repair-index` alias passed.
- Lineage edge writes passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed after implementation.

Known residual risk:
- The checklist intentionally excludes P1/P2 adapters, platform integration, dashboard-first UI, cloud sync, and raw tensor capture by default because the canonical archive marks those as later work or non-goals.

## Phase 12: World-Class Source Mining Docset

Status: complete
Commit: `b3c740c`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260624/ml_research/mechinterp_tracker/*.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/**/*.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0001.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0002.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0003.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0004.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0005.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0006.md`

Implemented:
- `docs/world_class_buildout/README.md` with source scope, scale, reading order, and scope rule.
- `docs/world_class_buildout/00_source_mining_findings.md` with full file inventory, converged decisions, anti-patterns, source tensions, and immediate buildout implications.
- `docs/world_class_buildout/01_target_architecture.md` with the target local-first research OS architecture, core subsystems, file layout, and authority boundary.
- `docs/world_class_buildout/02_implementation_plan.md` with buildout streams, acceptance sources, rejected acceptance patterns, docs rules, QC gate, and commit/push discipline.
- `docs/world_class_buildout/03_phased_tdd_checklist.md` with a source-resolved phased checklist for evidence graph, ledgers, hypothesis lifecycle, space typing, static compiler, exact verification, example geometry, diagnosis/probes, reference mechanisms, claim grammar, policy profiles, adapters, and release hardening.
- `docs/world_class_buildout/04_qc_commit_push_protocol.md` with the mandatory TDD/RGR, QC-green, commit, and push protocol for each future phase.
- README and usage-guide links to the buildout docset.

Commands run:

```bash
find docs/world_class_buildout -maxdepth 1 -type f -print | sort
wc -l docs/world_class_buildout/*.md
uv sync
uv run ruff check .
uv run pytest
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
git status --short --branch
```

Observed result:
- Source inventory identified 31 tracker markdown files and 22 `mi_docs` markdown files.
- Final docset totals 1,927 lines across 6 files.
- `uv sync`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `53 passed, 1 skipped`.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite`: passed with `status: ok`.

Known residual risk:
- This phase is documentation and planning only. It intentionally does not implement the future buildout capabilities described in the checklist.

## Phase 13: Evidence Graph Query Core

Status: complete
Commit: `e20d307`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0003.md` Evidence Core, persistence model, evidence graph, and SQLite schema sections.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0004.md` object registration, artifact capture, lineage, and recovery sections.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/mechinterp_framework/0020_gpt.md` "Evidence as a typed causal graph".
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260624/ml_research/mechinterp_tracker/0430_revised_v6.md` SQLite-never-canonical and index rebuild sections.

Implemented:
- `EvidenceEdge` domain object with validated typed relations: `supports`, `contradicts`, `depends_on`, `derived_from`, `tested_by`, `confounded_by`, `fails_on`, `generalizes_to`, and `cited_by`.
- `EvidenceGraphService` that rebuilds `.mechanism/graph/evidence_edges.jsonl` and `.mechanism/graph/graph_summary.json` from file-backed workbench records.
- SQLite `evidence_edges` operational index.
- `mwb graph rebuild`.
- `mwb graph query claims-depending-on <ref>`.
- `mwb graph query controls-contradicting <run-ref>`.
- `mwb graph query cells-producing <artifact-ref>`.
- `mwb graph query debt-blocking <claim-ref>`.
- `mwb rebuild-index` / `mwb repair-index` restoration of `evidence_edges` from graph JSONL.
- `docs/EVIDENCE_GRAPH.md`, README, usage guide, and buildout checklist updates.

Commands run:

```bash
uv run pytest tests/test_phase12_evidence_graph.py
uv sync
uv run ruff check .
uv run pytest
uv run mwb graph rebuild
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
git status --short --branch
```

Observed result:
- Phase 13 RGR tests first failed on missing `EvidenceEdge`, missing `EvidenceGraphService`, and missing `mwb graph`.
- Focused Phase 13 test suite passed after implementation, `3 passed`.
- `uv sync`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `56 passed, 1 skipped`.
- `uv run mwb graph rebuild`: passed and rebuilt 23 evidence edges across 16 nodes.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite`: passed with `status: ok` and restored 23 `evidence_edges`.

Known residual risk:
- The graph records declared provenance and evidence relationships. It does not upgrade evidence tiers or make a claim scientifically true by itself.

## Phase 14: Git-Native Research Ledgers

Status: complete
Commit: `185b6e1`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260624/ml_research/mechinterp_tracker/0430_revised_v6.md` claim ledger, run ledger, decision log, research log, and SQLite-never-canonical sections.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260624/ml_research/mechinterp_tracker/0431_selfground_refactor.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260624/ml_research/mechinterp_tracker/0432_selfground_refactor.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260624/ml_research/mechinterp_tracker/0300_research_landscape_for_git_native_research_integrity_systems.md`

Implemented:
- Git-visible `research/` scaffold and committed ledger templates.
- Claim ledger parser for H3 claim headings plus required YAML blocks.
- Run ledger CSV schema validation with canonical column order.
- Decision log parser for H2 decision headings plus required YAML blocks.
- Research log parser for date entries plus required YAML blocks.
- `mwb ledger validate` for parser validation and SQLite indexing.
- `mwb ledger propose-run <run-ref>` with human-reviewable `run_ledger_row.csv` output.
- `mwb ledger propose-claim <card-ref>` with human-reviewable Markdown and JSON proposal files.
- `mwb rebuild-index` / `mwb repair-index` restoration of ledger rows from Git-visible files.
- `docs/LEDGERS.md`, README, usage guide, and buildout checklist updates.

Commands run:

```bash
uv run pytest tests/test_phase14_research_ledgers.py
uv sync
uv run ruff check .
uv run pytest
uv run mwb ledger validate
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
git status --short --branch
```

Observed result:
- Phase 14 RGR tests first failed on missing `mwb.ledgers`, missing `mwb ledger`, and missing Git-visible research scaffold.
- Focused Phase 14 test suite passed after implementation, `4 passed`.
- `uv sync`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `60 passed, 1 skipped`.
- `uv run mwb ledger validate`: passed with `status: ok`.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite`: passed with `status: ok`.

Known residual risk:
- Ledger validation proves structure and local traceability. It does not silently append proposals or promote claim status without human review.

## Phase 15: Hypothesis Lifecycle And Alternative Explanations

Status: complete
Commit: `0dca227`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0005.md` research modes, hypothesis creation, next-probe, MechanismCard, and acceptance criteria sections.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/mechinterp_framework/0020_gpt.md` hypothesis state machine and alternative-explanation engine sections.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260624/ml_research/mechinterp_tracker/0020_critique_claude.md` critique of collapsed workflow state and epistemic status.

Implemented:
- `HypothesisState` domain object with workflow state, evidence tier, and claim status as separate fields.
- `HypothesisTransitionReceipt` domain object.
- `AlternativeExplanation` domain object.
- `HypothesisLifecycleService` for valid transitions, transition receipt persistence, and alternative-explanation generation from blocker reports.
- `mwb hypothesis transition <hypothesis-ref> --to-state <state>`.
- `mwb hypothesis explain <run-ref>`.
- Explicit `--approved-by` and `--decision-ref` requirements for `claimable` promotion.
- SQLite indexing and rebuild/repair recovery for `hypothesis_states`, `hypothesis_transitions`, and `alternative_explanations`.
- `docs/HYPOTHESIS_LIFECYCLE.md`, README, usage guide, and buildout checklist updates.

Commands run:

```bash
uv run pytest tests/test_phase15_hypothesis_lifecycle.py
uv sync
uv run ruff check .
uv run pytest
uv run mwb hypothesis transition hyp_qc_phase15 --to-state triaged --evidence-tier association
uv run mwb hypothesis explain latest
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
git status --short --branch
```

Observed result:
- Phase 15 RGR tests first failed on missing `HypothesisState`, missing `mwb hypothesis`, and missing SQLite lifecycle recovery.
- Focused Phase 15 test suite passed after implementation, `4 passed`.
- `uv sync`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `64 passed, 1 skipped`.
- `uv run mwb hypothesis transition hyp_qc_phase15 --to-state triaged --evidence-tier association`: passed and wrote a transition receipt.
- `uv run mwb hypothesis explain latest`: passed and generated a live `control_leaky` alternative from the latest run's blocker metrics.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite`: passed with `status: ok` and restored `hypothesis_states`, `hypothesis_transitions`, and `alternative_explanations`.

Known residual risk:
- Lifecycle state and alternative explanations are deterministic workflow records. They do not create or accept paper claims without the separate ledger/proposal review path.

## Phase 16: Mechanistic Space Type System

Status: complete
Commit: `b89d147`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0003.md` TensorSpace, MechanisticUnitRef, and invalid mechanistic operations sections.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/mechinterp_framework/0020_gpt.md` typed tensor spaces, mechanistic unit registry, compatibility rules, and valid/invalid operations.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/mechinterp_framework/0010_claude.md` static compiler and space type system prerequisite sections.

Implemented:
- Expanded `TensorSpace` with backend, layer, stream kind, basis, normalization context, token-position semantics, and device fields.
- Added `TensorRef`, `SpaceTransform`, `SpaceCompatibilityReport`, and expanded `MechanisticUnitRef`.
- Added `MechanisticUnitRegistry` to enforce valid and invalid unit operations.
- Added `SpaceTypeService` to fail closed on incompatible SAE dictionaries, missing source/target/unit spaces, pre-LN/post-LN mismatches without transforms, transforms without provenance, wrong-hook patching, and invalid unit operations.
- Added `mwb space check <json>` with JSON output, nonzero exit on failure, `.mechanism/space_checks/latest_space_check.json` persistence, and SQLite indexing.
- Added SQLite schema/index rebuild support for `tensor_refs`, `space_transforms`, and `space_checks`.
- Updated `mwb doctor` to refresh rebuildable SQLite schema drift before checking table presence, so older workspaces upgrade cleanly without changing canonical evidence files.
- Added `docs/SPACE_TYPES.md`, a valid fixture, README, usage-guide, and buildout checklist updates.

Commands run:

```bash
uv run pytest tests/test_phase16_space_types.py
uv run ruff check src/mwb/space_types.py tests/test_phase16_space_types.py
uv sync
uv run ruff check .
uv run pytest
uv run mwb space check docs/fixtures/space_check_valid.json
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
git status --short --branch
```

Observed result:
- Phase 16 RGR tests first failed on missing `mwb.space_types`, missing `SpaceTransform`, missing expanded `TensorSpace` fields, and missing `mwb space`.
- Added provenance and missing-space tests; the missing-space test first failed because unit-space validation was skipped when the target space was unknown, then passed after fail-closed validation moved outside the target-space branch.
- Added a schema-drift regression test after QC surfaced that existing SQLite indexes from older phases lacked newly introduced tables until a writer refreshed schema.
- Focused Phase 16 test suite passed, `8 passed`.
- Focused ruff check passed.
- `uv sync`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `72 passed, 1 skipped`.
- `uv run mwb space check docs/fixtures/space_check_valid.json`: passed and wrote a `SpaceCompatibilityReport` with `status: pass`.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite`: passed with `status: ok` and restored `space_checks: 1`.

Known residual risk:
- Space checks are structural compatibility gates. A passing report does not establish causal evidence, validate controls, or promote a claim.

## Phase 17: Static Mechanistic Compiler

Status: complete
Commit: `90253a9`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0005.md` static preflight, decoder-unembedding projection math, preflight statuses, blocker taxonomy, and required command sections.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/mechinterp_framework/0010_claude.md` compiler, direct-effect attribution, dictionary geometry diagnostics, and weakest-link plausibility gate sections.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/mechinterp_framework/0020_gpt.md` typed algebra compiler, static checks, live alternatives, and static compile usage sections.

Implemented:
- `StaticCheckResult` and `StaticCompilationReport` domain objects.
- `StaticCompiler` with a check registry for decoder-unembedding projection, dictionary neighbor interference, and activation density.
- Real L2 cosine projection over provided decoder vectors and target-vs-foil unembedding vectors.
- Dictionary neighbor geometry scan with fail/warn thresholds.
- Activation density warning check using symmetric target/control ratios.
- Weakest-link plausibility gate aggregation: `PASS`, `WEAK`, or `FAIL`.
- Claim-bearing verification blocking when static compiler input is missing or the static gate fails.
- `mwb compile hypothesis <json>` with report persistence under `.mechanism/static_compiler/` and SQLite indexing.
- SQLite repair recovery for static compiler reports and per-check rows.
- Optional real-adapter integration test that builds compiler input from TransformerLens unembedding weights and SAELens decoder vectors.
- `docs/STATIC_COMPILER.md`, README, usage-guide, fixture, and buildout checklist updates.

Commands run:

```bash
uv run pytest tests/test_phase17_static_compiler.py
uv run pytest tests/test_phase17_static_compiler.py tests/test_static_compiler_integration.py
uv run ruff check src/mwb/static_compiler.py src/mwb/workflows/preflight.py src/mwb/workflows/verify.py src/mwb/cli.py tests/test_phase17_static_compiler.py
uv run pytest tests/test_phase5_workflow.py tests/test_phase17_static_compiler.py
uv run mwb compile hypothesis docs/fixtures/hypothesis_phase5.json
uv sync
uv run ruff check .
uv run pytest
MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_static_compiler_integration.py -m integration
uv run mwb compile hypothesis docs/fixtures/hypothesis_phase5.json
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
git status --short --branch
```

Observed result:
- Phase 17 RGR tests first failed on missing `mwb.static_compiler`, missing `mwb compile`, and claim-bearing verification ignoring failed static gates.
- Focused compiler tests passed, `6 passed`.
- Optional integration file is skipped by default unless `MWB_RUN_REAL_ADAPTER_TESTS=1`.
- Existing preflight/verify workflow tests still pass with static compiler integration.
- `uv sync`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `78 passed, 2 skipped`.
- `MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_static_compiler_integration.py -m integration`: passed, `1 passed`; warnings were upstream deprecations from TransformerLens/SAELens imports.
- `uv run mwb compile hypothesis docs/fixtures/hypothesis_phase5.json`: passed and wrote a `StaticCompilationReport` with `plausibility_gate: PASS`.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite`: passed with `status: ok` and restored `static_compiler_reports: 1` and `static_check_results: 3`.

Known residual risk:
- Static compiler reports are structural plausibility evidence only. They cannot produce causal evidence or claim promotion without subsequent prediction-locked causal verification.

## Phase 18: Exact Causal Verification Operations

Status: complete pending commit and push
Commit: `430ac2f`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0005.md` causal verification operations, required outputs, verification metrics, blocker taxonomy, and required verification tests.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/mechinterp_framework/0020_gpt.md` causal engine, research taste policies, noising/denoising, resample ablation, and telemetry guidance.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260624/ml_research/mechinterp_tracker/0010_mechinterp_tracker_gpt.md` activation patching variants, supported operations, telemetry, and ablation/amplification artifact requirements.

Implemented:
- `InterventionReceipt` and `TelemetryReport` domain objects.
- `CausalVerificationService` for exact verification runs, metrics, receipts, telemetry, policy downgrades, and artifact persistence.
- Resample-ablation receipts with target/control metric calculation.
- Distinct noising and denoising receipts with causal direction.
- Feature amplification receipts with coefficients.
- KL drift and activation norm drift telemetry with `off_manifold_intervention` blockers.
- Zero-ablation claim ceiling policy that downgrades claim-bearing attempts to `diagnostic_only` unless policy changes.
- Real TransformerLens/SAELens resample-ablation path: clean/corrupt activation cache, SAE encode/decode delta, hook patch, and logit rerun.
- `mwb verify` artifact-writing integration.
- SQLite schema and repair-index recovery for verification runs, verification results, intervention receipts, and telemetry reports.
- `docs/CAUSAL_VERIFICATION.md`, fixture, README, usage-guide, MechanismCard evidence examples, and buildout checklist updates.

Commands run:

```bash
uv run pytest tests/test_phase18_causal_verification.py
uv run pytest tests/test_phase18_causal_verification.py tests/test_causal_verification_integration.py
uv run ruff check src/mwb/causal_verification.py src/mwb/cli.py src/mwb/domain/objects.py src/mwb/domain/__init__.py src/mwb/sqlite_index.py tests/test_phase18_causal_verification.py tests/test_causal_verification_integration.py
uv run pytest tests/test_phase5_workflow.py tests/test_phase18_causal_verification.py
MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_causal_verification_integration.py -m integration
uv run mwb verify docs/fixtures/hypothesis_phase5.json --diagnostic-only --dry-run
uv sync
uv run ruff check .
uv run pytest
MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_causal_verification_integration.py -m integration
uv run mwb verify docs/fixtures/hypothesis_phase5.json --diagnostic-only --dry-run
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
git status --short --branch
```

Observed result:
- Phase 18 RGR tests first failed on missing `mwb.causal_verification` and `mwb verify` not writing run artifacts.
- Focused Phase 18 artifact suite passed, `6 passed`.
- Existing Phase 5 preflight/verify workflow tests still pass with artifact-writing verification.
- Real Pythia/SAE resample-ablation integration first reached the TransformerLens hook path and failed on hook signature, then passed after the hook accepted TransformerLens' keyword argument.
- `uv sync`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `84 passed, 3 skipped`.
- `MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_causal_verification_integration.py -m integration`: passed, `1 passed`; warnings were upstream TransformerLens/SAELens deprecations.
- `uv run mwb verify docs/fixtures/hypothesis_phase5.json --diagnostic-only --dry-run`: passed and wrote a diagnostic run directory with one planned resample-ablation operation.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite`: passed with `status: ok` and restored `verification_runs: 2`, `intervention_receipts: 3`, `verification_results: 3`, and `telemetry_reports: 1`.

Known residual risk:
- Exact verification receipts are intervention evidence, but mechanism-level claims still require clean controls, generalization, mediation where applicable, and claim-ledger review.

## Phase 19: Example Geometry And Control Audits

Status: complete pending commit and push
Commit: `e8801ac`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0005.md` control blockers, next-probe inputs, and density/control failure taxonomy.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/mechinterp_framework/0020_gpt.md` first-class example geometry, bundle audit, contamination, and balancing sections.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260624/ml_research/mechinterp_tracker/0070_revised.md` baseline calibration, control condition taxonomy, and specificity metrics.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260624/ml_research/mechinterp_tracker/0180_revised_v5.md` candidate claim requirements for token validation, matched controls, baseline calibration, and telemetry.
- `/home/home/p/g/n/learning/ml_research/self-ground/runs/e004_specificity_rescue_matrix/forensics/forensics_summary.md` and related E004 forensics tables.

Implemented:
- `ExampleGeometryReport`, `ControlContaminationReport`, and `BundleRebalanceProposal` domain objects.
- `BundleAuditService` for token validity, role balance, control contamination, baseline margin, heldout-template, and heldout-vocabulary checks.
- `mwb bundle audit <bundle>`.
- `mwb bundle rebalance --dry-run`.
- Persistence under `.mechanism/bundle_audits/` and `.mechanism/bundle_rebalance/`.
- SQLite schema and repair-index recovery for geometry reports, contamination reports, and rebalance proposals.
- SELF-GROUND E004 forensics links in bundle audit outputs when local artifacts are present.
- `docs/EXAMPLE_GEOMETRY.md`, README, usage guide, and buildout checklist updates.

Commands run:

```bash
uv run pytest tests/test_phase19_example_geometry.py
uv run ruff check src/mwb/bundle_audit.py src/mwb/cli.py src/mwb/domain/objects.py src/mwb/domain/__init__.py src/mwb/project.py src/mwb/sqlite_index.py tests/test_phase19_example_geometry.py
uv run mwb bundle audit negation_phase3_calibrated
uv run mwb bundle rebalance --dry-run
uv sync
uv run ruff check .
uv run pytest
uv run mwb bundle audit negation_phase3_calibrated
uv run mwb bundle rebalance --dry-run
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
git status --short --branch
```

Observed result:
- Phase 19 RGR tests first failed on missing `mwb.bundle_audit` and missing `mwb bundle`.
- Focused Phase 19 test suite passed, `6 passed`.
- Focused ruff check passed.
- `uv sync`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `90 passed, 3 skipped`.
- `uv run mwb bundle audit negation_phase3_calibrated`: passed with `status: warn`, no blockers, role-balance and missing-baseline warnings, and a SELF-GROUND E004 forensics link.
- `uv run mwb bundle rebalance --dry-run`: passed and produced control-family, heldout-template, and heldout-vocabulary proposals.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite`: passed with `status: ok` and restored `example_geometry_reports: 1`, `control_contamination_reports: 1`, and `bundle_rebalance_proposals: 1`.

Known residual risk:
- Bundle audits identify geometric and control-design risks. They do not automatically rewrite source bundles or prove behavioral validity under a live tokenizer/model.

## Phase 20: Diagnosis Tree And Probe Materialization

Status: complete
Commit: `9b9293d`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0005.md` next-probe planning, materialized probe requirements, and blocker provenance sections.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/mechinterp_framework/0020_gpt.md` mechanistic debugger, diagnosis tree, probe synthesis, and implemented-probe execution sections.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260624/ml_research/mechinterp_tracker/0430_revised_v6.md` scientific debt, negative evidence, and unresolved blocker treatment.

Implemented:
- `DiagnosisTree` and `MaterializedProbe` domain objects.
- `DiagnosisService` for run-local diagnosis trees, materialized probes, and implemented probe execution.
- Deterministic `ProbeRegistry` with implemented `sweep_axis_extension` and `switch_patch_mode` probe kinds.
- Blocked materialized probes for unsupported recommendations with `runnable: false` and no command.
- Source-provenance propagation from `run_manifest.json`, `control_metrics.json`, `blocker_report.json`, and `scientific_debt.json` into diagnosis/probe artifacts.
- `mwb diagnose <run>`.
- `mwb next-probe <run> --materialize`.
- `mwb run-probe <probe.yaml>` for implemented probes only.
- SQLite schema and repair-index recovery for `diagnosis_trees` and `materialized_probes`.
- `docs/DIAGNOSIS_AND_PROBES.md`, README, usage guide, fundamental checklist, target architecture, and buildout checklist updates.

Commands run:

```bash
uv run pytest tests/test_phase20_diagnosis_probes.py
uv run ruff check src/mwb/workflows/diagnosis.py src/mwb/workflows/next_probe.py src/mwb/cli.py src/mwb/domain src/mwb/sqlite_index.py tests/test_phase20_diagnosis_probes.py
uv run mwb diagnose latest
uv run mwb next-probe latest --materialize
uv sync
uv run ruff check .
uv run pytest
uv run mwb diagnose latest
uv run mwb next-probe latest --materialize
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
git status --short --branch
```

Observed result:
- Phase 20 RGR tests first failed on missing `mwb.workflows.diagnosis`.
- Focused Phase 20 diagnosis/probe suite passed, `6 passed`.
- Focused ruff check passed.
- `uv sync`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `96 passed, 3 skipped`.
- `uv run mwb diagnose latest`: passed and wrote a `DiagnosisTree` for `run_9815cd2998d6f99b` with primary blocker `insufficient_effect_size`.
- `uv run mwb next-probe latest --materialize`: passed and wrote a blocked materialized probe for unsupported `heldout_generalization`.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite`: passed with `status: ok` and restored `diagnosis_trees: 1` and `materialized_probes: 1`.

Known residual risk:
- The implemented probe runner intentionally covers only sweep axis extension and patch-mode switching. Other recommendation kinds remain recorded as blocked materialized probes until a concrete workflow runner exists.

## Phase 21: Reference Mechanism Suite

Status: complete
Commit: `79bbe4c`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/mechinterp_framework/0020_gpt.md` reference tasks with known ground truth, toy circuits, planted features, synthetic SAE dictionaries, and negative controls.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/mechinterp_framework/0010_claude.md` Tracr ground-truth circuits, calibration loop, empirical nulls, and FDR correction sections.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/BEST_EVALS_github.md` eval registry structure, deterministic assertions, contribution standards, and CI-oriented benchmark patterns.

Implemented:
- `ReferenceTask` and `ReferenceBenchmarkReport` domain objects.
- `ReferenceBenchmarkService` with a deterministic built-in `toy` suite.
- Planted residual-direction fixture requiring exact-effect recovery of `unit_direct_writer`.
- Negative-control surface-confound fixture that blocks the tempting high-proxy non-causal unit.
- Synthetic SAE split/absorption fixture with deterministic artifact detection.
- Empirical-null p-values, Benjamini-Hochberg q-values, proxy-vs-exact correlation, null seed counts, and calibration summary fields.
- `mwb benchmark framework`.
- Benchmark report persistence under `.mechanism/benchmarks/`.
- SQLite schema and repair-index recovery for `benchmark_reports` and `reference_tasks`.
- `docs/REFERENCE_MECHANISMS.md`, README, usage guide, fundamental checklist, and buildout checklist updates.

Commands run:

```bash
uv run pytest tests/test_phase21_reference_mechanisms.py
uv run ruff check src/mwb/reference_benchmarks.py src/mwb/cli.py src/mwb/domain src/mwb/sqlite_index.py tests/test_phase21_reference_mechanisms.py
uv run mwb benchmark framework
uv sync
uv run ruff check .
uv run pytest
uv run mwb benchmark framework
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
git status --short --branch
```

Observed result:
- Phase 21 RGR tests first failed on missing `mwb.reference_benchmarks`.
- Focused Phase 21 reference benchmark suite passed, `4 passed`.
- Focused ruff check passed.
- `uv sync`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `100 passed, 3 skipped`.
- `uv run mwb benchmark framework`: passed with `status: pass`, `task_count: 3`, planted mechanism recovery, false-positive blocking, and synthetic SAE artifact detection.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite`: passed with `status: ok` and restored `benchmark_reports: 1` and `reference_tasks: 3`.

Known residual risk:
- The built-in `toy` suite is deterministic and CI-friendly. It does not replace future optional heavy integrations with Tracr, ACDC/EAP, SAEBench/RAVEL, or live backend reference circuits.

## Phase 22: Rich Claim Grammar

Status: complete
Commit: `9ed966c`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0005.md` evidence tiers, claim atoms, blocker-to-claim mapping, Draft Guard behavior, caveats, and scientific debt.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/mech_specs.md` richer claim types and mechanism evidence requirements.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0006.md` claim-bearing gate and non-upgrading override requirements.

Implemented:
- `ClaimGrammarReport` domain object.
- `ClaimGrammarService` with deterministic claim-intent matching.
- Evidence requirement resolution for association, projection, causal necessity, causal sufficiency, mediation, generalization, and mechanism claims.
- Blocker and unresolved scientific-debt handling for blocked and caveated claims.
- Visible inline override records that cannot upgrade blocked claims.
- `mwb claim check <claim-json>`.
- Draft Guard integration that runs typed claim grammar before the older phrase fallback while preserving legacy `blocked_terms`.
- SQLite schema and repair-index recovery for `claim_grammar_reports`.
- `docs/CLAIM_GRAMMAR.md`, fixture, README, usage guide, fundamental checklist, and buildout checklist updates.

Commands run:

```bash
uv run pytest tests/test_phase22_claim_grammar.py
uv run ruff check src/mwb/claim_grammar.py src/mwb/workflows/draft_guard.py src/mwb/cli.py src/mwb/domain src/mwb/sqlite_index.py tests/test_phase22_claim_grammar.py
uv sync
uv run ruff check .
uv run pytest
uv run mwb draft-check docs/fixture_draft.md
uv run mwb claim check docs/fixtures/claim_association.json
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
git status --short --branch
```

Observed result:
- Phase 22 RGR tests first failed on missing `mwb.claim_grammar`.
- Focused Phase 22 claim grammar suite passed, `6 passed`.
- Focused Draft Guard plus claim grammar regression suite passed, `9 passed`.
- Focused ruff check passed.
- `uv sync`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `106 passed, 3 skipped`.
- `uv run mwb draft-check docs/fixture_draft.md`: passed with `status: allowed`.
- `uv run mwb claim check docs/fixtures/claim_association.json`: passed with `status: allowed`.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite`: passed with `status: ok` and restored `claim_grammar_reports: 1`.

Known residual risk:
- The grammar is deterministic and local. It does not parse arbitrary scientific prose perfectly; claims should use explicit `[CLAIM:<ref>]` tags or JSON fixtures for paper-facing enforcement.

## Phase 23: Policy Profiles And Research Taste

Status: complete
Commit: `bbab83d`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/mechinterp_framework/0020_gpt.md` research taste policies and configurable `PolicyProfile` schema.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0005.md` evidence tiers, blockers, and claim grammar boundaries.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260624/ml_research/mechinterp_tracker/0430_revised_v6.md` scientific debt visibility, waiver policy, and configurable debt/blocker handling.

Implemented:
- `PolicyEvaluationReport` domain object.
- Built-in `strict` and `exploratory` policy profiles.
- Default `[policy] profile = "strict"` in new project config, with strict fallback for older projects.
- `PolicyProfileService` for project profile loading, profile evaluation, and claim-bearing verification policy checks.
- Strict zero-ablation `diagnostic_only` claim ceiling.
- Strict paired noising/denoising and resample-ablation requirements for claim-bearing candidate verification.
- Policy-profile integration in claim grammar and Draft Guard.
- Policy claim-ceiling integration in MechanismCard generation.
- `mwb policy check`.
- SQLite schema and repair-index recovery for `policy_evaluations`.
- `docs/POLICY_PROFILES.md`, README, usage guide, fundamental checklist, and buildout checklist updates.

Commands run:

```bash
uv run pytest tests/test_phase23_policy_profiles.py
uv run ruff check src/mwb/policy_profiles.py src/mwb/claim_grammar.py src/mwb/causal_verification.py src/mwb/workflows/cards.py src/mwb/cli.py src/mwb/domain src/mwb/project.py src/mwb/sqlite_index.py tests/test_phase23_policy_profiles.py
uv sync
uv run ruff check .
uv run pytest
uv run mwb policy check
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
git status --short --branch
```

Observed result:
- Phase 23 RGR tests first failed on missing `mwb.policy_profiles`.
- Focused Phase 23 policy profile suite passed, `5 passed`.
- Focused ruff check passed.
- `uv sync`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `111 passed, 3 skipped`.
- `uv run mwb policy check`: passed with `policy_profile: strict` and `status: pass`.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite`: passed with `status: ok` and restored `policy_evaluations: 1`.

Known residual risk:
- Policy profiles encode deterministic local gates. They do not remove the need for human review of new lab standards, waivers, or domain-specific thresholds.

## Phase 24: Adapter Expansion With Conformance

Status: complete
Commit: `17d3045`
Pushed: yes

Required reading completed:
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/0006.md` adapter strategy, provenance archive, P1 acceptance, and no diagnostic-to-evidence promotion invariant.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260624/ml_research/mechinterp_tracker/0010_mechinterp_tracker_gpt.md` technique coverage and external backend landscape.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/mech.md` ecosystem survey for nnsight, nnterp, pyvene, Neuronpedia, and artifact tooling.

Implemented:
- Optional `NNsightAdapter` with capability manifest, version manifest, model identity, module TensorSpace mapping, dry-run conformance, missing dependency diagnostics, and an honest real-trace path when `nnsight` is installed.
- Optional `PyVeneAdapter` with capability manifest, version manifest, intervention-contract serialization, missing dependency diagnostics, and no fake backend execution pass.
- Read-only `NeuronpediaAdapter` with external metadata URI refs, dictionary/unit refs, optional HTTP metadata fetch, and explicit no-write/no-claim posture.
- Stable adapter manifest and backend-version refs plus uniform conformance artifact persistence under `.mechanism/adapters/<adapter>/`.
- SQLite repair-index restoration for `adapter_manifests` and `backend_versions`.
- Claim-bearing gate now blocks adapters whose manifest declares `claim_bearing.supported=false`.
- Git LFS, DVC, and git-annex artifact pointer detection in `ArtifactRegistry`, recorded with `materialized=false`.
- `mwb adapter conformance nnsight`, `pyvene`, and `neuronpedia` CLI commands.
- `docs/ADAPTERS.md`, README, usage guide, fundamental checklist, and buildout checklist updates.

Commands run:

```bash
uv run pytest tests/test_phase24_adapter_expansion.py
uv run pytest tests/test_phase3_adapters.py tests/test_phase1_domain.py
uv run ruff check src/mwb/adapters src/mwb/artifacts.py src/mwb/sqlite_index.py src/mwb/cli.py tests/test_phase24_adapter_expansion.py
uv sync
uv run ruff check .
uv run pytest
uv run mwb adapter conformance transformer-lens --model EleutherAI/pythia-70m-deduped --device cpu
uv run mwb adapter conformance saelens --model EleutherAI/pythia-70m-deduped --hook blocks.2.hook_resid_post --device cpu
uv run mwb adapter conformance nnsight --model gpt2 --module-path transformer.h.0.mlp --device cpu --dry-run
uv run mwb adapter conformance pyvene --model gpt2 --module-path transformer.h.0.mlp --intervention-kind resample_ablation --device cpu --dry-run
uv run mwb adapter conformance neuronpedia --model-id gemma-2-2b --sae-id 20-gemmascope-res-16k --feature-index 123 --dry-run
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
git status --short --branch
```

Observed result:
- Phase 24 RGR tests first failed on missing `mwb.adapters.neuronpedia`.
- Focused Phase 24 adapter expansion suite passed, `8 passed`.
- Focused adapter/domain regression suite passed, `19 passed`.
- Focused ruff check passed.
- `uv sync`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `119 passed, 3 skipped`.
- TransformerLens conformance: passed with real model load and activation capture.
- SAELens conformance: passed with real SAE load and feature ref round-trip.
- Optional nnsight, pyvene, and Neuronpedia dry-run conformance: passed with `diagnostic_only` posture.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite`: passed with `status: ok` and restored `adapter_manifests: 5`, `backend_versions: 5`.

Known residual risk:
- Optional nnsight and pyvene dependencies are not installed in the current QC environment. Their adapters therefore remain diagnostic-only unless a user installs/configures those backends and runs real conformance; this is deliberate and prevents fake support.

## Phase 25: Release Hardening

Status: complete; implementation commit pending
Commit: pending
Pushed: pending

Required reading completed:
- Full `docs/world_class_buildout/` docset, especially source mining findings, target architecture, implementation plan, phased checklist, and QC/commit/push protocol.
- Current `docs/mwb_phase0_ledger.md` through Phase 24.
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/BEST_EVALS_github.md` eval engineering patterns: deterministic checks, versioned evals, reproducibility docs, commandable CI-style gates, and failure analysis.

Implemented:
- Release-hardening regression suite for known false-positive blocking and overclaim blocking.
- Compatibility test for legacy adapter manifest/backend-version records without new stable refs.
- README docs-link regression for release report links.
- Public CLI help snapshot coverage for root commands, adapter conformance commands, and pyvene options.
- `docs/RELEASE_HARDENING_REPORT.md` with release scope, command gate, scan policy, and result slots.
- README and buildout checklist updates.

Commands run:

```bash
uv run pytest tests/test_phase25_release_hardening.py
uv run ruff check tests/test_phase25_release_hardening.py
uv sync
uv run ruff check .
uv run pytest
MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_phase4_context.py -m integration
uv run mwb adapter conformance transformer-lens --model EleutherAI/pythia-70m-deduped --device cpu
uv run mwb adapter conformance saelens --model EleutherAI/pythia-70m-deduped --hook blocks.2.hook_resid_post --device cpu
uv run mwb graph rebuild
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
rg -n "fake|dummy|mock|simulated|placeholder|smoke" src tests docs README.md pyproject.toml
rg -n "implements|mechanism for|proves|isolated.*circuit|strong_candidate_evidence" src tests docs README.md pyproject.toml
git status --short --branch
```

Observed result:
- Phase 25 RGR tests first failed on missing `docs/RELEASE_HARDENING_REPORT.md` link in README.
- Focused Phase 25 release-hardening suite passed, `4 passed`.
- Focused ruff check passed.
- `uv sync`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `123 passed, 3 skipped`.
- Real adapter integration test: passed, `1 passed, 3 deselected`.
- TransformerLens conformance: passed with real model load and activation capture.
- SAELens conformance: passed with real SAE load and feature ref round-trip.
- `uv run mwb graph rebuild`: passed with `30` edges and `23` nodes.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite`: passed with `status: ok` and restored `adapter_manifests: 5`, `backend_versions: 5`, `evidence_edges: 30`.
- Release scans reviewed; hits were expected protocol/report text, blocked-language tables, tests asserting overclaim blocking, historical ledger entries, and source-mined anti-pattern notes.

Known residual risk:
- Optional nnsight and pyvene dependencies remain absent from the current QC environment and therefore remain diagnostic-only until installed/configured for real conformance.
