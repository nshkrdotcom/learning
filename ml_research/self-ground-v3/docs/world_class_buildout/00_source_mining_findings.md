# Source Mining Findings

## Source Inventory

### 20260624 Tracker Evolution

The tracker corpus evolves from broad "complete professional mechinterp tracker" specs into a narrower MechLedger/ISOMORPH-style discipline. Files mined:

- `0001_mechinterp_tracker_gemini.md`
- `0010_mechinterp_tracker_gpt.md`
- `0020_critique_claude.md`
- `0030_revised_gpt.md`
- `0040_revised.md`
- `0050_with_research_framework_idea.md`
- `0060_critique_claude.md`
- `0070_revised.md`
- `0080_critique_gemini.md`
- `0090_critique_claude.md`
- `0100_revised.md`
- `0110_critique_claude.md`
- `0120_revised.md`
- `0130_critique_claude.md`
- `0140_revised.md`
- `0150_critique_claude.md`
- `0151_critique_gemini.md`
- `0160_revised_v4.md`
- `0170_critique_gemini.md`
- `0180_revised_v5.md`
- `0190_critique_gemini.md`
- `0200_v5_1.md`
- `0210_critique_gemini.md`
- `0220_v5_2.md`
- `0300_research_landscape_for_git_native_research_integrity_systems.md`
- `0400_critique_claude.md`
- `0410_revised_v6.md`
- `0420_critique_claude.md`
- `0430_revised_v6.md`
- `0431_selfground_refactor.md`
- `0432_selfground_refactor.md`

### 20260625 Additional `mi_docs`

The `mi_docs` corpus adds product, architecture, eval-isomorphism, IPython, and framework-design context. Files mined:

- `BEST_EVALS_github.md`
- `CLAUDE_LAST_WORD.md`
- `EVALS_ARE_MECHINTERP_claude.md`
- `EVALS_ARE_MECHINTERP_nshkr.md`
- `EVALS_nshkr.md`
- `GEMINI_LAST_WORD_2of2.md`
- `GEMINI_LAST_WORD_wow_or_slop_1of2.md`
- `GPT_LAST_WORD_1of2.md`
- `GPT_LAST_WORD_1of2_synthesis_with_claude.md`
- `GPT_LAST_WORD_2of2.md`
- `GPT_LAST_WORD_2of2_synthesis_with_claude.md`
- `MechanisticWorkbench_PRD.md`
- `could_we_cut_anything.md`
- `ipython_integration_discussion.md`
- `mech.md`
- `mech_discuss.md`
- `mech_interp_framework_gptDeepResearch.md`
- `mech_specs.md`
- `mechinterp_framework/0001_gemini.md`
- `mechinterp_framework/0010_claude.md`
- `mechinterp_framework/0020_gpt.md`
- `senior_management_note.md`

## Converged Decisions

### 1. The Product Is Not An Execution Backend

The corpus repeatedly rejects building another TransformerLens, SAELens, nnsight, pyvene, or generic intervention framework. The winning product is the layer above those tools:

- typed identity,
- provenance capture,
- static preflight,
- exact verification orchestration,
- run/artifact evidence discipline,
- claim grammar,
- draft guard,
- scientific debt,
- next-probe diagnosis.

Implementation consequence: all future execution features must be adapter-backed and conformance-tested. No generic backend wrapper may claim support without a real integration path.

### 2. IPython-First Is The Product Center

The mi_docs framework critiques repeatedly say the mistake is starting from claims or formal contracts. The researcher's real loop is exploratory:

```text
inspect activations -> form a hunch -> slice examples -> patch something
-> controls move too -> inspect why -> revise -> later write a claim
```

Implementation consequence: the workbench must remain scratch-friendly. Formal gates apply when promoting hypotheses or checking prose, not when exploring.

### 3. Git-Native Evidence Is A Hard Invariant

The late tracker docs converge on:

- human-readable committed research state,
- ignored local run directories,
- SQLite as a rebuildable index,
- flat-file ledgers for claims/runs/decisions/debt,
- no hidden canonical database.

Implementation consequence: every new capability must write canonical files or reconstructable JSONL/JSON/Markdown records. SQLite can accelerate queries but cannot be the only durable source.

### 4. Claim Status And Workflow State Must Not Collapse

Critiques identify a recurring taxonomy bug: "exploratory signal", "candidate", "paper-ready", "causal support", and "replicated evidence" are not one flat enum. Workflow stage and evidential strength are separate axes.

Implementation consequence: future Hypothesis state, evidence tier, run status, and claim status should be modeled as distinct fields with explicit transition gates.

### 5. Evidence Graph Beats Flat Lists

The corpus keeps returning to graph questions:

- which cell created this feature table?
- what claims depend on this SAE dictionary?
- what controls contradicted this mechanism?
- what changed between believed and failed notebook states?
- which artifacts support or contradict this claim?

Implementation consequence: parent refs are not enough long-term. The index needs typed edges with relations such as `supports`, `contradicts`, `depends_on`, `derived_from`, `same_space_as`, `tested_by`, `confounded_by`, `fails_on`, and `generalizes_to`.

### 6. Static Algebra Is The Compute-Saving Differentiator

The ISOMORPH/mechinterp_framework notes consistently identify the missing field-level capability: a typed algebra compiler that kills structurally impossible hypotheses before GPU sweeps.

Implementation consequence: static preflight should grow from a simple projection check into a typed compiler over TensorSpace, MechanisticUnitRef, dictionary geometry, DLA/logit-lens, OV/QK composition, attribution estimates, and off-manifold risk.

### 7. Alternative Explanations Are Product-Critical

The strongest framework notes argue that a world-class tool does not only say "insufficient evidence." It keeps live confounds and suggests discriminating tests:

- control leakage,
- generic sentiment,
- token-position artifact,
- dictionary neighbor interference,
- self-repair,
- off-manifold intervention,
- task calibration failure.

Implementation consequence: BlockerReport should feed an AlternativeExplanation graph and deterministic probe templates, not only a single next command.

### 8. Example Geometry Is Fundamental

SELF-GROUND's task calibration failures show that prompt/control geometry is not metadata. Example bundles need auditability:

- lexical overlap,
- semantic similarity,
- token length,
- baseline margin,
- target/foil token validity,
- syntactic template,
- position semantics,
- role separability.

Implementation consequence: ExampleBundle and ControlBundle should gain audit reports, role balance checks, contamination detection, and rebalance/materialization workflows.

### 9. Evidence Needs Reference Mechanisms

The framework notes call for benchmark fixtures for the framework itself:

- Tracr/RASP compiled models with known circuits,
- toy transformers with planted features,
- synthetic SAE dictionaries with known splitting/absorption,
- known IOI/path-patching examples,
- negative controls with tempting confounds.

Implementation consequence: world-class confidence requires known-ground-truth tests, not only SELF-GROUND dogfood or synthetic file fixtures.

### 10. Draft Guard Should Become Claim Grammar

Phrase blocking is useful but insufficient. The richer grammar has claim types:

- observation,
- static projection,
- causal necessity,
- causal sufficiency,
- mediation,
- generalization,
- mechanism.

Each claim type has required evidence and allowed scope.

Implementation consequence: Draft Guard should parse claim intents and compare them with structured MechanismCards and evidence graph state. Phrase matching remains a deterministic safety net.

## Anti-Patterns To Avoid

- Building a dashboard before the IPython loop is excellent.
- Adding fake backend abstractions.
- Promoting fixture outputs into claim-bearing evidence.
- Treating SQLite as canonical.
- Hiding blocked/null/contradictory evidence.
- Requiring researchers to author heavyweight schemas before exploring.
- Letting execution code become the authority for claims.
- Adding broad adapters without conformance fixtures.
- Treating raw autointerp labels as evidence.
- Collapsing "workflow stage" and "evidence tier" into a single status.

## Source Tensions Resolved For This Repo

### MechLedger Versus Mechanistic Workbench

The MechLedger docs want a dependency-light CLI. Mechanistic Workbench currently has real TransformerLens and SAELens as hard dependencies because Phase 0 needed a real IPython dogfood loop. The long-term resolution is:

- Workbench remains the research UX and adapter-backed local monolith.
- A future extracted ledger/audit kernel can be made dependency-light.
- Heavy execution dependencies must not leak into pure ledger/draft-guard logic.

### Local Workbench Versus NSHKR Integration

The eval-is-mechinterp notes argue that mechanism verification should eventually be a Mezzanine domain plugin with Citadel authorization and AITrace observation. The canonical Phase 0 decision still stands: build the local monolith first. NSHKR integration starts only after the local evidence lifecycle is robust.

### Claim Compiler Versus Investigation Workbench

The framework notes warn against making the tool claim-first. The implementation plan below keeps the order:

```text
capture -> type -> compare -> hypothesize -> preflight -> verify -> diagnose -> claim
```

## Immediate Buildout Implications

The next serious work should not be another prose pass. It should implement, with TDD/RGR:

1. typed evidence graph queries,
2. flat-file claim/run/decision/debt ledgers,
3. hypothesis state machine and alternative explanations,
4. richer example/control geometry audits,
5. static mechanistic compiler expansion,
6. exact causal verification operations beyond dry-run planning,
7. deterministic probe materialization,
8. known-ground-truth reference tasks,
9. richer claim grammar,
10. adapter conformance expansion only after core behavior is stable.
