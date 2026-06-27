# World-Class Buildout Docset

This docset mines the broader pre-archive material the canonical Phase 0 implementation did not previously consume:

- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260624/ml_research/mechinterp_tracker/*.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260625/mi_docs/**/*.md`

It is a buildout plan for making Mechanistic Workbench world-class without losing the constraints that made Phase 0 useful: local-first, IPython-native, real backend adapters, file-backed evidence, claim-safe language, and no fake execution paths.

## Source Scale

- 31 tracker markdown files, 53,538 lines.
- 22 `mi_docs` markdown files, 8,063 lines.
- 53 markdown files total, 61,601 lines.

## Reading Order

1. `00_source_mining_findings.md`
2. `01_target_architecture.md`
3. `02_implementation_plan.md`
4. `03_phased_tdd_checklist.md`
5. `04_qc_commit_push_protocol.md`

## Scope Rule

This docset separates:

- **fundamental buildout**: robust generalized features that make the local workbench scientifically stronger;
- **integration buildout**: adapter and interop work that should happen only after the core behavior is proven;
- **deferred platform work**: dashboard-first, cloud, multi-user, NSHKR service split, and broad ecosystem parity.

The implementation sequence must use TDD/RGR. No phase may be accepted by smoke tests, synthetic-only fixtures, or prose-only claims.

## Current Baseline

The repo already has a complete Phase 0 local workbench. This docset starts from that baseline and plans the next buildout. It does not re-open settled Phase 0 decisions unless the mined sources reveal a correctness gap.
