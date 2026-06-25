# TDD Checklist

Source of truth: `0430_revised_v6.md`, with `0431_selfground_refactor.md` and `0432_selfground_refactor.md` used as reuse boundaries.

## Completed Foundation

- [x] Create a new dependency-light package in `self-ground-v2`.
- [x] Use `uv` project metadata and a `mechledger` console script.
- [x] Write parser tests before implementation for claim, decision, and experiment contracts.
- [x] Implement claim ledger heading grammar and YAML validation.
- [x] Implement decision log heading grammar and YAML validation.
- [x] Implement ExperimentSpec YAML and prerequisite parsing.
- [x] Implement canonical claim YAML hashing with order-insensitive language/debt lists.
- [x] Write Draft Guard tests for Markdown, LaTeX, HTML-comment tags, forbidden language, caveats, debt, unknown tags, malformed tags, and inline overrides.
- [x] Implement deterministic Draft Guard without semantic/LLM checks.
- [x] Write run-capture tests for run ID format, run directory contract, stdout/stderr capture, heartbeat cleanup, auto-collect artifacts, SDK logging, alias resolution, artifact attach, and annotation.
- [x] Implement run wrapper, SDK context manager, alias cache, artifact manifest, and event logs.
- [x] Write workflow tests for scaffold init, index check, scientific debt gate check, next-action classification, and CLI surface.
- [x] Implement research scaffold, pre-commit hook generation, SQLite index rebuild, `status`, `next`, `format`, `session close`, and `experiment crystallize`.
- [x] Implement basic Tier 2 debt checks for run class, baseline calibration, positive control, empirical null seed count, nonfinite rows, skipped rows, norm drift, and annotated supporting artifacts.

## Deferred Or Partially Implemented

- [ ] Full claim proposal diff/apply workflow with stale proposal detection.
- [ ] Decision detection from declared config diffs.
- [ ] Full run reclassification and waiver command surfaces.
- [ ] Notebook/IPython `%mech` magic registration.
- [ ] Empirical null plan/register command.
- [ ] Paired-test SDK computation helpers.
- [ ] Prediction lock/score workflow.
- [ ] Bundle, archive, garbage collection, and redaction commands.
- [ ] Dashboard, copilot, external labels, advanced technique schemas, and RO-Crate export.

## QC Targets

- [x] `uv run pytest`
- [x] `uv run ruff check .`
- [x] `uv run mechledger --help`
