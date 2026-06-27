# Phase 9 Hardening Report

Status: release candidate.

## Commands

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

## Results

- `uv sync`: passed; lockfile state already resolved.
- Ruff: passed.
- Pytest: passed, `42 passed, 1 skipped`.
- Init and doctor: passed with `status: ok`.
- Real integration test: passed; dependency deprecation warnings came from upstream packages.
- TransformerLens conformance: passed with `EleutherAI/pythia-70m-deduped`, activation shape `[1, 5, 512]`.
- SAELens conformance: passed with `pythia-70m-deduped-res-sm` / `blocks.2.hook_resid_post`.
- Non-real-work scan: no matches.
- Overclaim scan: matches are restricted to blocked-language declarations, fixture cards, and tests that verify blocking. No generated claim text upgrades E004 beyond association-tier language.

Phase 10 completion extended this gate to `50 passed, 1 skipped`, added resume/sweep/rebuild coverage, and preserved the same E004 evidence boundary.

## Release Boundary

The release candidate is suitable for local dogfood and reproducible Phase 0 evidence hygiene. The real E004 dogfood run remains non-claim-bearing because its own adjudication is `insufficient_evidence` and the generated blocker report is `control_leaky`.
