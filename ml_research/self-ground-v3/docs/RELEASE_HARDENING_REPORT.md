# Release Hardening Report

This report records the release-hardening gate for the world-class buildout.

## Scope

Phase 25 hardens the local workbench after the evidence graph, ledgers,
hypothesis lifecycle, static compiler, causal verification, example geometry,
diagnosis/probes, reference mechanisms, claim grammar, policy profiles, and
adapter expansion phases.

The release goal is not to add a new scientific claim. It is to prove that the
repo has executable regressions for prior false positives/negatives, can read
older file-backed state, documents public commands, and still passes the full
QC gate.

## Regression Coverage

- Known false-positive confound remains blocked by the reference mechanism suite.
- Control-leaky MechanismCard prose still blocks mechanism wording such as
  `implements`.
- Legacy adapter manifests and backend version files without stable refs still
  rebuild into SQLite by directory ref.
- README doc links are checked by tests.
- Public CLI help snapshots cover root commands, adapter conformance commands,
  and pyvene options.

## Commands

```bash
uv run pytest tests/test_phase25_release_hardening.py
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

## Observed Results

- Release hardening suite: passed, `4 passed`.
- `uv sync`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, `123 passed, 3 skipped`.
- Real adapter integration test: passed, `1 passed, 3 deselected`.
- TransformerLens conformance: passed with real model load and activation capture.
- SAELens conformance: passed with real SAE load and feature ref round-trip.
- Graph rebuild: passed with `30` edges, `23` nodes, and `7` relation kinds.
- `uv run mwb doctor`: passed with `status: ok`.
- `uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite`: passed with `status: ok`, `adapter_manifests: 5`, `backend_versions: 5`, and `evidence_edges: 30`.
- Scan review: passed. Hits were limited to protocol/report text, blocked-language tables, tests asserting blocked overclaims, historical ledger entries, and source-mined anti-pattern notes.

## Prior Green Baseline

The preceding Phase 24 gate passed:

- `uv run pytest`: `119 passed, 3 skipped`.
- `uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite` restored
  `adapter_manifests: 5` and `backend_versions: 5`.

## Scan Policy

The release scans are not expected to be empty because the repo intentionally
contains blocked-language tables, tests asserting blocked overclaims, and QC
protocol text that names forbidden patterns. All scan hits must be explainable
as one of:

- blocked-language or claim-grammar enforcement;
- tests that assert overclaim blocking;
- protocol/report text describing forbidden patterns;
- historical ledger entries.

No scan hit may represent a live fake backend path, smoke-only acceptance, or an
unqualified paper-facing mechanism claim.

The final scan review found no live fake backend path, no smoke-only acceptance
criterion, and no unqualified paper-facing mechanism claim.

## Residual Risk

Optional nnsight and pyvene dependencies are not installed in the current QC
environment. Their adapters remain diagnostic-only unless a user installs and
configures those backends for real conformance.
