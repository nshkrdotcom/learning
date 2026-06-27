# QC, Commit, And Push Protocol

This protocol applies to every phase in `03_phased_tdd_checklist.md`.

## Non-Negotiables

- Use TDD/RGR.
- No smoke-only acceptance.
- No fixture-only acceptance for claim-bearing behavior.
- No fake backend path.
- No silent diagnostic-to-claim upgrade.
- No hidden failed controls.
- No uncommitted phase completion.
- No push without QC-green.

## Phase Loop

1. Read required sources.
2. Write or update the failing test.
3. Run the focused test and confirm red.
4. Implement the smallest robust code path.
5. Run focused test and confirm green.
6. Add regression coverage for the discovered edge.
7. Update docs and ledger.
8. Run full QC.
9. Review diff.
10. Commit.
11. Push.
12. Record commit hash in ledger.

## Minimum QC

```bash
uv sync
uv run ruff check .
uv run pytest
uv run mwb doctor
uv run mwb repair-index --output .mechanism/workbench.repaired.sqlite
git status --short --branch
```

## Real Adapter QC

Run when touching adapters, TensorSpace, activation capture, feature ranking, static compiler, or causal verification:

```bash
MWB_RUN_REAL_ADAPTER_TESTS=1 uv run pytest tests/test_phase4_context.py -m integration
uv run mwb adapter conformance transformer-lens --model EleutherAI/pythia-70m-deduped --device cpu
uv run mwb adapter conformance saelens --model EleutherAI/pythia-70m-deduped --hook blocks.2.hook_resid_post --device cpu
```

## Scan QC

Run before release or after claim/evidence/draft changes:

```bash
rg -n --glob '!docs/mwb_phase0_ledger.md' --glob '!docs/PHASE*_REPORT.md' "fake|dummy|mock|simulated|placeholder|smoke" src tests docs README.md pyproject.toml
rg -n --glob '!docs/mwb_phase0_ledger.md' --glob '!docs/PHASE*_REPORT.md' "implements|mechanism for|proves|isolated.*circuit|strong_candidate_evidence" src tests docs README.md pyproject.toml
```

Expected overclaim hits are allowed only in:

- blocked-language tables,
- mechanism-tier allowed-language definitions,
- tests asserting blocking,
- fixture cards explicitly labeled non-claim-bearing,
- audit reports documenting the scan.

## Documentation Requirements

Every phase updates at least one of:

- `docs/USAGE.md`,
- feature-specific docs,
- buildout checklist,
- phase ledger,
- acceptance/hardening report.

Docs must state:

- commands,
- input files,
- output files,
- claim-bearing boundaries,
- failure modes,
- how to reproduce.

## Commit Discipline

Commit message format:

```text
phase N: <specific capability>
```

or for ledger-only follow-up:

```text
docs: record phase N commit
```

Each ledger entry records:

- status,
- commit hash,
- pushed yes/no,
- required reading,
- implementation summary,
- tests,
- commands,
- observed results,
- residual risk.

## Push Discipline

Push every completed phase:

```bash
git push
```

After pushing:

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
```

HEAD and origin must match before reporting completion.

## Handling Failures

If QC fails:

- keep the phase open,
- fix with RGR,
- do not commit partial green-looking work,
- document any true external blocker,
- never relabel a failed real integration as "passed by dry-run".

If a dependency is unavailable:

- mark tests with an explicit integration marker,
- keep non-integration behavior covered locally,
- record the skipped integration condition,
- do not claim adapter support.
