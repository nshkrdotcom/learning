# Research Ledgers

Mechanistic Workbench keeps human-approved research state in Git-visible flat files under `research/`.

Canonical ledgers:

- `research/logs/claim_ledger.md`
- `research/logs/run_ledger.csv`
- `research/logs/decision_log.md`
- `research/logs/research_log.md`

Generated local proposals are written under `.mechanism/` and must be reviewed before copying into the committed ledgers.

## Validate

```bash
uv run mwb ledger validate
```

Validation checks parser contracts and indexes valid entries into SQLite:

- claim entries into `claims`;
- decision entries into `decisions`;
- research log entries into `research_log_entries`;
- run ledger rows into `runs`.

SQLite remains rebuildable. `mwb rebuild-index` and `mwb repair-index` re-parse the Git-visible ledgers and restore indexed rows.

## Claim Ledger

Each claim starts with an H3 heading:

```markdown
### C001 - Short claim title
```

It must be followed by a fenced `yaml` block with at least:

```yaml
claim_id: C001
status: single_run_evidence
allowed:
  - "is associated with"
forbidden:
  - "is the mechanism"
```

The parser enforces:

- `claim_id` matches the heading ID;
- required fields exist;
- `allowed` and `forbidden` are lists of strings;
- duplicate claim IDs fail validation;
- status is a known claim status.

## Run Ledger

`research/logs/run_ledger.csv` must use the canonical column order:

```text
date,run_id,git_commit,phase,purpose,hypothesis,command,model,hook_point,sae_release,sae_id,ranking_dir,out_dir,seed,per_family,top_k_features,baseline_mode,operations,status,blocker,key_metric_1,key_metric_2,artifact_paths,decision
```

Run rows are human-approved. The workbench only proposes rows:

```bash
uv run mwb ledger propose-run <run-ref>
```

The proposal is written to:

```text
.mechanism/runs/<run-ref>/run_ledger_row.csv
```

## Decision Log

Each decision starts with an H2 heading:

```markdown
## D001 - Short decision title
```

It must be followed by a fenced `yaml` block with at least:

```yaml
decision_id: D001
status: accepted
```

Allowed statuses:

- `proposed`
- `accepted`
- `superseded`
- `rejected`

## Research Log

Each research log entry starts with a date heading:

```markdown
## 2026-06-26
```

It must include a fenced `yaml` block with an `entry_id`:

```yaml
entry_id: R2026-06-26-001
linked_runs: []
linked_claims: []
linked_decisions: []
open_questions: []
copilot_session_id: null
```

## Claim Proposals

```bash
uv run mwb ledger propose-claim <card-ref>
```

The proposal is generated from a MechanismCard and written to:

```text
.mechanism/proposals/claims/<claim-id>.md
.mechanism/proposals/claims/<claim-id>.json
```

These files are review artifacts. The command does not silently append to `research/logs/claim_ledger.md`.

## Failure Boundary

Ledger validation proves structure and local traceability only. It does not make a claim scientifically true, and it does not promote claim status without human review.
