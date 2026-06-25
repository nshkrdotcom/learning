# MechLedger Usage Guide

## 10-Minute Draft Guard Setup

```bash
uv run mechledger init
uv run mechledger install-hooks
pre-commit install
```

Add claims to `research/logs/claim_ledger.md`, then tag prose:

```markdown
This is preliminary single-run evidence. [CLAIM:C001]
```

Run:

```bash
uv run mechledger draft check research/paper/draft.md
```

Draft Guard supports `[CLAIM:C003]`, `\claim{C003}`, and
`<!-- CLAIM:C003 -->`. It enforces forbidden phrases, required caveats,
unresolved debt flags, and visible same-paragraph overrides.

## Wrapping An Existing Script

```bash
uv run mechledger run \
  --experiment E001 \
  --class diagnostic \
  --purpose "verify decoded SAE intervention path" \
  --hypothesis "patch produces finite deltas" \
  -- python scripts/run_existing.py --config configs/e001.yaml
```

The wrapper captures command, stdout/stderr, git state, environment allowlist,
resource usage, a heartbeat while active, run-local artifacts, a run-ledger row
proposal, a claim proposal, and a scientific-debt report.

MechLedger does not execute interventions for you. Your script remains native
TransformerLens/PyTorch/SAELens code if that is what the research needs.

## SDK In A Notebook Or Script

Inside a process launched by `mechledger run`, use:

```python
import mechledger as ml

ml.log_metric("specificity_gap_mean", 0.123)
ml.log_event("intervention_completed", "ablation finished")
ml.log_intervention_metadata(
    target_hook="blocks.2.hook_resid_post",
    operation="ablate",
    features_modified=["sae_12300"],
)
```

The SDK writes JSONL records to the active run directory from the environment
variables set by `mechledger run`. It imports no heavy ML libraries.

## Run-Local Artifact Auto-Collection

Write outputs into:

```text
.mechledger/runs/RUN_ID/artifacts/
```

At run completion, MechLedger adds those files to `artifact_manifest.json` with:

```yaml
claim_relevance: none
review_status: unannotated
```

They cannot support a claim until annotated.

## Registering And Annotating Artifacts

```bash
uv run mechledger attach latest results/e001.jsonl --claim-relevance supporting
uv run mechledger artifact annotate latest A001 --claim-relevance supporting
```

MechLedger records path, hash, size, storage backend, relevance, and review
status. It does not discover arbitrary artifacts outside registered paths or
run-local artifact directories.

## Short Run Aliases

Commands that take a run ID also accept:

```text
latest
latest:2
#1
unique run-id prefix
unique experiment/slug prefix
```

Alias resolution reads `.mechledger/alias_cache.txt`, not a routine directory
sweep.

## Crystallizing Exploratory Runs

```bash
uv run mechledger experiment crystallize \
  --runs latest \
  --id E003 \
  --title "Observed induction-like early head"
```

This creates a draft ExperimentSpec with `source_runs` populated. It does not
promote claims and does not mutate run directories.

## Reviewing A Claim Update

```bash
uv run mechledger claim propose --run latest
uv run mechledger claim review latest
```

Proposals include expected claim-ledger hashes. Freeform prose edits do not make
a proposal stale; semantic YAML changes do.

## Writing A Decision Record

Manually add an h2 record to `research/logs/decision_log.md`:

````markdown
## D012 - Waive empirical null for scoped diagnostic appendix

```yaml
decision_id: D012
status: accepted
affected_experiments: [E001]
affected_claims: [C003]
decision_type: methodology
copilot_session_id: null
```
````

`mechledger decision new --from-diff` can append a proposed decision based on
declared surfaces. It does not detect Python constants, implicit notebook state,
undeclared config files, or external data changes.

## Waiving Scientific Debt

```bash
uv run mechledger debt waive DPT002 --decision D012
```

The decision must exist and be accepted. Waived debt remains visible in reports.

## Assessment Examples

Baseline calibration metrics:

```python
ml.log_metric("baseline_target_score", 0.8)
ml.log_metric("baseline_foil_score", 0.2)
ml.log_metric("baseline_contrast", 0.6)
```

Positive control:

```python
ml.log_metric("positive_control_pass_rate", 0.95)
```

Empirical null:

```python
ml.log_metric("random_null_seed_count", 30)
ml.log_artifact("runs/e001/null_distribution.jsonl", claim_relevance="required")
```

Paired statistic:

```python
ml.log_metric("paired_test_p_value", 0.01)
ml.log_metric("paired_test_n_pairs", 69)
```

The current CLI records and surfaces debt from available metadata. It does not
compute SciPy/NumPy statistics in the core.

## Large Artifacts

Use DVC, git-annex, Git LFS, or external storage for large tensors and model
outputs. MechLedger records pointers, hashes when local bytes exist, and storage
metadata. It does not implement large-file versioning.

## Composing With Other Linters

Use sciwrite-lint, statcheck, or citation-specific tools as separate
pre-commit hooks. MechLedger does not verify citations, bibliography state,
reported-statistic arithmetic, or scientific truth.

## Deferred Export

RO-Crate export remains future work. Core canonical state remains Markdown,
YAML, CSV, JSON, and JSONL; SQLite is disposable and never merged.
