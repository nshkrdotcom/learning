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

For paired delta JSONL files, the SDK also provides a dependency-light sign
test helper:

```python
result = ml.stats.compute_paired_test(
    "results/per_task_results.jsonl",
    paired_by="task_id",
    metric="specificity_gap",
    test="sign",
)

ml.stats.write_paired_test_result(result, "results/paired_test.json")
```

The SDK helper does not compute Wilcoxon or permutation tests. Register those
from the research environment that produced them.

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

## Evidence Gate Check

```bash
uv run mechledger gate check latest
```

`gate check` resolves the run alias, reads `run.json`, `metrics.jsonl`,
`artifacts.jsonl`, `artifact_manifest.json`, and existing debt reports, then
writes:

```text
.mechledger/runs/RUN_ID/evidence_assessment.json
.mechledger/runs/RUN_ID/evidence_assessment.md
.mechledger/runs/RUN_ID/scientific_debt_report.json
.mechledger/runs/RUN_ID/scientific_debt_report.md
```

The assessment is deterministic policy logic over registered metadata. Core
does not compute p-values, empirical-null percentiles, or other statistics, and
it does not import heavy ML libraries. Register those values from your own
research environment through the SDK, run-local JSONL files, or artifact
registration.

Clean candidate support requires an allowed run class plus passing or accepted
waiver coverage for baseline calibration, positive control, empirical-null seed
count, paired statistic metadata, matched controls/specificity, compatibility
when present, and telemetry. Missing or failed required evidence emits visible
scientific debt and prevents clean candidate support unless waived by an
accepted decision record.

Useful metric names include:

```text
random_null_seed_count
null_distribution_path
percentile_rank
paired_test_name
paired_by
paired_test_n_pairs
paired_test_p_value
effect_direction
sign_consistency
target_delta
matched_control_delta
specificity_gap
top_control_ratio
multi_control_min_gap
family_min_gap
relative_norm_drift
nonfinite_rate
skip_rate
```

## Tier 2 Convenience And Register Commands

Calibration and telemetry checks are filtered wrappers over the same assessment
policy used by `gate check`:

```bash
uv run mechledger calibration check latest
uv run mechledger telemetry check latest
```

They write:

```text
.mechledger/runs/RUN_ID/calibration_check.json
.mechledger/runs/RUN_ID/calibration_check.md
.mechledger/runs/RUN_ID/telemetry_check.json
.mechledger/runs/RUN_ID/telemetry_check.md
```

Both reports include condition status, debt/resolution text, and
`threshold_source` for evaluated threshold conditions. Calibration exits 1 only
for blocking calibration or positive-control findings. Telemetry exits 1 only
for telemetry blockers such as non-finite rows or all rows skipped.

Create an empirical-null plan:

```bash
uv run mechledger null run --plan \
  --experiment E001 \
  --feature-set-size 20 \
  --seeds 30 \
  --sampling density_matched
```

This writes `research/experiments/E001_null_plan.yaml` with
`experiment_id`, `feature_set_size`, `seed_count`, `sampling_method`,
`exclude_feature_ids`, `output_metric`, and `planned_output_artifact`. Existing
plans are refused unless `--force` is supplied.

Register researcher-produced empirical-null output:

```bash
uv run mechledger null run --register latest \
  --null-distribution results/null_distribution.jsonl \
  --metric specificity_gap_mean \
  --seed-count 30 \
  --percentile-rank 0.99
```

The command verifies the file exists, attaches it as required evidence, appends
`random_null_seed_count`, `null_distribution_path`, `null_metric`, and optional
`percentile_rank` to `metrics.jsonl`, writes `null_check.json/md`, and
regenerates the scientific debt report. It refuses duplicate null artifact
registration unless `--force` is supplied. It does not compute a percentile
rank unless one is explicitly provided by the researcher.

Register a paired-test result:

```bash
uv run mechledger stats paired-test latest --register results/paired_test.json
```

The JSON must contain the documented paired-test fields including `run_id`,
`paired_by`, `metric`, `test`, `n_pairs`, `p_value`, `effect_direction`,
`sign_consistency`, threshold metadata, and input/output artifact paths.
Supported test labels are `sign`, `wilcoxon`, `permutation`, and
`custom_registered`; the core CLI records and evaluates metadata only. The
command writes `.mechledger/runs/RUN_ID/paired_test.json`, writes a Markdown
summary, appends the paired-test metrics consumed by the policy evaluator,
registers the JSON as required evidence, and regenerates the scientific debt
report. Existing paired-test registration is refused unless `--force` is
supplied.

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

## Reclassifying A Run

```bash
uv run mechledger run reclassify latest \
  --to serious_evidence_run \
  --decision D012 \
  --reason "human-reviewed transition after artifact review"
```

The decision must exist in `research/logs/decision_log.md` with
`status: accepted`. Reclassification updates only the local run directory:
`run.json`, `events.jsonl`, `run_class_transition.json`, and the generated
scientific-debt report. It does not edit `research/logs/run_ledger.csv`.

Supported classes are `scratch`, `notebook_exploration`, `diagnostic`,
`serious_evidence_run`, `paper_candidate`, `replication`, and
`published_result`.

## Experiment Prerequisites And Next

ExperimentSpecs can declare machine-readable prerequisites in YAML:

```yaml
prerequisites:
  - type: decision_accepted
    id: D012
  - type: claim_status_at_least
    id: C003
    status: single_run_evidence
  - type: artifact_exists
    path: results/e003/baseline.json
    consequence: scientific_debt
```

Supported prerequisite types are `decision_accepted`, `experiment_completed`,
`experiment_completed_and_reviewed`, `claim_status_at_least`, and
`artifact_exists`. Consequence defaults to `blocking`; `scientific_debt` and
`warning` stay visible but do not make the experiment a clean pass.

```bash
uv run mechledger experiment validate research/experiments/E003_*.md
uv run mechledger next
```

`next` uses the same prerequisite engine and separates `READY`, `BLOCKED`, and
`DEBT/WARNING GATED` experiments. Claim status comparisons use the claim-status
DAG; incomparable statuses are input errors.

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
a proposal stale; semantic YAML changes do. `claim review --apply` refuses stale
proposals unless explicitly run with `--force-stale --yes`, which records the
forced stale review in the proposal and still does not silently mutate the claim
ledger.

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

`mechledger decision new --from-diff` appends a proposed decision from changed
research files. `mechledger decision new --from-declared-surfaces` appends a
proposed decision from machine-readable declared surfaces such as ExperimentSpec
`config_files`, `expected_artifacts`, claim-linked runs, and the configured
research log. It explicitly documents implicit surfaces it refused to infer,
including Python constants, notebook state, undeclared config files, and
unregistered external data changes.

## Waiving Scientific Debt

```bash
uv run mechledger debt waive DPT002 --decision D012
```

The decision must exist and be accepted. Waived debt remains visible in reports.
Missing debt, missing decisions, and proposed/rejected decisions fail as unsafe
input errors.

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
compute SciPy/NumPy statistics in the core. `gate check` assesses registered
values and artifacts and reports whether they meet the default evidence policy.

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
