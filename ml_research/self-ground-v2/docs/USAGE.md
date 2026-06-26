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
uv run mechledger draft suggest research/paper/draft.md --out research/paper/draft_suggestions.md
```

Draft Guard supports `[CLAIM:C003]`, `\claim{C003}`, and
`<!-- CLAIM:C003 -->`. It enforces forbidden phrases, required caveats,
unresolved debt flags, and visible same-paragraph overrides.

`draft suggest` is deterministic reviewer support. It reports Draft Guard
diagnostics, allowed phrases, forbidden phrases, required caveats, and
unresolved debt. It does not rewrite prose with AI and does not suppress hard
Draft Guard violations.

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

## Explainer Prediction Locking

Explainer predictions are pre-intervention JSON records. Lock them before an
intervention run, then score them against registered run outputs:

```bash
uv run mechledger prediction lock research/predictions/sae_12300.json
uv run mechledger prediction score PRED001 --against-run latest
```

Prediction records are JSON objects with these fields:

```json
{
  "prediction_id": "PRED001",
  "feature_id": "sae_12300",
  "source_examples_path": "research/examples/sae_12300.jsonl",
  "prediction_artifact_path": "research/predictions/sae_12300.json",
  "label_source_model": "gpt-4.1",
  "label_prompt_path": "prompts/explainer_label.md",
  "label_generated_at": "2026-06-25T00:00:00Z",
  "short_label": "negation-sensitive direction feature",
  "predicted_target_direction": "increase",
  "predicted_control_direction": "decrease",
  "predicted_relative_magnitude": "target_gt_control",
  "locked_at": null,
  "locked_content_hash": null,
  "scored_against_run_id": null,
  "sign_match": null,
  "relative_magnitude_match": null,
  "tamper_status": "not_locked"
}
```

Supported direction values are `increase`, `decrease`, `no_change`, and
`unknown`. Supported relative-magnitude values are `target_gt_control`,
`target_lte_control`, and `unknown`.

`prediction lock` writes `locked_at`, `locked_content_hash`, and
`tamper_status: locked_valid`. The hash is SHA-256 over canonical semantic
prediction content: JSON key order and formatting do not matter, and mutable
lock/score fields are excluded. Re-locking an unchanged record is idempotent.
If a locked prediction is edited, locking or scoring reports
`modified_after_lock` and exits 1 unless `prediction lock --force` is used as a
visible relock.

`prediction score` discovers prediction files under
`research/predictions/**/*.json` and `predictions/**/*.json`; add
`--prediction-dir PATH` for another directory. Duplicate prediction IDs are
input errors. The run ID is resolved with the normal run alias cache.

Scoring reads only registered run files:

```text
.mechledger/runs/RUN_ID/run.json
.mechledger/runs/RUN_ID/metrics.jsonl
.mechledger/runs/RUN_ID/events.jsonl
```

Required scoring metrics are `target_delta` and `matched_control_delta`.
Accepted variants are `top_target_delta`, `top_matched_control_delta`, and
`top_control_delta`. If `specificity_gap` or `specificity_gap_mean` is present,
it must agree with `target_delta - matched_control_delta`.

Feature matching must come from run evidence, not filenames. The prediction
`feature_id` must appear in `run.json` `feature_id`, metrics row metadata
`feature_id`, events metadata `feature_id`, or events metadata
`features_modified`.

Scoring writes `scored_against_run_id`, `sign_match`,
`relative_magnitude_match`, and keeps `tamper_status: locked_valid` on success.
If a prediction uses `unknown` for a direction or relative magnitude, the
corresponding match field is `null`. MechLedger does not generate labels,
compute model outputs, or execute interventions; it scores registered run
outputs only.

## Export, Bundles, And Appendices

Export deterministic RO-Crate-compatible metadata:

```bash
uv run mechledger export ro-crate --out bundles/ro-crate/
```

This writes `bundles/ro-crate/ro-crate-metadata.json` from canonical flat files,
local run directories, artifact manifests, debt reports, external labels, and
optional platform records. The JSON is deterministic and uses simple
JSON-LD-compatible fields without importing RDF tooling. Missing optional files
are warnings; malformed canonical ledgers fail.

Create a reproducibility bundle:

```bash
uv run mechledger export bundle --out bundles/mechledger_bundle.tar.gz --run latest
uv run mechledger export bundle --out bundles/manifest.json --manifest-only
```

Bundles include canonical files and selected run metadata. Artifact metadata is
always recorded in `manifest.json`; artifact bytes are included only with
`--include-artifacts` and only for registered local paths inside the project.
Environment redaction is recorded and enabled by default. `.mechledger` caches,
SQLite indexes, tmp files, session/copilot scratch records, and unregistered
large files are not swept into bundles. `.tar.zst` requires the local `zstd`
tool; MechLedger does not silently write a different archive format.

Generate a paper-safe appendix:

```bash
uv run mechledger export appendix --out research/paper/mechledger_appendix.md \
  --include-debt --include-decisions --include-artifacts
```

The appendix includes project ID, claim status/scope, linked runs and decisions,
unresolved scientific debt, artifact summaries, and claim language policy. It
does not generate new claims, promote claims, phrase negative evidence as
support, verify citations, or prove scientific truth.

## Session Audit Records

Local session records are opt-in audit trails for human or assistant outputs:

```bash
uv run mechledger session start --title "Review feature label evidence"
uv run mechledger session note --session SESSION_ID --text "Checked C001 wording."
uv run mechledger session attach --session SESSION_ID research/paper/draft.md
uv run mechledger session close --session SESSION_ID
uv run mechledger session review --session SESSION_ID --accept --decision D012
uv run mechledger session list
uv run mechledger session show SESSION_ID
```

Records live under `.mechledger/copilot/SESSION_ID/` and remain local. Notes are
append-only, attachments record path/hash/size when bytes exist, and close writes
JSON plus Markdown summary. Accepting a session requires an accepted decision;
rejected sessions remain visible. Session records are not canonical evidence and
do not auto-edit claim, research, or decision logs.

## Open Questions

Existing `research_log.md` `open_questions` entries are surfaced by:

```bash
uv run mechledger questions list
uv run mechledger questions add --text "Need another control family?" --claim C003 --experiment E004 --priority high
uv run mechledger questions show Q001
uv run mechledger questions resolve Q001 --decision D012 --resolution "Accepted added control requirement."
uv run mechledger next
```

New questions are stored in `research/logs/open_questions.md`. Resolving a
question requires an accepted decision. `next` prints open questions linked to
claims or experiments, but questions do not become blockers unless another
configured prerequisite/debt policy already makes the work gated.

## External Label Registry

Import external labels as metadata:

```bash
uv run mechledger labels validate labels.jsonl
uv run mechledger labels import labels.jsonl
uv run mechledger labels list
uv run mechledger labels show L001
uv run mechledger labels link L001 --claim C003
```

The canonical registry is `research/literature/external_labels.jsonl`. Records
include source attribution, source URL/model, label text, feature ID, model/hook
metadata, confidence/license/notes, linked claims, and a semantic hash. External
labels are not causal or mechanistic evidence by default; linked claims still
follow their claim status, evidence, caveat, and debt policy.

## Local Dashboard Data And Queries

Generate deterministic JSON suitable for a future dashboard:

```bash
uv run mechledger dashboard data --out .mechledger/dashboard/data.json
```

Inspect canonical records without a server:

```bash
uv run mechledger query claims --json --status candidate_claim
uv run mechledger query runs --json --experiment E001
uv run mechledger query debt --json --severity serious
uv run mechledger query artifacts --json --run RUN_E001
uv run mechledger query decisions --json
uv run mechledger query experiments --json
```

Query commands read flat files as source of truth. SQLite remains disposable
cache state and is not required.

## Claim Language Reports

```bash
uv run mechledger claim language-report --claim C003
uv run mechledger claim language-report --all --out research/paper/claim_language_report.md
```

Language reports use claim YAML `status`, `allowed`, `forbidden`,
`required_caveats`, and `debt_flags`. They are deterministic checklists, not LLM
writing, and do not claim semantic correctness.

## Optional Platform Records

Optional future-work metadata can be validated without adding ML dependencies:

```bash
uv run mechledger records validate research/records/activation_REC001.json
uv run mechledger records list
uv run mechledger records show REC001
```

Supported record types are `ActivationRecord`, `CircuitGraphRecord`,
`WeightAnalysisRecord`, `CrossModelComparisonRecord`,
`FeatureCorrespondenceRecord`, `TrainingDynamicsRecord`, and
`RemoteJobMetadataRecord`. Validation checks structure, IDs, source paths,
linked runs/claims/decisions, and artifact pointers only; MechLedger does not
compute activations, circuits, weights, correspondences, training dynamics, or
remote jobs.

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

## Storage And Export Boundaries

Core canonical state remains Markdown, YAML, CSV, JSON, and JSONL; SQLite is
disposable and never merged. RO-Crate JSON is an export target, not canonical
storage. MechLedger does not use RDF/JSON-LD as the internal data model, does
not run a dashboard server, does not sync/merge remote state, and does not
discover hidden methods from notebooks or Python constants.

MechLedger also does not execute interventions, generate labels, enforce
untagged claims, verify citations, recompute reported statistics, make
scientific truth decisions, or discover arbitrary artifacts outside registered
paths/run-local artifact directories. It may allow progress with unresolved
scientific debt, but it surfaces that debt. External labels are metadata by
default. Session/copilot records require human review and accepted decision
linkage before they can support canonical interpretation.
