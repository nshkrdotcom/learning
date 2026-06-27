# Hypothesis Lifecycle

Hypothesis lifecycle state is separate from evidence tier and claim status.

The workflow state answers where the investigation is in the research process. Evidence tier answers what kind of support exists. Claim status answers what paper-facing ledger state, if any, has been approved.

## Transition

```bash
uv run mwb hypothesis transition <hypothesis-ref> --to-state triaged
```

Optional independent fields:

```bash
uv run mwb hypothesis transition <hypothesis-ref> \
  --to-state structurally_plausible \
  --evidence-tier projection \
  --claim-status single_run_evidence \
  --reason "static projection passed"
```

Promotion to `claimable` requires explicit human approval:

```bash
uv run mwb hypothesis transition <hypothesis-ref> \
  --to-state claimable \
  --approved-by local_user \
  --decision-ref D001
```

The command writes:

- `.mechanism/hypotheses/<hypothesis-ref>_lifecycle.json`
- `.mechanism/hypotheses/<hypothesis-ref>_transitions.jsonl`

SQLite indexes the current state in `hypothesis_states` and receipts in `hypothesis_transitions`.

## States

Forward workflow states:

- `noticed`
- `triaged`
- `structurally_plausible`
- `cheap_proxy_supported`
- `exact_patch_supported`
- `control_clean`
- `generalized`
- `claimable`

Terminal or kill states:

- `structurally_impossible`
- `proxy_false_positive`
- `control_leaky`
- `self_repair_confounded`
- `off_manifold`
- `task_artifact`
- `dictionary_artifact`
- `abandoned`

Forward transitions must happen in order. A live state may move into a terminal state when the evidence kills or retires the hypothesis.

## Alternative Explanations

Generate live alternatives from a run's blocker report:

```bash
uv run mwb hypothesis explain <run-ref>
```

The command writes:

- `.mechanism/runs/<run-ref>/alternative_explanations.json`
- `.mechanism/hypotheses/<hypothesis-ref>_alternatives.json`

Current blocker mappings include:

- `control_leaky` -> target/control semantic separation or control contamination.
- `density_matching_failed` -> generic high-activity feature.
- `dictionary_interference` and `neighbor_feature_interference` -> decoder-neighbor interference.
- `self_repair_suspected` -> downstream self-repair.
- `off_manifold_intervention` -> off-manifold intervention.
- `specificity_gap_failed` -> task artifact.

Each alternative records evidence for, evidence against, and the next discriminating test.

## Recovery

`mwb rebuild-index` and `mwb repair-index` restore lifecycle state, transition receipts, and alternative explanations from file-backed records.

## Claim Boundary

Lifecycle commands do not create paper claims. `claimable` means the hypothesis is eligible for human-reviewed claim proposal work, not that the claim is true or already accepted.
