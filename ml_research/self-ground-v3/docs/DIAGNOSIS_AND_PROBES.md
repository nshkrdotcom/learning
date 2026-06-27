# Diagnosis Trees And Materialized Probes

Diagnosis and probe materialization turn a blocked run into an auditable next experiment without upgrading the claim language.

## Commands

Generate a diagnosis tree for the latest run:

```bash
uv run mwb diagnose latest
```

Generate the next-probe plan and materialize a runnable probe when the recommendation has an implemented runner:

```bash
uv run mwb next-probe latest --materialize
```

Execute a materialized probe through the local workflow runner:

```bash
uv run mwb run-probe .mechanism/runs/<run_ref>/probe.yaml
```

`run-probe` does not shell out to the command text in `probe.yaml`. It validates the probe kind, parses its structured parameters, and calls the existing workbench workflow implementation. Unsupported probe kinds exit non-zero and do not emit runnable commands.

## Run-Local Artifacts

`mwb diagnose <run>` writes:

- `diagnosis_tree.json`
- `diagnosis_tree.yaml`
- `diagnosis_tree.md`

`mwb next-probe <run> --materialize` writes or refreshes:

- `next_probe.json`
- `next_probe.yaml`
- `next_probe.md`
- `diagnosis_tree.json`
- `diagnosis_tree.yaml`
- `diagnosis_tree.md`
- `probe.json`
- `probe.yaml`
- `probe.md`

`probe.yaml` contains:

- `source_run_ref`
- `next_probe_ref`
- `diagnosis_tree_ref`
- `template_id`
- `probe_kind`
- `runnable`
- structured `parameters`
- a command preview for implemented probes
- provenance rows with artifact names and JSON paths

## Diagnosis Semantics

The diagnosis tree is built from run-local evidence:

- `run_manifest.json` for run identity, status, source hypothesis, tried axes, available axes, and backend capabilities.
- `control_metrics.json` for measured target/control behavior.
- `blocker_report.json` for blockers, primary blocker, and blocking metrics.
- `scientific_debt.json` for unresolved negative evidence and claim-blocking debt.

The tree keeps negative evidence first-class. Unresolved scientific debt and failed blocking metrics remain visible in `negative_evidence`; they are not folded into a passing summary.

## Implemented Probe Kinds

The current implemented probe runners are:

- `sweep_axis_extension`: materialized from `smallest_axis_extension`.
- `switch_patch_mode`: materialized from `switch_patch_mode`.

Both run as dry-run sweep workflows and write a concrete non-claim-bearing run under `.mechanism/runs/<run_ref>/`.

Recommendations without an implemented runner are still recorded as blocked materialized probes. They include a `template_id` such as `unsupported.heldout_generalization.v1`, `runnable: false`, and no command.

## SQLite Recovery

`mwb repair-index` and `mwb rebuild-index` restore:

- `diagnosis_trees` from `diagnosis_tree.json`
- `materialized_probes` from `probe.json`

This keeps diagnosis and probe decisions recoverable from Git-visible code plus file-backed `.mechanism` artifacts.
