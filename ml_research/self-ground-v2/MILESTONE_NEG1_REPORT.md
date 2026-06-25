# Milestone -1 Report

## Scope

This pass implements PRD §50.0 only: extraction and dogfooding of the
MechLedger audit kernel. It deliberately does not implement a CLI entrypoint,
run wrapper, alias cache, SQLite index, pre-commit hooks, dashboard, or copilot
surface.

## Resolved Up Front

- Repo placement: `self-ground-v2/` is the standalone MechLedger package.
  SELF-GROUND remains read-only. Original SELF-GROUND work logs were copied to
  `research/logs/source/`.
- Dependencies: core dependencies are `typer`, `pydantic`, and `ruamel.yaml`.
  There is no console script and no `typer.Typer()` application in this pass.
- YAML policy: claim parsing uses `ruamel.yaml` round-trip mode; pydantic
  validates parsed data.
- Severity split: `DebtSeverity` and `DraftSeverity` are separate enums.
  `DebtSeverity` includes `serious`; `DraftSeverity` does not.
- Threshold-default noise: `ScientificDebtReport.tool_default_rollup()`
  aggregates `unjustified_threshold_default` info debt by count and assessment
  ID for human-facing summaries while preserving full JSON records.
- Reuse decision: `research/logs/decision_log.md` entry `D001` consolidates
  the two reuse assessments and names the exact modules/boundaries.

## Deferred Defect

PRD §31.4 lists `failed -> serious_evidence_run` and
`cancelled -> serious_evidence_run` as disallowed run-class transitions, but
`failed` and `cancelled` are `Run.status` values, not `Run.run_class` values.
This is logged here for Milestone 2. Likely intent: failed or cancelled runs
cannot be reclassified regardless of run class, but the spec owner should
confirm before implementation.

## Extracted Logic

- `core/claim_status.py`: PRD §11 support DAG, terminal status behavior,
  SELF-GROUND status mapping, and conservative status seed.
- `core/debt.py`: PRD §12 `ScientificDebtRecord` and report filtered views.
- `core/claim_ledger.py`: PRD §10 heading/YAML parser and exact canonical hash.
- `assessments/calibration.py`: pure dict version of task calibration and
  baseline/positive-control condition logic.
- `assessments/compatibility.py`: pure shape and metadata compatibility logic.
- `assessments/telemetry.py`: framework-free telemetry warning/debt logic.
- `draftguard_proto.py`: Markdown claim tags, sentence window, and forbidden
  phrase matching only.

Architecture guard tests verify there are no imports of `torch`,
`transformer_lens`, `sae_lens`, `numpy`, or `scipy` in core/assessment modules.

## SELF-GROUND Status Mapping

SELF-GROUND has four mechanism-report statuses:

- `blocked` -> `unsupported`
- `insufficient_evidence` -> `failed_or_weakened`
- `candidate_evidence` -> `candidate_claim`
- `strong_candidate_evidence` -> `causal_support`

The main divergence is `insufficient_evidence`. In SELF-GROUND E002-E004 this
usually means a real run weakened the target feature-specific claim under
controls, not merely that no evidence exists. The backfilled claim ledger maps
those records to `failed_or_weakened` and records the specific matched-control
or multi-control failure.

The conservative status test encodes the real E002, E003, and E004 outcomes:
all remain `insufficient_evidence` under SELF-GROUND logic.

## Dogfood Corpus

Known-good real sentence, correctly passed:

```text
The calibrated E003 task bank fixes the baseline task-suite coverage blocker
for Pythia-70M-deduped, including `property_negation`. The selected SAE feature
set moves logits under real decoded intervention, but the effect is not
negation-specific under the current matched-control evaluation. [CLAIM:C003]
```

Known-bad real SELF-GROUND anti-example phrase, correctly flagged when tagged:

```text
If you catch yourself writing "this is the negation mechanism," stop and write
instead. [CLAIM:C004]
```

False positive observed:

```text
We do not claim broad negation mechanism discovery, upstream SAEBench/RAVEL
benchmark evidence, model-general conclusions, or monosemantic feature claims.
[CLAIM:C004]
```

Reason: the prototype is deterministic phrase matching and does not understand
negation. Milestone 0 should either keep this as a visible override case or add
a narrow deterministic negation-aware heuristic. It must not use LLM output for
blocking enforcement.

False negatives observed in the tagged dogfood corpus: none. Residual risk:
the prototype only catches configured multi-word phrases, so unlisted overclaim
paraphrases such as "establishes a complete causal role" will pass until the
claim ledger's forbidden list is expanded.

## Backfilled Claims

`research/logs/claim_ledger.md` contains PRD §10 records for:

- `C001`: Phase 1-2 real model/residual/decoded-SAE path execution.
- `C002`: E002 matched controls moved more than target prompts.
- `C003`: E003 fixed baseline coverage but not feature specificity.
- `C004`: E004 rescue matrix produced no candidate cells.

These were derived from `self-ground/work_logs/claim_ledger.md`,
`self-ground/work_logs/run_ledger.csv`, SELF-GROUND mechanism reports, and E004
adjudication/forensics artifacts. The source logs are copied under
`research/logs/source/`.

## QC

- `uv run pytest`
- `uv run ruff check .`

Milestone 0 should start only after reviewing this report and deciding how to
handle the deterministic false-positive class above.
