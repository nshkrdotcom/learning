# SELF-GROUND Research Operating System

This folder is the research record for SELF-GROUND. Drop it into the repo root
(e.g. as `research/`) and start updating it alongside code changes. It is not
a separate bureaucratic exercise — every file here should get touched in the
same sitting as the code that produced the evidence it describes.

## What's in here

```text
research/
  README.md                      <- this file
  paper/
    draft.md                     <- living paper draft (outline + working abstract + claim ladder)
    abstract_drafts.md           <- alternate abstract phrasings to pick from later
  logs/
    research_log.md              <- dated entries: question -> work -> result -> decision
    decision_log.md              <- why the project changed direction, with alternatives considered
    run_ledger.csv                <- one row per real run, audit trail
    claim_ledger.md               <- the spine of the paper: every claim + its current evidence status
  literature/
    prior_art_matrix.md           <- structured comparison table, becomes Related Work
  experiments/
    E001_phase3_token_contrast_evaluation.md   <- next experiment, written before code
```

## The one rule that matters

Every time you're tempted to write "this feature represents negation" or
"this is the negation mechanism," stop and write instead:

> This feature set has evidence consistent with influencing negation-sensitive
> token contrasts under this model/hook/SAE configuration.

Then ask: what evidence would falsify that claim? If you can't answer, the
claim isn't ready for `claim_ledger.md` yet.

## Current state in one paragraph (as of 2026-06-20)

Phase 1 (real residual-dimension ranking + intervention) and Phase 2 (real
decoded SAE feature intervention) are implemented and have real, verified
evidence behind them — see `logs/claim_ledger.md` for exact status and
`logs/run_ledger.csv` for the runs that produced it. Phase 3 (multi-task
token-contrast evaluation with baseline feature-set comparisons and a
thresholded mechanism-evidence report) has a complete implementation spec
but has not been built or run yet. `experiments/E001...md` is that spec
compressed into the experiment-registry format. Nothing in this repo
currently supports a claim stronger than "single-run evidence that the
real decoded-SAE-intervention path works." That's the honest starting line
for the paper.

## Workflow going forward

1. Before writing code for a phase: fill out (or update) the matching
   `experiments/E0XX_*.md` file. What would falsify the hypothesis?
2. While running real commands: append a row to `logs/run_ledger.csv`
   immediately, not from memory later.
3. After a run produces evidence: update `logs/claim_ledger.md` *before*
   touching `paper/draft.md`. The paper should only ever say what the ledger
   already supports.
4. When you change the plan (a metric, a control, a threshold): write one
   entry in `logs/decision_log.md` with the reason and the rejected
   alternatives. This is what saves you when a reviewer asks "why this
   control and not that one?"
5. Daily or per-session: one entry in `logs/research_log.md`. Five fields,
   short. The point is reconstructability, not prose.
