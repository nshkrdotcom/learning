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
  portfolio/
    WRITEUP_PLAN.md              <- separate plan for the LW/AF-style write-up (not the academic draft)
  logs/
    research_log.md              <- dated entries: question -> work -> result -> decision
    decision_log.md               <- why the project changed direction, with alternatives considered
    run_ledger.csv                 <- one row per real run, audit trail
    claim_ledger.md                <- the spine of the paper: every claim + its current evidence status
  literature/
    prior_art_matrix.md           <- structured comparison table, becomes Related Work
  experiments/
    E000_baseline_behavioral_calibration.md     <- hard prerequisite, run before any Phase 3 feature code
    E001_phase3_token_contrast_evaluation.md    <- feature-set validity evaluation (the original Phase 3 spec)
    E002_explainer_prediction_baseline.md       <- prediction-vs-intervention scoring (closes the "self" gap)
```

## The one rule that matters

Every time you're tempted to write "this feature represents negation" or
"this is the negation mechanism," stop and write instead:

> This feature set has evidence consistent with influencing negation-sensitive
> token contrasts under this model/hook/SAE configuration.

Then ask: what evidence would falsify that claim? If you can't answer, the
claim isn't ready for `claim_ledger.md` yet.

## Current state in one paragraph (as of 2026-06-21)

Phase 1 (real residual-dimension ranking + intervention) and Phase 2 (real
decoded SAE feature intervention) are implemented and have real, verified
evidence behind them — see `logs/claim_ledger.md` for exact status and
`logs/run_ledger.csv` for the runs that produced it. Phase 3 (multi-task
token-contrast evaluation) has a complete implementation spec but has not
been built or run yet, and that spec has been revised in three ways before
any of it gets implemented (see decisions D007-D012 in `decision_log.md`):
(1) a standalone behavioral-calibration gate (`E000`) now runs first,
because small models are documented in the literature to handle negation
unreliably and that needs checking before more SAE infrastructure gets
built on top of it; (2) the fixed 3-seed random baseline is replaced with a
real empirical null (~30-50 seeds) and percentile-rank significance,
because a point-estimate ratio over 3 seeds won't hold up to scrutiny from
this field; (3) a new experiment (`E002`) adds the one thing missing from
the original Phase 3 plan that the project is actually named for — scoring
a pre-intervention prediction against the real intervention outcome, not
just comparing feature-selection methods. Nothing in this repo currently
supports a claim stronger than "single-run evidence that the real
decoded-SAE-intervention path works." That's the honest starting line for
both the paper and the portfolio write-up.

## Workflow going forward

1. Run `E000` first. Do not start `E001`'s feature-selection code until
   E000's result is recorded in `logs/claim_ledger.md`, whatever it shows.
2. Before writing code for a phase: fill out (or update) the matching
   `experiments/E0XX_*.md` file. What would falsify the hypothesis?
3. While running real commands: append a row to `logs/run_ledger.csv`
   immediately, not from memory later.
4. After a run produces evidence: update `logs/claim_ledger.md` *before*
   touching `paper/draft.md`. The paper should only ever say what the ledger
   already supports.
5. When you change the plan (a metric, a control, a threshold): write one
   entry in `logs/decision_log.md` with the reason and the rejected
   alternatives. This is what saves you when a reviewer asks "why this
   control and not that one?"
6. Daily or per-session: one entry in `logs/research_log.md`. Five fields,
   short. The point is reconstructability, not prose.
7. Once E000-E002 have real results: start filling in
   `portfolio/WRITEUP_PLAN.md` — that document, not the academic draft, is
   the one most likely to actually get this project noticed.
