# Claim Grammar

Claim grammar maps draft language to typed evidence requirements. It sits in front of the older phrase fallback in Draft Guard, so overclaims can be blocked even when a forbidden phrase list is incomplete.

## Commands

Check one claim fixture:

```bash
uv run mwb claim check docs/fixtures/claim_association.json
```

Check a draft with claim tags:

```bash
uv run mwb draft-check docs/fixture_draft.md
```

`mwb claim check` writes `.mechanism/claims/<claim_ref>_grammar_report.json` and indexes the report in SQLite as `claim_grammar_reports`.

## Claim Types

Supported claim types are:

- `association`
- `projection`
- `causal_necessity`
- `causal_sufficiency`
- `mediation`
- `generalization`
- `mechanism`

The deterministic intent matcher infers claim type from the sentence unless `claim_type` is provided in the input JSON.

## Evidence Requirements

Each claim type has required evidence:

- `association`: association evidence.
- `projection`: association plus static projection or path algebra.
- `causal_necessity`: projection, causal necessity, specificity controls, and clean telemetry.
- `causal_sufficiency`: causal necessity plus sufficiency or restoration evidence.
- `mediation`: causal necessity plus mediation/path evidence.
- `generalization`: causal necessity plus heldout generalization.
- `mechanism`: necessity, sufficiency, controls, clean telemetry, resolved alternatives, and generalization minimum.

Blockers such as `metadata_mismatch`, `backend_untrusted`, `control_leaky`, `off_manifold_intervention`, `dictionary_interference`, `self_repair_suspected`, and `insufficient_heldout_generalization` can block or caveat claim types even when the evidence tier is otherwise high enough.

## Statuses

Claim grammar reports return:

- `allowed`: evidence and blockers support the requested claim type.
- `caveated`: the claim can be made only with required caveats.
- `blocked`: required evidence is missing, blockers apply, or unresolved debt blocks the requested claim type.

Overrides are visible but non-upgrading. An inline override is recorded in the report with `visible: true` and `applied: false`; it cannot turn a blocked claim into an allowed claim.

## Draft Guard

Draft Guard resolves each `[CLAIM:<ref>]` tag to a MechanismCard/claim card and runs claim grammar first. If grammar allows the sentence, Draft Guard still applies the older phrase fallback over `blocked_language`, `allowed_language`, and `required_caveats`.

This makes deterministic claim grammar the primary paper-facing guard while preserving the earlier MechanismCard language checks.
