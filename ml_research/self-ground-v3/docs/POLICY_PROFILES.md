# Policy Profiles

Policy profiles make research-taste rules explicit and reproducible. They do not block scratch exploration; they apply to claim-bearing verification, MechanismCards, claim grammar, and Draft Guard.

## Commands

Evaluate the project default profile:

```bash
uv run mwb policy check
```

Evaluate a named profile:

```bash
uv run mwb policy check --profile strict
uv run mwb policy check --profile exploratory
```

Reports are written to `.mechanism/policy/latest_policy_evaluation.json` and indexed in SQLite as `policy_evaluations`.

## Project Config

New projects default to:

```toml
[policy]
profile = "strict"
```

Older projects without this section also resolve to `strict`.

## Built-In Profiles

`strict` encodes the default lab-standard policy:

- zero ablation cannot support claim-bearing evidence beyond `diagnostic_only`;
- claim-bearing candidate verification requires resample ablation;
- claim-bearing candidate verification requires both noising and denoising;
- static space/type checks are expected;
- alternative explanations are expected;
- mechanism wording requires heldout/generalization evidence;
- unresolved scientific debt blocks stronger claims.

`exploratory` relaxes those gates for local exploration:

- zero ablation can remain visible as weaker evidence;
- noising/denoising are not required as a pair;
- mechanism wording does not require the policy-level generalization gate;
- unresolved scientific debt is surfaced as caveat/debt instead of a hard policy block.

## Integration Points

Verification:

- `CausalVerificationService` evaluates the selected profile for claim-bearing runs.
- Policy blockers are attached to `VerificationRun.metadata["blockers"]`.
- Policy reports are embedded in run manifests as `policy_report`.

MechanismCards:

- Card generation applies the policy claim ceiling from `policy_report`.
- Policy blockers are preserved in card metadata and draft-visible claim records.

Claim Grammar and Draft Guard:

- Claim grammar reads `policy_profile` from the claim payload or MechanismCard.
- Draft Guard runs claim grammar before the legacy phrase fallback.
- Inline overrides remain visible but do not upgrade blocked policy outcomes.
