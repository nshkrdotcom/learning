# Decision Log

Records *why* the project changed shape, not just what changed. This is the
file that answers a reviewer who asks "why this control and not that one?"
or "why this threshold?" Write the entry when you make the decision, not
when you remember you made one.

---

## D001 — Use a pretrained SAE rather than train one

**Decision:** SELF-GROUND uses an existing SAELens pretrained SAE
(`pythia-70m-deduped-res-sm`) rather than training a new sparse autoencoder.

**Reason:** Training an SAE adds a full sub-project (sparsity tuning,
dead-feature monitoring, dataset selection) before any causal-evidence
question can even be asked. A widely-used pretrained SAE gets to the
causal-intervention question fastest.

**Alternatives considered:**
- Train a small custom SAE on Pythia-70m activations — rejected for this
  pass; revisit only if no pretrained SAE proves compatible.

**Consequences:** All current causal evidence is tied to one specific
checkpoint (`EleutherAI/pythia-70m-deduped`) and one specific SAE release.
Generalization claims must be scoped to that pair until a second compatible
SAE/model pair is verified.

---

## D002 — Fail closed on SAE/model metadata mismatch; shape compatibility is not enough

**Decision:** The compatibility checker (`sae_compat.py`) requires the SAE's
declared model and hook point to match the requested model and hook point.
A same-width SAE trained on a *different* checkpoint must be rejected even
though its tensor shapes line up.

**Reason:** `EleutherAI/pythia-70m` and `EleutherAI/pythia-70m-deduped` have
identical residual width but are different trained models. A shape-only
check would silently accept a semantically meaningless SAE/model pairing.

**Evidence this was the right call:** the intentional-mismatch check
(`pythia-70m` against the deduped-trained SAE) correctly returned
`metadata_compatible=false`, `compatible=false` (see `run_ledger.csv`,
`check_sae_compatibility_expected_mismatch`).

**Consequences:** Any future SAE candidate needs real, checkable metadata
(declared model + hook point) or must be run only under an explicit
`--allow-metadata-mismatch` diagnostic flag whose output can never reach
`candidate_evidence` or `strong_candidate_evidence` status.

---

## D003 — Proxy feature-space arithmetic is never allowed to look like causal evidence

**Decision:** Feature-space scoring that doesn't go through a real
encode→modify→decode→patch→measure loop is kept in an explicitly named
`FeatureProxyEffect`-style schema, separate from real intervention artifacts.
The earlier non-proxy `FeatureEffect` schema (with causal-sounding field
names like `necessity`/`sufficiency` attached to non-causal numbers) was
retired.

**Reason:** Field names leak claims. A `necessity` field on a
non-interventional score will eventually get cited as if it were
interventional, even by the person who wrote it.

**Alternatives considered:**
- Keep `FeatureEffect` but rename fields — rejected; cleaner to remove the
  schema entirely once nothing in `src/` depended on it.

**Consequences:** Necessity/sufficiency-style language is now reserved for
metrics computed from real activation-patching deltas.

---

## D004 — `delta` patch mode as the default over `replace`

**Decision:** `run_real_sae_intervention` and the Phase 2/3 intervention
primitives default to `patch_mode="delta"` (apply `decoded_modified -
decoded_original` as an additive correction) rather than `patch_mode=
"replace"` (overwrite the residual activation with the decoded
reconstruction outright).

**Reason:** SAE reconstructions are lossy (see `reconstruction_l2_relative
≈ 0.087` for the verified SAE). `replace` would inject that reconstruction
error into every patched run regardless of which feature was targeted,
confounding the feature-specific effect with generic reconstruction noise.
`delta` isolates the contribution of the *selected* features.

**Consequences:** Results under `replace` and `delta` are not directly
comparable; experiment configs must record which mode was used, and
cross-mode comparisons need their own justification if ever attempted.

---

## D005 — Phase 3 controls are matched non-negation prompts, not unrelated collateral prompts

**Decision:** Phase 3 task design uses `control_type="matched_non_negation"`
— a prompt with the same surface structure as the target but without a
negation condition — rather than a generic unrelated "does anything break"
collateral probe.

**Reason:** The immediate hypothesis under test is negation-*specificity*,
not general off-target drift. A matched control isolates exactly the
variable (negation) the claim is about. Unrelated collateral-drift probes
are a reasonable later addition but would muddy the first specificity claim.

**Consequences:** `specificity_gap` and `collateral_ratio` in the Phase 3
report are defined relative to the matched control, not a general drift
baseline. Any future "does this feature set break unrelated behavior"
question needs a new task family and a new metric — don't retrofit this one.

---

## D006 — Strong-candidate-evidence status has a hard floor that a smoke run cannot reach

**Decision:** The Phase 3 mechanism report can only reach
`strong_candidate_evidence` with ≥30 valid tasks, ≥3 task families, ≥3
random baseline seeds, both ablation and amplification tested, and collateral
ratio / norm drift under the strong thresholds. A `per_family=1` or
`per_family=2` verification run is structurally incapable of reaching that
status, no matter how clean its numbers look.

**Reason:** Without this floor, a convenient small run with one favorable
feature pair (e.g. the existing `sae_12300 + sae_25521` Phase 2 evidence)
could get summarized in the paper as stronger than it is. The floor is
mechanical, not a judgment call made after seeing the numbers.

**Consequences:** The existing Phase 2 single-run evidence (Claim 2.3) can
never be upgraded to a strong claim on its own — it can only ever support
`single_run_evidence`, by construction.

---

## Template for new entries

```text
## D0XX — <short title>

**Decision:**
**Reason:**
**Alternatives considered:**
**Evidence (if any):**
**Consequences:**
```
