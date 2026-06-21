# D007: SELF-GROUND Is Not an Intervention Engine

Status: accepted

## Decision

SELF-GROUND will not maintain a generic model-intervention engine. The project
is a negation-scope feature-claim evaluation protocol and artifact ledger on top
of existing engines.

Production intervention paths must name their upstream engines:

- local model execution and activation patching: TransformerLens
- SAE load, encode, and decode: SAELens
- RAVEL-style cause/isolation evaluation: SAEBench/RAVEL where feasible, with a
  SELF-GROUND negation adapter only for the negation-specific delta
- remote large-model execution: nnsight/NDIF
- serializable or trainable intervention abstractions: pyvene when needed

## Consequences

- SELF-GROUND-owned residual-dimension outputs are diagnostic smoke patches only.
- Feature-space proxy outputs are legacy-only and cannot enter the claim ledger.
- A decoded SAE claim must pass semantic SAE compatibility before intervention
  rows can be written.
- Mechanism reports must record the external backend used.
- No output may reach `candidate_evidence` or `strong_candidate_evidence` from a
  proxy-only run, a residual smoke diagnostic, a metadata mismatch, missing
  artifacts, skipped rows, or a forbidden backend.

## 2026-06-21 Update

The MechanismLab-style framework extraction attempt is frozen and removed from
the active package. SELF-GROUND remains a task/evidence harness over
TransformerLens and SAELens, with local JSONL/CSV/Markdown artifacts as the
claim source of truth. Future framework extraction requires multiple completed
end-to-end tasks and must reduce code rather than add generic interfaces.
