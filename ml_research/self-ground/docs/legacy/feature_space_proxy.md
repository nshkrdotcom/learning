# Legacy Feature-Space Proxy Path

`src/self_ground/experiment.py` is retained only for legacy artifact inspection
and fast local tests of feature-space metric arithmetic.

It does not reinject decoded activations into TransformerLens, does not measure
model logits after a patch, and cannot enter `candidate_evidence` or
`strong_candidate_evidence`.

Paper-facing evidence must come from artifact-backed runs that name an external
execution backend, pass SAE semantic compatibility when using SAE features, and
write claim-ledger reports.
