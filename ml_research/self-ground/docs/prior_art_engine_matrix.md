# Prior-Art Engine Matrix

SELF-GROUND owns the negation task specification, compatibility gates, and
artifact-backed claim ledger. It does not own generic model patching or SAE
runtime machinery.

| Engine / project | Use directly / wrap / not used | Reason | SELF-GROUND-owned delta |
| --- | --- | --- | --- |
| TransformerLens patching | Use directly | Local model execution exposes activations and `run_with_hooks` patching. | Negation task prompts, selected hook/feature configuration, artifacts. |
| SAELens | Use directly | SAE loading, encode, and decode are upstream responsibilities. | Semantic compatibility gate and run packaging. |
| explanare/ravel | Wrap | RAVEL’s cause/isolation framing matches target-change plus control-preservation evaluation. | Negation-scope attribute schema and token-contrast adapter. |
| adamkarvonen/SAEBench | Wrap / investigate first | SAEBench includes RAVEL evaluations and should be tried before maintaining custom evaluation code. | Evidence ledger around negation-specific runs and blocked-run records. |
| MaheepChaudhary/SAE-Ravel | Reference / investigate | Shows SAE evaluation on RAVEL-style tasks. | Document whether negation requires a different schema. |
| nnsight/NDIF | Optional | Remote execution path for larger models; not needed for MVP local runs. | Same claim ledger around remote run artifacts. |
| pyvene | Optional reference | Useful for serializable/trainable intervention abstractions beyond the current TransformerLens path. | None in core Phase 3. |
| Delphi / sae-auto-interp | Not used in core path | Auto-interpretation can label features but does not replace causal/isolation evidence. | Possible annotation input only after claim gates pass. |

## Current status

The current Phase 3 implementation uses TransformerLens for execution/patching,
SAELens for SAE transforms, and a SELF-GROUND negation RAVEL-style adapter for
cause/isolation summaries. A bounded SAEBench/RAVEL probe now exists:

```bash
uv run python scripts/probe_saebench_ravel_bridge.py \
  --out runs/probe_saebench_ravel_bridge
```

The latest result should be checked at
`runs/probe_saebench_ravel_bridge/probe_result.json`. Expanding the custom
evaluator requires first recording whether the upstream bridge is feasible,
blocked by a missing package, or blocked by API incompatibility.
