# Code Classification

This pass classifies active SELF-GROUND code by what it actually does. The goal
is to keep real execution and conservative audit code, and remove generic
framework-shaped code that does not perform a real run or a real artifact audit.

| path | classification | keep_action | reason | existing stack that owns this if any |
| --- | --- | --- | --- | --- |
| `src/self_ground/real_behavioral_intervention.py` | execution | keep | Runs the Phase 3 decoded SAE token-contrast path and writes artifacts. | TransformerLens owns model execution/hooks; SAELens owns SAE encode/decode. |
| `src/self_ground/real_sae_intervention.py` | execution | keep | Runs decoded SAE intervention rows for minimal pairs. | TransformerLens + SAELens. |
| `src/self_ground/sae_interventions.py` | execution | keep | Thin SAE feature edit/decode/patch call site used by the real decoded intervention path. | SAELens owns encode/decode; TransformerLens owns hook execution. |
| `src/self_ground/sae_compat.py` | audit | keep | Fail-closed semantic/shape/reconstruction compatibility gate before decoded SAE runs. | SAELens provides SAE object/metadata where available; SELF-GROUND audits compatibility. |
| `src/self_ground/sae_metadata.py` | audit | keep | Extracts SAE-declared model/hook metadata and compares it to the requested run. | SAELens metadata source. |
| `src/self_ground/task_validation.py` | task | keep | Validates Phase 3 token strings and required task families. | SELF-GROUND task layer. |
| `src/self_ground/behavioral_tasks.py` | task | keep | Defines deterministic negation token-contrast tasks. | SELF-GROUND task layer. |
| `src/self_ground/baseline_samplers.py` | audit | keep | Selects activation-density-matched SAE control feature sets from ranking artifacts. | SELF-GROUND evidence controls. |
| `src/self_ground/baselines.py` | audit | keep | Builds top/random/density/bottom-active feature-set controls from SAE rankings. | SELF-GROUND evidence controls. |
| `src/self_ground/mechanism_report.py` | audit | keep | Reads run artifacts and computes conservative claim status. | SELF-GROUND claim ledger. |
| `src/self_ground/ravel_adapter/scoring.py` | audit | keep | Names target/control deltas in RAVEL-style cause/isolation terms without claiming upstream RAVEL integration. | SELF-GROUND temporary negation adapter. |
| `src/self_ground/ravel_adapter/saebench_probe.py` | optional_probe | keep | Bounded import/API probe that writes blocker artifacts if SAEBench/RAVEL is missing or incompatible. | SAEBench/RAVEL if installed. |
| `src/self_ground/engine_boundary.py` | audit | keep | Rejects forbidden generic engine claims and marks residual/proxy paths claim-ineligible. | SELF-GROUND boundary guard. |
| `src/self_ground/hooking.py` | execution | keep | Thin TransformerLens `run_with_hooks` call site. | TransformerLens. |
| `src/self_ground/residual_intervention.py` | legacy | demote | Residual dimensions are diagnostic smoke patches only, not evidence features. | TransformerLens. |
| `src/self_ground/real_residual_intervention.py` | legacy | demote | Produces diagnostic residual intervention artifacts only. | TransformerLens. |
| `src/self_ground/experiment.py` | legacy | demote | Feature-space proxy path remains legacy-only and claim-ineligible. | None; not a real behavioral intervention. |
| `src/self_ground/mechanismlab_adapter.py` | framework_slop | delete | Generic report adapter implied a framework extraction without adding real execution evidence. | None. |
| `src/mechanismlab/*` | framework_slop | delete | Generic schemas/protocols/trackers/CLI did not perform real execution or real SELF-GROUND audit. | None. |
| `tests/mechanismlab/*` | framework_slop | delete | Tested deleted generic framework code rather than SELF-GROUND evidence behavior. | None. |
| `scripts/write_mechanismlab_report.py` | framework_slop | delete | Wrote generic sidecar reports that duplicated the real `mechanism_report` audit. | None. |
| `scripts/check_real_model.py` | execution | keep | Verifies real TransformerLens model activation capture. | TransformerLens. |
| `scripts/run_real_activation_ranking.py` | execution | keep | Runs real residual/SAE activation ranking. | TransformerLens + SAELens. |
| `scripts/check_sae_compatibility.py` | audit | keep | Runs real SAE compatibility verification and writes artifact-backed pass/blocker output. | SAELens + TransformerLens. |
| `scripts/run_real_sae_intervention.py` | execution | keep | Runs real decoded SAE intervention on minimal pairs. | TransformerLens + SAELens. |
| `scripts/run_phase3_behavioral_evaluation.py` | execution | keep | Runs Phase 3 decoded SAE token-contrast evaluation. | TransformerLens + SAELens. |
| `scripts/run_negation_ravel_eval.py` | execution | wrap_thinly | RAVEL-shaped entry point over the same real decoded intervention path; not upstream RAVEL integration. | TransformerLens + SAELens. |
| `scripts/probe_saebench_ravel_bridge.py` | optional_probe | keep | Writes concrete SAEBench/RAVEL import/API feasibility artifacts. | SAEBench/RAVEL if installed. |
| `scripts/run_real_residual_intervention.py` | legacy | demote | Legacy residual diagnostic script; not candidate evidence. | TransformerLens. |
| `scripts/diagnostics/run_residual_smoke_patch.py` | legacy | keep | Explicit diagnostic residual smoke patch alias. | TransformerLens. |
| `docs/architecture_mechanismlab.md` | framework_slop | delete | Described the frozen framework extraction attempt as active architecture. | None. |
| `docs/integrations_mechanismlab.md` | framework_slop | delete | Described generic integrations that are not active SELF-GROUND execution. | None. |
| `docs/migration_self_ground_to_mechanismlab.md` | framework_slop | delete | Encouraged migration away from the real experiment path. | None. |

## Current Boundary

SELF-GROUND is not an intervention engine, model runtime, SAE runtime, tracker
platform, or benchmark suite. It is a negation-scope task and evidence harness
over real TransformerLens and SAELens calls, with optional bounded probes for
upstream benchmark tools.
