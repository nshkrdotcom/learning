# QC Checklist

Scope: Phase 3 artifact-backed blocker and acceptance-invariant pass for
SELF-GROUND token-contrast evaluation. This checklist is updated during final
QC for the current implementation.

## Phase 1 Preservation

- [x] real TransformerLens model check command verified
- [x] real residual-dimension activation ranking command verified
- [x] real residual intervention command verified
- [x] Phase 1 artifacts inspected under `runs/check_real_model.json`
- [x] Phase 1 artifacts inspected under `runs/test_real_activation_ranking`
- [x] Phase 1 artifacts inspected under `runs/test_real_residual_intervention`

Commands run:

```bash
uv run python scripts/check_real_model.py --device cpu
uv run python scripts/run_real_activation_ranking.py --device cpu --per-family 1 --top-k-features 5 --out runs/test_real_activation_ranking
uv run python scripts/run_real_residual_intervention.py --ranking-dir runs/test_real_activation_ranking --device cpu --top-k-features 2 --out runs/test_real_residual_intervention
```

Status: all passed.

## Phase 2 Preservation

- [x] semantic SAE compatibility command verified for the documented deduped SAE
- [x] SAE ranking command verified for the documented deduped SAE
- [x] decoded SAE intervention command verified for the documented deduped SAE
- [x] compatibility artifact reports metadata, shape, and reconstruction compatibility
- [x] SAE ranking metadata reports declared/requested model and hook information
- [x] no shape-only diagnostic is treated as production-compatible

Commands run:

```bash
uv run python scripts/check_sae_compatibility.py --model EleutherAI/pythia-70m-deduped --hook-point blocks.2.hook_resid_post --sae-release pythia-70m-deduped-res-sm --sae-id blocks.2.hook_resid_post --device cpu --out runs/check_sae_compatibility_pythia70m_deduped_res_sm.json
uv run python scripts/run_real_activation_ranking.py --model EleutherAI/pythia-70m-deduped --hook-point blocks.2.hook_resid_post --feature-source sae --sae-release pythia-70m-deduped-res-sm --sae-id blocks.2.hook_resid_post --device cpu --per-family 1 --top-k-features 5 --out runs/test_real_sae_ranking
uv run python scripts/run_real_sae_intervention.py --ranking-dir runs/test_real_sae_ranking --out runs/test_real_sae_intervention --model EleutherAI/pythia-70m-deduped --hook-point blocks.2.hook_resid_post --sae-release pythia-70m-deduped-res-sm --sae-id blocks.2.hook_resid_post --top-k-features 2 --operation ablate --patch-mode delta --device cpu
```

Status: all passed.

## Phase 3 Finish Pass

- [x] required Phase 3 task families are enforced by default
- [x] missing required families appear in validation summaries with zero counts
- [x] model-load failures write artifact-backed blocked runs
- [x] SAE compatibility is verified before post-compatibility SAE loading
- [x] post-compatibility SAE-load failures write artifact-backed blocked runs
- [x] baseline non-finite scores write `baseline_validation.json`
- [x] baseline non-finite scores block before decoded intervention rows
- [x] reports include stable `blocker_reason`
- [x] Markdown mechanism report includes full evidence sections
- [x] required artifact gate includes the full successful Phase 3 artifact layout
- [x] mechanism report uses required artifact presence, not config-only claims
- [x] strong evidence is denied by high relative norm drift
- [x] strong evidence is denied by high norm-drift warning rate
- [x] strong evidence requires actual random control result rows
- [x] zero top deltas cannot produce candidate evidence
- [x] NaN/Inf/malformed summary values cannot produce candidate evidence
- [x] missing required artifacts produce `blocked`
- [x] diagnostic metadata mismatch cannot produce candidate or strong evidence
- [x] tiny smoke runs cannot produce strong evidence
- [x] baseline scores in result rows come from `baseline_task_scores.jsonl`
- [x] target and matched-control intervention telemetry are recorded separately
- [x] row-level telemetry provenance is explicit
- [x] skipped non-finite rows are counted in `skipped_behavioral_rows.json`
- [x] all-skipped runs become `blocked`
- [x] partially skipped runs cannot become `strong_candidate_evidence`
- [x] Phase 3 script exposes telemetry warning thresholds
- [x] Phase 3 Typer CLI exposes telemetry warning thresholds
- [x] README duplicate/misplaced CLI section removed
- [x] Phase 3 docs describe final telemetry, skipped-row, blocker, baseline, and claim-gate behavior

Command run:

```bash
uv run python scripts/run_phase3_behavioral_evaluation.py --ranking-dir runs/test_real_sae_ranking --out runs/test_phase3_behavioral_evaluation --model EleutherAI/pythia-70m-deduped --hook-point blocks.2.hook_resid_post --sae-release pythia-70m-deduped-res-sm --sae-id blocks.2.hook_resid_post --per-family 2 --top-k-features 2 --baseline-mode top-vs-random-multiseed --random-seeds 7,11,13 --operations ablate --patch-mode delta --device cpu --write-report
```

Status: passed with `compatible=true`, `n_tasks_valid=6`, `n_rows=24`, and
`report_written=true`.

Artifact inspection:

- [x] `runs/test_phase3_behavioral_evaluation/skipped_behavioral_rows.json` shows `n_skipped_rows=0`
- [x] `runs/test_phase3_behavioral_evaluation/baseline_validation.json` shows `finite=true`
- [x] `runs/test_phase3_behavioral_evaluation/mechanism_report.json` shows `claim_status=insufficient_evidence`
- [x] `runs/test_phase3_behavioral_evaluation/mechanism_report.json` shows `blocker_reason=null`
- [x] `runs/test_phase3_behavioral_evaluation/mechanism_report.json` shows `required_artifacts_present=true`
- [x] `runs/test_phase3_behavioral_evaluation/mechanism_report.json` shows `n_written_rows=24`
- [x] `runs/test_phase3_behavioral_evaluation/behavioral_intervention_results.jsonl` contains target/control deltas and separate telemetry fields
- [x] `runs/test_phase3_behavioral_evaluation/mechanism_report.md` includes all required evidence sections
- [x] `runs/test_phase3_behavioral_evaluation/README.md` includes skipped-row accounting and conservative interpretation text

## Tests And Static Review

- [x] `uv sync` passed
- [x] `uv run ruff check .` passed
- [x] `uv run pytest` passed: `200 passed, 12 skipped`
- [x] `uv run pytest --run-integration` passed without SAE env:
  `207 passed, 5 skipped`
- [x] SAE-configured integration passed: `212 passed`
- [x] root CLI help inspected
- [x] Phase 3 CLI help inspected
- [x] SAE compatibility CLI help inspected
- [x] SAE intervention CLI help inspected
- [x] activation-ranking CLI help inspected
- [x] fake/dummy/mock/placeholder scan reviewed; hits are tests or negative blocker/QC/docs language only
- [x] overclaiming scan reviewed; hits are explicit not-supported claims or claim-status identifiers
- [x] proxy/residual-SAE wording scan reviewed; no production overclaiming found

Commands run:

```bash
uv sync
uv run ruff check .
uv run pytest
uv run pytest --run-integration
SELF_GROUND_SAE_MODEL=EleutherAI/pythia-70m-deduped SELF_GROUND_SAE_RELEASE=pythia-70m-deduped-res-sm SELF_GROUND_SAE_ID=blocks.2.hook_resid_post uv run pytest --run-integration
uv run self-ground --help
uv run self-ground run-phase3-behavioral-evaluation --help
uv run self-ground check-sae-compatibility --help
uv run self-ground run-sae-intervention --help
uv run self-ground run-activation-ranking --help
grep -RniE "fake|dummy|mock|simulated|placeholder" src scripts docs README.md tests || true
grep -RniE "broad behavioral|mechanism discovery|introspection|monosemantic|complete SELF-GROUND|strong_candidate_evidence" README.md docs src tests || true
grep -RniE "feature_space_proxy.*causal|residual.*SAE feature|adapter fake|--adapter fake" src scripts README.md docs tests || true
```

Notes:

- A bare `python` artifact-inspection command failed because `python` is not on
  PATH in this shell. The inspection was rerun successfully with
  `uv run python`.
- No external blockers remain for the documented Pythia-70M-deduped SAE path in
  this environment.

## Git Finalization

- [x] `git status --short` reviewed before commit
- [ ] final commit created
- [ ] push completed

The exact final commit hash and push result are recorded in the final response
after commit/push, because the commit hash cannot be embedded in the same
commit before it exists.

## E003 Calibrated Task Bank Pass

- [x] richer Phase 3 task bank implemented
- [x] task bank generation is deterministic
- [x] tokenization rejection artifacts are written
- [x] baseline-only task bank calibration implemented
- [x] calibration fails closed when a required family underfills
- [x] Phase 3 evaluation accepts `--task-source file`
- [x] SAE ranking accepts `--task-source file`
- [x] task-source metadata is written to ranking and evaluation artifacts
- [x] mechanism reports include task-source metadata
- [x] calibrated candidate wording is qualified in reports
- [x] E003 orchestration script implemented
- [x] E002/E003 comparison script implemented
- [x] E003 ran on CUDA
- [x] E003 repaired `property_negation` baseline coverage
- [x] E003 remained `insufficient_evidence`
- [x] docs and ledgers updated with artifact-backed E003 result

Commands run:

```bash
uv run python scripts/run_e003_calibrated_negation_sae.py --device cuda --task-bank data/phase3_task_bank/pythia70m_negation_candidate_bank.json --per-family-candidates 80 --min-calibrated-per-family 10 --min-baseline-margin 0.1 --ranking-top-k 50 --eval-top-k 5 --operations ablate --random-seeds 7,11,13 --out-root runs --force
uv run python scripts/inspect_claim_run.py --run-dir runs/e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density --json
uv run python scripts/compare_e002_e003.py --e002 runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density --e003 runs/e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density --out runs/diagnostics/e003_vs_e002_comparison
```

Observed E003 artifacts:

- [x] task bank:
  `data/phase3_task_bank/pythia70m_negation_candidate_bank.json`
- [x] calibration:
  `runs/e003_task_bank_calibration_pythia70m_margin0p1_min10/calibration_summary.json`
- [x] ranking:
  `runs/e003_real_sae_ranking_pythia70m_l2_calibrated_pf10_top50/feature_rankings.csv`
- [x] evaluation:
  `runs/e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density/mechanism_report.json`
- [x] inspection:
  `runs/e003_negation_eval_pythia70m_l2_calibrated_pf10_top5_density/inspection_summary.json`
- [x] comparison:
  `runs/diagnostics/e003_vs_e002_comparison/comparison.json`

Observed E003 result:

- candidate tasks: 240 total, 80 per required family
- token-valid candidates: 240
- kept by calibration:
  `property_negation=10`, `sentiment_negation=36`, `state_negation=23`
- evaluation baseline pass rate: `1.0`
- behavioral rows: 552
- skipped rows: 0
- claim status: `insufficient_evidence`
- top target delta: `0.6277369900026183`
- top matched-control delta: `0.7188387469968934`
- specificity gap: `-0.09110175699427508`

Interpretation: E003 fixed the baseline task-suite coverage problem but did not
produce candidate evidence because matched-control movement exceeded target
movement.

## E002 Calibration And Feature-Selection Pass

- [x] task calibration analyzer implemented
- [x] feature-selection analyzer implemented
- [x] preregistered baseline-only task calibration implemented
- [x] calibrated task artifacts are written when calibration is enabled
- [x] calibration blockers write no intervention rows
- [x] feature selection modes are recorded in `feature_sets.json`
- [x] `top-positive` feature selection implemented
- [x] `top-family-consistent` fails closed without per-family ranking fields
- [x] original E002 calibration analyzed from artifacts
- [x] original E002 feature selection analyzed from artifacts
- [x] bounded calibrated intended-direction/top-absolute variant run
- [x] bounded calibrated margin/top-absolute variant run
- [x] bounded calibrated intended-direction/top-positive variant run
- [x] variant comparison artifact written
- [x] SAEBench/RAVEL probe rerun
- [x] D008 SAEBench/RAVEL boundary decision written
- [x] claim ledger updated
- [x] research log updated
- [x] run ledger updated

Commands run:

```bash
uv run python scripts/analyze_task_calibration.py --run-dir runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density --out runs/diagnostics/e002_task_calibration
uv run python scripts/analyze_feature_selection.py --ranking-dir runs/e002_real_sae_ranking_pythia70m_deduped_l2_pf10_top50 --eval-dir runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density --out runs/diagnostics/e002_feature_selection
uv run python scripts/run_negation_ravel_eval.py --ranking-dir runs/e002_real_sae_ranking_pythia70m_deduped_l2_pf10_top50 --out runs/e002_calibrated_intended_direction_top_abs --model EleutherAI/pythia-70m-deduped --hook-point blocks.2.hook_resid_post --sae-release pythia-70m-deduped-res-sm --sae-id blocks.2.hook_resid_post --per-family 10 --top-k-features 5 --baseline-mode top-vs-random-density-and-bottom-active --random-seeds 7,11,13 --operations ablate --patch-mode delta --device cuda --task-calibration-mode baseline-intended-direction --min-calibrated-tasks-per-family 3 --allow-family-drop false --feature-selection-mode top-absolute
uv run python scripts/run_negation_ravel_eval.py --ranking-dir runs/e002_real_sae_ranking_pythia70m_deduped_l2_pf10_top50 --out runs/e002_calibrated_margin_top_abs --model EleutherAI/pythia-70m-deduped --hook-point blocks.2.hook_resid_post --sae-release pythia-70m-deduped-res-sm --sae-id blocks.2.hook_resid_post --per-family 10 --top-k-features 5 --baseline-mode top-vs-random-density-and-bottom-active --random-seeds 7,11,13 --operations ablate --patch-mode delta --device cuda --task-calibration-mode baseline-margin --min-baseline-margin 0.1 --min-calibrated-tasks-per-family 3 --allow-family-drop false --feature-selection-mode top-absolute
uv run python scripts/run_negation_ravel_eval.py --ranking-dir runs/e002_real_sae_ranking_pythia70m_deduped_l2_pf10_top50 --out runs/e002_calibrated_intended_direction_top_positive --model EleutherAI/pythia-70m-deduped --hook-point blocks.2.hook_resid_post --sae-release pythia-70m-deduped-res-sm --sae-id blocks.2.hook_resid_post --per-family 10 --top-k-features 5 --baseline-mode top-vs-random-density-and-bottom-active --random-seeds 7,11,13 --operations ablate --patch-mode delta --device cuda --task-calibration-mode baseline-intended-direction --min-calibrated-tasks-per-family 3 --allow-family-drop false --feature-selection-mode top-positive
uv run python scripts/compare_e002_variants.py --out runs/diagnostics/e002_variant_comparison runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density runs/e002_calibrated_intended_direction_top_abs runs/e002_calibrated_margin_top_abs runs/e002_calibrated_intended_direction_top_positive
uv run python scripts/probe_saebench_ravel_bridge.py --out runs/tooling_spikes/saebench_ravel_bridge
```

Observed results:

- [x] baseline intended-direction pass rate is `0.23333333333333334`
- [x] property-negation calibration pass count is `0/10`
- [x] feature-selection diagnosis labels include `target_effect_present_but_not_specific`
- [x] feature-selection diagnosis labels include `control_effect_dominates`
- [x] intended-direction/top-absolute variant blocked with `task_calibration_failed`
- [x] margin/top-absolute variant blocked with `task_calibration_failed`
- [x] intended-direction/top-positive variant blocked with `task_calibration_failed`
- [x] no calibrated variant wrote behavioral intervention rows
- [x] no calibrated variant reached candidate evidence
- [x] SAEBench/RAVEL probe status remains `not_installed`

## Engine Boundary Hardening

- [x] `docs/decision_log/D007_engine_boundary.md` exists
- [x] `docs/prior_art_engine_matrix.md` exists
- [x] TransformerLens is recorded as the local patching backend
- [x] SAELens is recorded as the SAE backend
- [x] Phase 3 reports record `engine_backend`
- [x] forbidden `self_ground_generic_engine` backend blocks claim reports
- [x] residual-dimension runs are diagnostic-only and claim-ineligible
- [x] feature-space proxy runs are legacy-only and claim-ineligible
- [x] RAVEL-style cause/isolation aliases are tested
- [x] nnsight and pyvene remain outside core dependencies
- [x] new boundary tests pass

Commands run for this boundary pass:

```bash
uv run ruff check .
uv run pytest tests/test_engine_boundary.py tests/test_ravel_alignment.py tests/test_residual_intervention_artifacts.py tests/test_experiment.py tests/test_mechanism_report.py -q
uv run pytest
uv run pytest --run-integration
SELF_GROUND_SAE_MODEL=EleutherAI/pythia-70m-deduped SELF_GROUND_SAE_RELEASE=pythia-70m-deduped-res-sm SELF_GROUND_SAE_ID=blocks.2.hook_resid_post uv run pytest --run-integration
uv run python scripts/diagnostics/run_residual_smoke_patch.py --ranking-dir runs/test_real_activation_ranking --device cpu --top-k-features 2 --out runs/test_residual_smoke_patch
uv run python scripts/run_negation_ravel_eval.py --ranking-dir runs/test_real_sae_ranking --out runs/test_negation_ravel_eval --model EleutherAI/pythia-70m-deduped --hook-point blocks.2.hook_resid_post --sae-release pythia-70m-deduped-res-sm --sae-id blocks.2.hook_resid_post --per-family 2 --top-k-features 2 --baseline-mode top-vs-random-multiseed --random-seeds 7,11,13 --operations ablate --patch-mode delta --device cpu
```

Boundary artifact inspection:

- [x] `runs/test_real_residual_intervention/config.json` shows `diagnostic_only=true`
- [x] `runs/test_residual_smoke_patch/config.json` shows `claim_eligible=false`
- [x] `runs/test_real_sae_ranking/activation_metadata.json` shows `engine_backend=transformer_lens`
- [x] `runs/test_negation_ravel_eval/config.json` shows `evaluation_adapter=negation_ravel_adapter`
- [x] `runs/test_negation_ravel_eval/mechanism_report.json` shows `engine_backend=transformer_lens`

## Density-Matched Controls And SAEBench Probe

- [x] activation-density-matched control sampler implemented
- [x] `feature_sets.json` records `matched_control_metadata`
- [x] density-matched labels are emitted as `density_matched_seed_*`
- [x] density-matched controls exclude top feature IDs
- [x] stats source is recorded as `per_condition_mean_approximation` when true per-example density is unavailable
- [x] relaxed tolerance metadata is recorded
- [x] strong candidate evidence requires at least three actual density-matched control rows
- [x] candidate reports without density-matched controls carry an explicit limitation
- [x] SAEBench/RAVEL probe writes `config.json`, `probe_result.json`, and `README.md`
- [x] probe status in this environment is `not_installed`

Commands run for this pass:

```bash
uv sync
uv run ruff check .
uv run pytest
uv run pytest --run-integration
SELF_GROUND_SAE_MODEL=EleutherAI/pythia-70m-deduped SELF_GROUND_SAE_RELEASE=pythia-70m-deduped-res-sm SELF_GROUND_SAE_ID=blocks.2.hook_resid_post uv run pytest --run-integration
uv run python scripts/run_negation_ravel_eval.py --ranking-dir runs/test_real_sae_ranking --out runs/test_negation_ravel_eval_density_matched --model EleutherAI/pythia-70m-deduped --hook-point blocks.2.hook_resid_post --sae-release pythia-70m-deduped-res-sm --sae-id blocks.2.hook_resid_post --per-family 2 --top-k-features 2 --baseline-mode top-vs-density-matched-multiseed --random-seeds 7,11,13 --operations ablate --patch-mode delta --device cpu
uv run python scripts/probe_saebench_ravel_bridge.py --out runs/probe_saebench_ravel_bridge
```

Observed results:

- [x] `uv run pytest` passed: `161 passed, 12 skipped`
- [x] `uv run pytest --run-integration` passed without SAE env: `168 passed, 5 skipped`
- [x] SAE-configured integration passed: `173 passed`
- [x] density-matched run wrote `runs/test_negation_ravel_eval_density_matched`
- [x] density-matched run wrote three `density_matched_seed_*` feature sets
- [x] density-matched feature sets had no overlap with top feature IDs
- [x] density-matched metadata records `stats_source=per_condition_mean_approximation`
- [x] density-matched metadata records relaxed tolerances for the tiny ranking artifact
- [x] density-matched mechanism report status is `insufficient_evidence`
- [x] probe artifact status is `not_installed` with `ModuleNotFoundError` blockers for attempted SAEBench/RAVEL packages

## Library-Backed Evidence Refocus

- [x] `docs/code_classification.md` reviewed
- [x] `docs/execution_stack.md` reviewed
- [x] active `src/mechanismlab/` framework package removed
- [x] `mechanismlab` CLI removed from `pyproject.toml`
- [x] `self_ground/mechanismlab_adapter.py` removed
- [x] serious E002 GPU command documented
- [x] `scripts/inspect_claim_run.py` works on diagnostic artifacts
- [x] diagnostic density-matched run completed
- [x] diagnostic claim status recorded as conservative
- [x] run ledger updated
- [x] research log updated
- [x] claim ledger updated
- [x] SAEBench/RAVEL probe remains bounded and honest

Commands run for this pass:

```bash
uv sync
uv run ruff check .
uv run pytest
uv run pytest --run-integration
SELF_GROUND_SAE_MODEL=EleutherAI/pythia-70m-deduped SELF_GROUND_SAE_RELEASE=pythia-70m-deduped-res-sm SELF_GROUND_SAE_ID=blocks.2.hook_resid_post uv run pytest --run-integration
uv run python scripts/run_negation_ravel_eval.py --ranking-dir runs/test_real_sae_ranking --out runs/diagnostic_negation_ravel_eval_density_matched --model EleutherAI/pythia-70m-deduped --hook-point blocks.2.hook_resid_post --sae-release pythia-70m-deduped-res-sm --sae-id blocks.2.hook_resid_post --per-family 2 --top-k-features 2 --baseline-mode top-vs-density-matched-multiseed --random-seeds 7,11,13 --operations ablate --patch-mode delta --device cpu
uv run python scripts/inspect_claim_run.py --run-dir runs/diagnostic_negation_ravel_eval_density_matched
uv run python scripts/probe_saebench_ravel_bridge.py --out runs/tooling_spikes/saebench_ravel_bridge
```

Observed results:

- [x] `uv run ruff check .` passed
- [x] `uv run pytest` passed: `175 passed, 12 skipped`
- [x] `uv run pytest --run-integration` passed without SAE env: `182 passed, 5 skipped`
- [x] SAE-configured integration passed: `187 passed`
- [x] diagnostic run wrote `runs/diagnostic_negation_ravel_eval_density_matched`
- [x] diagnostic run used `engine_backend=transformer_lens`
- [x] diagnostic run used `sae_backend=sae_lens`
- [x] diagnostic run compatibility is semantic/shape/reconstruction compatible
- [x] diagnostic run wrote 24 behavioral rows and 0 skipped rows
- [x] inspector reports `run_classification=diagnostic_or_smoke_run`
- [x] inspector reports `claim_status=insufficient_evidence`
- [x] inspector reports `top_target_delta=0.0`, `top_control_delta=0.0`, and `specificity_gap=0.0`
- [x] SAEBench/RAVEL probe status is `not_installed`; no upstream integration claimed

## E002 Gate And Zero-Effect Diagnosis

- [x] CUDA capability artifact written
- [x] serious E002 GPU ranking completed
- [x] serious E002 GPU evaluation completed
- [x] E002 inspection summary written
- [x] serious E002 claim status remains artifact-backed
- [x] prior zero-effect diagnostic run diagnosed
- [x] decoded SAE patch sanity check completed for old diagnostic ranking
- [x] decoded SAE patch sanity check completed for E002 ranking
- [x] optional ablate+amplify diagnostic completed
- [x] ledgers updated

Commands run for this pass:

```bash
uv run python scripts/check_run_capability.py --out runs/capability_check --model EleutherAI/pythia-70m-deduped --sae-release pythia-70m-deduped-res-sm --sae-id blocks.2.hook_resid_post
uv run python scripts/run_e002_negation_sae_density_matched.py --device cuda --ranking-per-family 10 --eval-per-family 10 --ranking-top-k 50 --eval-top-k 5 --operations ablate --random-seeds 7,11,13 --out-root runs --allow-cpu-serious-run false
uv run python scripts/inspect_claim_run.py --run-dir runs/e002_negation_ravel_eval_pythia70m_deduped_l2_pf10_top5_density
uv run python scripts/diagnose_zero_effect_run.py --run-dir runs/diagnostic_negation_ravel_eval_density_matched --ranking-dir runs/test_real_sae_ranking --out runs/diagnostics/zero_effect_diagnostic_negation_ravel_eval_density_matched
uv run python scripts/check_decoded_sae_patch_nonzero.py --ranking-dir runs/test_real_sae_ranking --model EleutherAI/pythia-70m-deduped --hook-point blocks.2.hook_resid_post --sae-release pythia-70m-deduped-res-sm --sae-id blocks.2.hook_resid_post --top-k-features 2 --operation ablate --patch-mode delta --prompt "The movie was not" --device cpu --out runs/diagnostics/check_decoded_sae_patch_nonzero
uv run python scripts/check_decoded_sae_patch_nonzero.py --ranking-dir runs/e002_real_sae_ranking_pythia70m_deduped_l2_pf10_top50 --model EleutherAI/pythia-70m-deduped --hook-point blocks.2.hook_resid_post --sae-release pythia-70m-deduped-res-sm --sae-id blocks.2.hook_resid_post --top-k-features 5 --operation ablate --patch-mode delta --prompt "The movie was not" --device cuda --out runs/diagnostics/check_decoded_sae_patch_nonzero_e002
uv run python scripts/run_negation_ravel_eval.py --ranking-dir runs/e002_real_sae_ranking_pythia70m_deduped_l2_pf10_top50 --out runs/diagnostic_negation_ravel_eval_density_matched_ablate_amplify --model EleutherAI/pythia-70m-deduped --hook-point blocks.2.hook_resid_post --sae-release pythia-70m-deduped-res-sm --sae-id blocks.2.hook_resid_post --per-family 2 --top-k-features 2 --baseline-mode top-vs-density-matched-multiseed --random-seeds 7,11,13 --operations ablate,amplify --amplify-factors 2.0 --patch-mode delta --device cpu
```

Observed results:

- [x] capability says `can_attempt_e002_gpu=true`
- [x] CUDA device is `NVIDIA GeForce RTX 5060 Ti`
- [x] E002 inspection reports `run_classification=serious_gpu_evidence_run`
- [x] E002 inspection reports `claim_status=insufficient_evidence`
- [x] E002 inspection reports `top_target_delta=0.03419952392578125`
- [x] E002 inspection reports `top_control_delta=0.0546810785929362`
- [x] E002 inspection reports `specificity_gap=-0.02048155466715495`
- [x] zero-effect diagnosis labels prior diagnostic as `all_row_deltas_zero`, `decoded_delta_norm_zero_or_missing`, and `task_baseline_not_calibrated`
- [x] old-ranking patch sanity reports selected features inactive and logits unchanged
- [x] E002-ranking patch sanity reports nonzero decoded patch and changed logits
- [x] ablate+amplify diagnostic remains `insufficient_evidence`
