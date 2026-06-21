# QC Checklist

Scope: Phase 3 finish pass for SELF-GROUND token-contrast evaluation. This
checklist records the current repository state after hardening evidence gates,
telemetry provenance, non-finite accounting, docs, and public command parity.

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
- [x] Phase 3 docs describe final telemetry, skipped-row, and claim-gate behavior

Command run:

```bash
uv run python scripts/run_phase3_behavioral_evaluation.py --ranking-dir runs/test_real_sae_ranking --out runs/test_phase3_behavioral_evaluation --model EleutherAI/pythia-70m-deduped --hook-point blocks.2.hook_resid_post --sae-release pythia-70m-deduped-res-sm --sae-id blocks.2.hook_resid_post --per-family 2 --top-k-features 2 --baseline-mode top-vs-random-multiseed --random-seeds 7,11,13 --operations ablate --patch-mode delta --device cpu --write-report
```

Status: passed with `compatible=true`, `n_tasks_valid=6`, `n_rows=24`, and
`report_written=true`.

Artifact inspection:

- [x] `runs/test_phase3_behavioral_evaluation/skipped_behavioral_rows.json` shows `n_skipped_rows=0`
- [x] `runs/test_phase3_behavioral_evaluation/mechanism_report.json` shows `claim_status=insufficient_evidence`
- [x] `runs/test_phase3_behavioral_evaluation/mechanism_report.json` shows `required_artifacts_present=true`
- [x] `runs/test_phase3_behavioral_evaluation/mechanism_report.json` shows `n_written_rows=24`
- [x] `runs/test_phase3_behavioral_evaluation/behavioral_intervention_results.jsonl` contains target/control deltas and separate telemetry fields
- [x] `runs/test_phase3_behavioral_evaluation/README.md` includes skipped-row accounting and conservative interpretation text

## Tests And Static Review

- [x] `uv sync` passed
- [x] `uv run ruff check .` passed
- [x] `uv run pytest` passed: `130 passed, 12 skipped`
- [x] `uv run pytest --run-integration` passed without SAE env: `137 passed, 5 skipped`
- [x] SAE-configured integration passed: `142 passed`
- [x] root CLI help inspected
- [x] Phase 3 CLI help inspected
- [x] SAE compatibility CLI help inspected
- [x] SAE intervention CLI help inspected
- [x] activation-ranking CLI help inspected
- [x] fake/dummy/mock/placeholder scan reviewed; hits are tests or negative QC/docs language only
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
