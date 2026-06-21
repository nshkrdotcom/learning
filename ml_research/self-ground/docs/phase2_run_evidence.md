# Phase 2 Run Evidence

This note preserves the real SAE execution evidence for Phase 2 because `runs/`
artifacts are intentionally ignored by git.

## Verified SAE

- model: `EleutherAI/pythia-70m-deduped`
- hook point: `blocks.2.hook_resid_post`
- SAE release: `pythia-70m-deduped-res-sm`
- SAE id: `blocks.2.hook_resid_post`
- source: installed SAELens pretrained SAE directory in `sae-lens==6.44.3`

The matching model is the deduped Pythia checkpoint because the SAELens release
metadata declares `model='pythia-70m-deduped'`.

## Commands Run

```bash
uv run python scripts/check_sae_compatibility.py \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --device cpu \
  --out runs/check_sae_compatibility_pythia70m_deduped_res_sm.json
```

Result: passed with `compatible=true`.

```bash
uv run python scripts/run_real_activation_ranking.py \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --feature-source sae \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --device cpu \
  --per-family 1 \
  --top-k-features 5 \
  --out runs/test_real_sae_ranking
```

Result: passed with top features:

```text
sae_12300
sae_25521
sae_26935
sae_4863
sae_21948
```

```bash
uv run python scripts/run_real_sae_intervention.py \
  --ranking-dir runs/test_real_sae_ranking \
  --out runs/test_real_sae_intervention \
  --model EleutherAI/pythia-70m-deduped \
  --hook-point blocks.2.hook_resid_post \
  --sae-release pythia-70m-deduped-res-sm \
  --sae-id blocks.2.hook_resid_post \
  --top-k-features 2 \
  --operation ablate \
  --patch-mode delta \
  --device cpu
```

Result: passed with `compatible=true`, `n_pairs=4`, `n_features=2`.

```bash
SELF_GROUND_SAE_RELEASE=pythia-70m-deduped-res-sm \
SELF_GROUND_SAE_ID=blocks.2.hook_resid_post \
uv run pytest --run-integration
```

Result: `80 passed, 9 warnings`.

## Compatibility Artifact Summary

`runs/check_sae_compatibility_pythia70m_deduped_res_sm.json` contained:

```json
{
  "activation_shape": [4, 6, 512],
  "encoded_shape": [4, 6, 32768],
  "decoded_shape": [4, 6, 512],
  "d_model": 512,
  "d_sae": 32768,
  "compatible": true,
  "status": "ok"
}
```

## SAE Ranking Artifact Summary

`runs/test_real_sae_ranking/feature_rankings.csv` was nonempty and sorted by
`abs_score`. The top row was:

```text
feature_id=sae_12300
score=0.2323818802833557
mean_pos=0.39025214314460754
mean_neg=0.17711713910102844
mean_para=0.20496591925621033
mean_decoy=0.18571904301643372
```

## SAE Intervention Artifact Summary

`runs/test_real_sae_intervention/` contained:

- `config.json`
- `compatibility.json`
- `selected_features.json`
- `intervention_results.jsonl`
- `summary.csv`
- `README.md`

The run selected:

```text
sae_12300
sae_25521
```

`summary.csv` contained:

```text
feature_set,operation,patch_mode,n_pairs,signed_negation_delta_mean,signed_control_delta_mean,absolute_negation_delta_mean,absolute_control_delta_mean,signed_specificity_score_mean,absolute_specificity_score_mean
sae_12300+sae_25521,ablate,delta,4,0.004544973373413086,0.006017208099365234,0.018645524978637695,0.014856815338134766,-0.0014722347259521484,0.0037887096405029297
```

Interpretation: the decoded SAE intervention path executed successfully and
measured nonzero real logit-contrast deltas. The small evidence run is not a
claim of mechanism discovery; it is a proof that the Phase 2 real path works
with this concrete SAE.
