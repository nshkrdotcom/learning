# Experiment Log

Add dated entries after script-generated artifacts exist. Keep entries short and link to the relevant run directory.

## 2026-06-26 GPT-2 Small GPU Capability Verification

- Run: `runs/20260626_142914_capability_check`
- Model: `gpt2-small`
- Result: CUDA was available and the model loaded on `cuda:0` with dtype `torch.float32`.
- GPU: NVIDIA GeForce RTX 5060 Ti.
- Verified checks: one-token forward pass, small-batch forward pass, and selected activation cache all succeeded.
- Blockers: none.

## 2026-06-26 GPT-2 Small Induction Controls

- Run: `runs/20260626_144001_gpt2_small_induction_controls`
- Model: `gpt2-small`
- Examples: 192 total, balanced across six families: `positive_repeat_sequence`, `no_repeat_control`, `shuffled_repeat_control`, `distractor_repeat_control`, `same_token_frequency_control`, and `random_expected_token_control`.
- Positive baseline summary: positive mean expected-token probability `0.2785`, median rank `1.0`, and 32/32 examples rank <= 10.
- Hardest control family by baseline behavior: `distractor_repeat_control`, mean expected-token probability `0.2323`, median rank `1.0`, and 32/32 examples rank <= 10.
- Other high-scoring controls: `same_token_frequency_control` and `shuffled_repeat_control` both had median rank `3.0` and 31/32 examples rank <= 10.
- Top heads by raw positive previous-occurrence attention: L0H1 `0.542`, L0H5 `0.531`, L0H10 `0.248`, L11H8 `0.221`, L0H4 `0.191`.
- Top heads by positive-minus-control gap: best reported gaps were `0.000` because the random-expected-token control uses the same repeated prompt with a wrong target token, exposing that raw previous-occurrence attention is not target-specific.
- Controls exposed false positives: yes. L0H1 and L0H5 attend strongly on positives, but also attend strongly on distractor and random-expected-token controls.
- Logit lens by family: `logit_lens_by_family.csv` exists; best positive layer was 9 and hardest control family by expected-token probability was `distractor_repeat_control`.
- Limitation: this is still practice work. These controls are simple and not a publication-quality induction-head benchmark.
- Next step: inspect `attention_by_family.csv`, then design a tiny controlled patching pass that compares a small number of positive examples against the high-scoring control families.

## 2026-06-26 Controlled Patching Follow-up Plan

- Source run: `runs/20260626_144001_gpt2_small_induction_controls`
- Question: when selected raw attention candidates are patched, do positives move more than the high-scoring controls?
- Candidate selection: read `attention_summary.json` and `attention_by_family.csv`, then select top raw-positive heads, top control-firing heads, any positive-gap heads, and a few deterministic comparison heads.
- Patching scope: tiny by default, selected candidates only, four families, eight examples per family, final position, and layer-level `attn_out` unless head-specific patching is explicitly recorded.
- Expected useful result: a candidate may be positive-specific, nonspecific, no-effect, insufficient, or denominator-problem. Nonspecific movement is a valid learning result.
- Next step: run candidate selection and controlled patching, then update this log with actual artifacts and effect sizes.

## 2026-06-26 GPT-2 Small Controlled Patching Follow-up

- Run: `runs/20260626_144001_gpt2_small_induction_controls`
- Model: `gpt2-small`; GPU path had already been verified for this workflow on CUDA, and the controlled patching command loaded the TransformerLens model under the CUDA config without blockers.
- Candidate-selection artifact: `runs/20260626_144001_gpt2_small_induction_controls/controlled_patching_candidates.csv`
- Controlled-patching artifacts: `controlled_patching_results.csv`, `controlled_patching_by_family.csv`, `controlled_patching_by_candidate.csv`, `controlled_patching_summary.json`, `controlled_patching_notes.md`, `figures/controlled_patching_by_family.png`, and `figures/controlled_patching_candidate_gap.png`
- Families patched: `positive_repeat_sequence`, `distractor_repeat_control`, `random_expected_token_control`, and `same_token_frequency_control`; eight examples per family.
- Candidates patched: 11 selected candidates, including top raw-positive attention heads, one top control-firing head, and deterministic comparison heads.
- Whether patching was head-specific or layer-level: not head-specific. All candidate rows were patched as `full_attn_out_layer` with `head_specific_patch=false`.
- Positive mean effect size: `0.0522`.
- Hardest control family by patching effect: `same_token_frequency_control`, with max control mean effect size `0.1755` on candidate `cand_010`.
- Best positive-minus-control causal gap: `0.5102` on `cand_009` (random comparison L9H6), not a raw attention candidate.
- Candidate specificity statuses: `positive_specific_candidate`: 6; `no_positive_effect`: 5.
- Did controls move as much as positives? Across the run, the max control mean effect (`0.1755`) exceeded the overall positive mean effect (`0.0522`). For the simple per-candidate rule, six candidates had positive effects larger than their max control effect, but the largest gap came from a random comparison candidate.
- What this teaches: controlled patching can separate descriptive attention false positives from causal practice candidates, but the first interesting causal gap did not come from the raw previous-occurrence attention heads.
- What this does not show: no induction-head mechanism was discovered. Layer-level `attn_out` patching is not head-specific, and the result is limited to selected prompts, candidates, component scope, position, and target-logit metric.
- Next step: write a short learning note, then run a small seed-1 replication because at least one candidate met the simple `positive_specific_candidate` rule.

## 2026-06-26 GPT-2 Small Controlled Patching Seed-1 Replication

- Run: `runs/20260626_150356_gpt2_small_induction_controls_seed1`
- Config: `configs/gpt2_small_induction_controls_seed1.yaml`
- Model: `gpt2-small`; run completed under the CUDA config without blockers.
- Families: same four-family patching subset as seed 0, with eight examples per family.
- Candidates patched: 11 selected candidates from the seed-1 attention artifacts.
- Whether patching was head-specific or layer-level: not head-specific. All patching was `full_attn_out_layer` with `head_specific_patch=false`.
- Positive mean effect size: `0.0127`.
- Max control mean effect size: `0.1397`.
- Best positive-minus-control causal gap: `0.0466` on `cand_008`, a random comparison L9H3 candidate.
- Candidate specificity statuses: `positive_specific_candidate`: 1; `nonspecific_moves_controls`: 4; `no_positive_effect`: 6.
- Replication result: the strong seed-0 gap did not replicate as a raw-attention-head result. The only seed-1 positive-specific candidate was again a random comparison head, and raw positive-attention heads L0H1/L0H5/L0H10/L0H4 were classified as `nonspecific_moves_controls`.
- What this teaches: the current raw attention candidates are still false-positive-prone, and a small causal gap on a random comparison head should be treated as a prompt/sample artifact until a narrower replication says otherwise.
- Next step: stop before adding a new task; inspect the two `controlled_patching_by_candidate.csv` files side by side and write down why raw attention did not survive the controlled causal check.

## 2026-06-26 GPT-2 Small Head Hook Inspection

- Run: `runs/20260626_151637_head_hook_inspection`
- Model: `gpt2-small`
- Prompt: `A B C D A B C`
- Result: head-specific patching is supported through `blocks.0.attn.hook_z`.
- Hook shapes: `hook_z` captured as `[1, 8, 12, 64]` with head axis 2 and sequence axis 1; `hook_result` captured as `[1, 8, 12, 768]`; `hook_attn_out` captured as `[1, 8, 768]` and is layer-level only.
- GPU: CUDA available on NVIDIA GeForce RTX 5060 Ti in the inspection artifact.
- Blockers: none. A prior failed attempt created an empty inspection run because the code initially used `.get` on `ActivationCache`; this was fixed before the recorded inspection.
- Next step: implement head-specific `hook_z` patching and ablation. Do not label `hook_attn_out` fallback as head-specific.

## 2026-06-26 Head-Specific Induction Causality Seed 0

- Run: `runs/20260626_152332_gpt2_small_head_specific_induction`
- Source run: `runs/20260626_144001_gpt2_small_induction_controls`
- Hook inspection: `runs/20260626_152313_head_hook_inspection`
- Head-specific hook used: `blocks.<layer>.attn.hook_z`
- Was patching truly head-specific? Yes. Artifacts record `head_specific_patch=true` and `actual_patch_scope=single_head_z`.
- Metric: `true_vs_control_logit_diff`
- Intervention: `head_clean_to_corrupt_patch`
- Families: `positive_repeat_sequence`, `distractor_repeat_control`, `random_expected_token_control`, and `same_token_frequency_control`
- Examples per family: 8
- Heads tested: 72 heads across layers `[0, 2, 4, 7, 9, 11]`
- Positive mean effect: `0.0033`
- Max control mean effect: `0.9617`
- Best positive-minus-control causal gap: `0.0884`
- Top heads by seed-0 gap: L7H7 `0.0884`, L9H9 `0.0772`, L7H11 `0.0715`, L11H4 `0.0646`, L0H8 `0.0414`
- Specificity statuses: `head_specific_positive_candidate`: 19; `nonspecific_moves_controls`: 21; `no_positive_effect`: 32
- Did raw attention heads survive? No. L0H1, L0H4, and L0H5 had `no_positive_effect`; L0H10 and L11H8 were `nonspecific_moves_controls`.
- Did random comparison heads survive? This seed tested all selected-layer heads, so random-comparison status will be evaluated in the multi-seed comparison rather than from seed 0 alone.
- What this shows: true head-specific `hook_z` patching works locally, and some heads have positive-minus-control gaps on seed 0 under the stricter metric.
- What this does not show: one seed is not enough. A seed-0 positive candidate is not a replicated induction head, and the raw previous-occurrence attention heads still failed this stricter causal check.
- Next step: run seed 1 and seed 2 head-specific replications.

## 2026-06-26 Head-Specific Induction Causality Seed 1

- Run: `runs/20260626_152840_gpt2_small_head_specific_induction_seed1`
- Source run: `runs/20260626_150356_gpt2_small_induction_controls_seed1`
- Head-specific hook used: `blocks.<layer>.attn.hook_z`
- Was patching truly head-specific? Yes. Artifacts record `head_specific_patch=true` and `actual_patch_scope=single_head_z`.
- Metric: `true_vs_control_logit_diff`
- Intervention: `head_clean_to_corrupt_patch`
- Families: `positive_repeat_sequence`, `distractor_repeat_control`, `random_expected_token_control`, and `same_token_frequency_control`
- Examples per family: 8
- Heads tested: 72 heads across layers `[0, 2, 4, 7, 9, 11]`
- Positive mean effect: `0.0025`
- Max control mean effect: `1.5706`
- Best positive-minus-control causal gap: `0.0893`
- Top heads by seed-1 gap: L7H7 `0.0893`, L7H3 `0.0819` but with `no_positive_effect`, L9H11 `0.0364`, L11H8 `0.0190`, L2H0 `0.0148`
- Specificity statuses: `head_specific_positive_candidate`: 14; `nonspecific_moves_controls`: 26; `no_positive_effect`: 32
- Did raw attention heads survive? Mostly no. L0H1, L0H4, and L0H5 had `no_positive_effect`; L0H10 was `nonspecific_moves_controls`; L11H8 was `head_specific_positive_candidate` for this seed.
- What this shows: the seed-1 replication again produced small head-specific positive-minus-control gaps for a subset of heads, with L7H7 matching the top seed-0 head.
- What this does not show: this still does not establish an induction head. Control effects remain large in aggregate, and a replicated-looking head needs the pre-registered multi-seed classification before any stronger interpretation.
- Next step: run seed 2 and then compare all seeds jointly.

## 2026-06-26 Head-Specific Induction Causality Seed 2

- Run: `runs/20260626_153044_gpt2_small_head_specific_induction_seed2`
- Source run: `runs/20260626_152722_gpt2_small_induction_controls_seed2`
- Generated source controls: `configs/gpt2_small_induction_controls_seed2.yaml` was used to build prompts, baseline behavior, selected activations, logit lens, and attention-pattern artifacts for seed 2.
- Head-specific hook used: `blocks.<layer>.attn.hook_z`
- Was patching truly head-specific? Yes. Artifacts record `head_specific_patch=true` and `actual_patch_scope=single_head_z`.
- Metric: `true_vs_control_logit_diff`
- Intervention: `head_clean_to_corrupt_patch`
- Families: `positive_repeat_sequence`, `distractor_repeat_control`, `random_expected_token_control`, and `same_token_frequency_control`
- Examples per family: 8
- Heads tested: 72 heads across layers `[0, 2, 4, 7, 9, 11]`
- Positive mean effect: `0.0022`
- Max control mean effect: `0.2855`
- Best positive-minus-control causal gap: `0.0641`
- Top heads by seed-2 gap: L7H7 `0.0641`, L11H4 `0.0608`, L11H8 `0.0422`, L9H11 `0.0328`, L11H6 `0.0314`
- Specificity statuses: `head_specific_positive_candidate`: 20; `nonspecific_moves_controls`: 19; `no_positive_effect`: 33
- Did raw attention heads survive? Mostly no. L0H1, L0H4, and L0H5 had `no_positive_effect`; L0H10 was `nonspecific_moves_controls`; L11H8 was `head_specific_positive_candidate` for this seed.
- What this shows: seed 2 again put L7H7 at the top under the stricter head-specific metric, while most original layer-0 raw-attention heads did not survive.
- What this does not show: seed-specific status is not a circuit claim. The consolidated report must compare controls and seeds before deciding whether any head is merely interesting enough for manual inspection.
- Next step: run the pre-registered multi-seed comparison and write the consolidated result.

## 2026-06-26 GPT-2 Small First Practice Loop

- Run: `runs/20260626_142215_gpt2_small_induction`
- Question: Does GPT-2 small show simple repeated-token induction behavior, and which selected heads show induction-like previous-occurrence attention?
- Model: `gpt2-small`
- Key artifacts: `baseline_by_example.csv`, `baseline_metrics.json`, `activations/manifest.json`, `logit_lens_summary.json`, `attention_patterns_by_head.csv`, `attention_summary.json`, `figures/logit_lens_expected_token.png`, `figures/attention_induction_scores.png`, `summary.md`
- Baseline result: 64 examples, mean expected-token probability `0.2849`, median expected-token rank `1.0`, mean probability diff versus control `0.2770`, and 64/64 examples rank <= 10.
- Logit lens: selected layers `[0, 2, 4, 7, 9, 11]`; best layer by mean expected-token probability was layer 9. This is descriptive only.
- Attention candidates: top previous-occurrence attention heads were L0H1 `0.537`, L0H5 `0.531`, L0H10 `0.244`, L11H8 `0.211`, and L0H4 `0.182`. These are induction-like attention pattern candidates, not identified induction heads.
- Patching practice run: `runs/20260626_142431_gpt2_small_clean_corrupt_tiny`; residual-stream final-position patching on the tiny clean/corrupt prompt pair produced effect sizes from `0.637` at layer 0 to `1.000` at layer 11.
- What changed: attention-pattern inspection now exists and the prompt generator records source-position metadata for induction practice.
- What did not change: no SAE, Gemma, nnsight, dashboard, database, or framework machinery was added.
- Limitation: attention-pattern evidence and logit-lens evidence are descriptive. Patching is causal only for the selected prompt pair, component, position, and metric. No broad mechanism claim is allowed.
- Blockers: none in this run.
- Next step: inspect `attention_patterns_by_head.csv` and compare the top attention candidates against a controlled prompt set before treating any head as worth causal follow-up.

## 2026-06-26 Held-Out Induction Robustness Seed 10

- Run: `runs/20260626_161445_gpt2_small_induction_heldout_seed10`
- Descriptive source run: `runs/20260626_161335_gpt2_small_induction_heldout_seed10`
- Candidate set: `reports/head_specific_induction_heldout_robustness_v1/heldout_candidate_set.csv`
- Prompt families: `heldout_symbolic_longer`, `heldout_word_sequences`, `heldout_number_sequences`, `heldout_double_repeat`, `heldout_wrong_target_same_prompt`, and `heldout_no_structure_same_tokens`
- Interventions: `head_clean_to_corrupt_patch`, `head_zero_ablation`, and `head_mean_ablation`
- Positions: `final` and `previous_occurrence`
- Examples per family: 12
- Primary metric: `true_vs_control_logit_diff`
- Candidates tested: 16 fixed candidates, including the five prior replicated candidates, prior raw-attention comparison heads, and deterministic negative controls.
- Survived seed: 10 candidate/intervention/position rows were classified as `heldout_survives_seed`.
- Downgraded: no rows used the weak-family-specific downgrade status in this seed.
- Falsified: 24 rows had `falsified_no_positive_effect`; 14 rows had `falsified_controls_move`; 48 rows had `insufficient_valid_examples`, mostly for position or denominator limitations.
- Controls-moving failures: 14 candidate/intervention/position rows had controls move as much as or more than positives.
- L7H7 status: survived seed 10 for final-position clean-to-corrupt patching, but remains a prior random-comparison candidate and is not interpretable without seed 11/12.
- L9H11 status: survived seed 10 only weakly for final clean-to-corrupt patching; the gap was near zero in that condition.
- L7H11 status: strongest seed-10 final clean-to-corrupt positive-minus-control gap among primary candidates; ablations did not support the same direction.
- L7H0 status: no positive effect under final clean-to-corrupt patching.
- L0H8 status: no positive effect under final clean-to-corrupt patching and controls moved in at least one condition.
- Prior raw-attention heads status: L0H10 survived seed 10 under final clean-to-corrupt patching, but this is a comparison head and does not rescue raw previous-occurrence attention without replication. Other prior raw-attention heads remained weak or failed.
- Negative controls status: at least one negative-control head also satisfied the seed-level survival rule, so seed 10 alone is not specific enough.
- Implementation note: an earlier robustness attempt, `runs/20260626_160547_gpt2_small_induction_heldout_seed10`, was rejected because positive held-out examples used identical clean/corrupt prompts and produced denominator-zero effects. The prompt pairing and clean/corrupt construction were fixed before this recorded run.
- What this shows: the held-out runner can execute true head-specific `hook_z` interventions across prompt, intervention, and position variants, and seed 10 produces mixed candidate effects.
- What this does not show: seed 10 does not establish a robust candidate. Negative controls and prior comparison heads also showed apparent survival under the seed-level rule, so the multi-seed report must be conservative.
- Next step: run held-out robustness seeds 11 and 12 without changing decision rules.

## 2026-06-26 Held-Out Induction Robustness Seed 11

- Run: `runs/20260626_162154_gpt2_small_induction_heldout_seed11`
- Baseline run: `runs/20260626_162134_gpt2_small_induction_heldout_seed11`
- Candidate set: `reports/head_specific_induction_heldout_robustness_v1/heldout_candidate_set.csv`
- Prompt families: `heldout_symbolic_longer`, `heldout_word_sequences`, `heldout_number_sequences`, `heldout_double_repeat`, `heldout_wrong_target_same_prompt`, and `heldout_no_structure_same_tokens`
- Interventions: `head_clean_to_corrupt_patch`, `head_zero_ablation`, and `head_mean_ablation`
- Positions: `final` and `previous_occurrence`
- Examples per family: 12
- Primary metric: `true_vs_control_logit_diff`
- Candidates tested: 16 fixed candidates.
- Status counts: `heldout_survives_seed`: 13; `downgraded_weak_family_specific`: 1; `falsified_no_positive_effect`: 24; `falsified_controls_move`: 10; `insufficient_valid_examples`: 48.
- L7H7 status: final clean-to-corrupt patching had a small positive gap but was classified `falsified_no_positive_effect`; final zero ablation was classified `heldout_survives_seed`. This remains a random-comparison candidate until the multi-seed report downgrades or confirms it.
- L9H11 status: final mean ablation was classified `heldout_survives_seed`; final clean-to-corrupt and zero ablation were control-moving or no-positive-effect failures.
- L7H11 status: final zero ablation had a large positive-minus-control gap but still had `falsified_no_positive_effect`; clean-to-corrupt and mean ablation were negative.
- L7H0 status: final clean-to-corrupt was `falsified_no_positive_effect`; ablations were also no-positive-effect.
- L0H8 status: final clean-to-corrupt and mean ablation were classified `heldout_survives_seed`; zero ablation was no-positive-effect.
- Prior raw-attention heads status: L0H1 and L0H5 satisfied some final-position seed-level survival rows; L0H10 had a large zero-ablation gap but no positive effect. This again shows the seed-level rule is not specific enough by itself.
- Negative controls status: negative-control L11H0 had the largest positive-minus-control gap in the seed under final zero ablation, so negative controls clearly expose false-positive risk.
- Descriptive artifacts skipped: cache, logit-lens, and attention were not run for this seed because the robustness runner and planned consolidation use the causal robustness artifacts directly.
- What this shows: seed 11 completed the full fixed matrix with true head-specific `hook_z` interventions.
- What this does not show: seed 11 does not support a clean candidate. Comparison heads and negative controls also produced apparent survival rows.
- Next step: run seed 12 with the same fixed candidate set and unchanged decision rules.

## 2026-06-26 Held-Out Induction Robustness Seed 12

- Run: `runs/20260626_162753_gpt2_small_induction_heldout_seed12`
- Baseline run: `runs/20260626_162728_gpt2_small_induction_heldout_seed12`
- Candidate set: `reports/head_specific_induction_heldout_robustness_v1/heldout_candidate_set.csv`
- Prompt families: `heldout_symbolic_longer`, `heldout_word_sequences`, `heldout_number_sequences`, `heldout_double_repeat`, `heldout_wrong_target_same_prompt`, and `heldout_no_structure_same_tokens`
- Interventions: `head_clean_to_corrupt_patch`, `head_zero_ablation`, and `head_mean_ablation`
- Positions: `final` and `previous_occurrence`
- Examples per family: 12
- Primary metric: `true_vs_control_logit_diff`
- Candidates tested: 16 fixed candidates.
- Status counts: `heldout_survives_seed`: 14; `falsified_no_positive_effect`: 25; `falsified_controls_move`: 9; `insufficient_valid_examples`: 48.
- L7H7 status: final zero ablation was classified `heldout_survives_seed`; final clean-to-corrupt patching had no positive effect and mean ablation was control-moving. This is intervention-specific and should be downgraded unless the consolidated report finds stronger support.
- L9H11 status: final clean-to-corrupt, zero ablation, and mean ablation were all classified `heldout_survives_seed`, with the largest primary-candidate seed-12 gap under zero ablation.
- L7H11 status: no final-position intervention produced a survival classification.
- L7H0 status: no final-position intervention produced a survival classification.
- L0H8 status: final clean-to-corrupt and mean ablation were classified `heldout_survives_seed`; zero ablation was no-positive-effect.
- Prior raw-attention heads status: L0H1, L0H4, L0H5, and L0H10 all satisfied at least one seed-level survival row. This reinforces that prior raw-attention heads can still pass permissive held-out rows without being specific induction candidates.
- Negative controls status: negative controls again produced survival rows, including L11H0 for final clean-to-corrupt patching, while L11H0 also had the largest raw gap under zero ablation despite being no-positive-effect by status.
- Descriptive artifacts skipped: cache, logit-lens, and attention were not run for this seed because the robustness runner and planned consolidation use the causal robustness artifacts directly.
- What this shows: seed 12 completed the full fixed held-out robustness matrix.
- What this does not show: seed 12 alone does not validate the prior candidates. Apparent survival remains mixed across interventions and is not specific to primary candidates.
- Next step: run the consolidated held-out robustness comparison across seeds 10, 11, and 12.

## 2026-06-26 Held-Out Induction Robustness Consolidated Result

- Runs: `runs/20260626_161445_gpt2_small_induction_heldout_seed10`, `runs/20260626_162154_gpt2_small_induction_heldout_seed11`, and `runs/20260626_162753_gpt2_small_induction_heldout_seed12`
- Candidate-set artifact: `reports/head_specific_induction_heldout_robustness_v1/heldout_candidate_set.csv`
- Consolidated report: `reports/head_specific_induction_heldout_robustness_v1/head_specific_induction_heldout_robustness_v1.md`
- Consolidated summary: `reports/head_specific_induction_heldout_robustness_v1/heldout_multiseed_summary.json`
- Tracked result summary: `docs/results/head_specific_induction_heldout_robustness_v1.md`
- Counterexample artifacts: `reports/head_specific_induction_heldout_robustness_v1/counterexamples_L7H7.md`, `reports/head_specific_induction_heldout_robustness_v1/counterexamples_L9H11.md`, and `reports/head_specific_induction_heldout_robustness_v1/counterexamples_L7H11.md`
- Replicated/downgraded/falsified statuses: `heldout_replicated`: 0; `heldout_downgraded`: 5; `heldout_falsified`: 11.
- L7H7 status: `heldout_falsified`; it had positive examples but also controls moved and the aggregate mean gap was negative.
- L9H11 status: `heldout_falsified`; it had seed-level survival rows but controls-moving and negative aggregate gaps broke the robustness story.
- L7H11 status: `heldout_downgraded`; it had a strong seed-10 clean-to-corrupt result but did not replicate robustly across held-out seeds and variants.
- L7H0 status: `heldout_falsified`.
- L0H8 status: `heldout_downgraded`; it had weak final-position support and a near-zero aggregate gap.
- Negative-control result: negative controls also produced apparent seed-level survival rows, including L11H0 surviving three seed-level rows while still being classified `heldout_falsified`. This shows the permissive seed-level rule is not specific enough.
- Prior raw-attention result: L0H1 and L0H5 were downgraded; L0H10, L0H4, and L11H8 were falsified. Raw previous-occurrence attention still does not provide robust causal evidence.
- Interpretation: the held-out robustness check falsified or downgraded the previously replicated candidates. This is a useful negative result because the pipeline caught its own false-positive-prone candidates.
- What this does not show: no induction-head discovery, circuit, or broad GPT-2 behavior claim is supported.
- Next step: pre-register candidate characterization without adding new candidates, then test attention/effect alignment, position sensitivity, OV/QK diagnostics, token-domain robustness, and counterexamples.

## Template

- Date:
- Run:
- Question:
- Model:
- Key artifact:
- What changed:
- What did not change:
- Next step:
