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

## Template

- Date:
- Run:
- Question:
- Model:
- Key artifact:
- What changed:
- What did not change:
- Next step:
