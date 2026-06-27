# Experiment Log

Add dated entries after script-generated artifacts exist. Keep entries short and link to the relevant run directory.

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
