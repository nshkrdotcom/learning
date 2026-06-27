# Learning Path

This is a six-stage path for local mechanistic-interpretability practice. Keep each stage small, script-first, and honest about what the artifacts show.

## Stage 1: Tokens, Logits, Residual Stream, Hooks

- What to learn: how prompts become token IDs, how final-position logits predict the next token, what a residual stream activation is, and how TransformerLens hook names are structured.
- Script/notebook to run: `scripts/check_model_capability.py`, `scripts/build_toy_prompts.py`, and `notebooks/001_model_and_tokens.ipynb`.
- Result to inspect: tokenizer examples, expected token IDs, one-token forward pass, and selected-layer cache shape.
- Mistake to avoid: treating tokenizer strings as obvious natural-language words; leading spaces often matter.
- Completion: you can explain which token is being scored and where the activation tensor came from.

## Stage 2: Logit Lens / Direct Logit Attribution

- What to learn: how residual stream states at different layers map into vocabulary logits under a simple logit-lens projection.
- Script/notebook to run: `scripts/run_logit_lens.py` and `notebooks/002_logit_lens.ipynb`.
- Result to inspect: `logit_lens_by_layer.csv`, `logit_lens_summary.json`, and `figures/logit_lens_expected_token.png`.
- Mistake to avoid: claiming a layer "contains the answer" from a projection alone.
- Completion: you can identify where the expected-token logit rises or falls and state that this is descriptive evidence only.

## Stage 3: Induction Heads

- What to learn: repeated-token prompts, copy/repetition behavior, attention patterns that can support induction, and the difference between behavior and mechanism.
- Script/notebook to run: `scripts/run_baseline_behavior.py`, `scripts/cache_activations.py`, and `notebooks/003_induction_heads.ipynb`.
- Result to inspect: `baseline_by_example.csv`, selected activation files, and prompt failures.
- Mistake to avoid: selecting only successful prompts and ignoring controls.
- Completion: you have a baseline table showing whether GPT-2 small predicts repeated tokens more strongly than controls.

## Stage 4: Activation Patching on a Known Task

- What to learn: clean/corrupt prompt pairs, causal interventions, target metrics, and effect-size normalization.
- Script/notebook to run: `scripts/run_activation_patching.py` and `notebooks/004_activation_patching.ipynb`.
- Result to inspect: `patching_results.csv`, `patching_heatmap.csv`, and `figures/patching_heatmap.png`.
- Mistake to avoid: patching many sites first and then inventing a story after seeing a heatmap.
- Completion: you can state which site was patched, what metric moved, and what the intervention does not prove.

## Stage 5: Controls and False Positives

- What to learn: prompt controls, random or irrelevant target tokens, corrupted pairs, seed sensitivity, and failure cases.
- Script/notebook to run: rerun baseline, cache, logit lens, and patching scripts with controlled prompt sets.
- Result to inspect: examples where the expected token is not preferred, and controls where patching should not help.
- Mistake to avoid: interpreting every large-looking number as meaningful without a counterexample.
- Completion: at least one negative control behaves as expected and is reported.

## Stage 6: Small Original Question

- What to learn: how to ask one narrow question that can be tested locally with behavior, descriptive analysis, and a causal intervention.
- Script/notebook to run: adapt the existing scripts rather than creating a new framework.
- Result to inspect: one short Markdown run summary plus the CSV/JSON/PNG artifacts needed to audit it.
- Mistake to avoid: starting with a broad claim about how the model works.
- Completion: a short report says what was tested, what moved, what did not move, and what remains unknown.
