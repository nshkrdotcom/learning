# Prior-Art Matrix

This becomes the Related Work section. Build the table first, write prose
last. Every row below is a working summary for triage — verify exact claims
against the actual paper before anything gets cited in the paper draft. Treat
arXiv IDs and dates as leads to re-check, not confirmed facts; several were
collected by an automated literature search and have not been independently
read in full.

Legend: SE = self-explanation evaluated, MT = mechanistic/causal test used,
SAE = sparse-autoencoder features used, AP = activation patching / causal
intervention used, NEC = necessity-style metric, SUF = sufficiency-style
metric, PRE = pre-intervention prediction (predicts effect before
intervening), TRAIN = trains the model on this objective (vs. eval-only).

## Closest prior art (read these first, in full)

| Paper | Year | SE | MT | SAE | AP | NEC | SUF | PRE | TRAIN | Gap vs. SELF-GROUND |
|---|---|---|---|---|---|---|---|---|---|---|
| Li, Guo, Huang, Steinhardt, Andreas — *Training Language Models to Explain Their Own Computations* | 2025 | Y | Y | partial | Y | ? | ? | partial | Y | Trains models to explain internal features/causal structure directly — closest "self-report" analogue. Need to check whether claims are *typed* (feature-ID + predicted intervention effect) or free text describing causal structure in prose. |
| NeuroFaith (Bhan et al.) | 2025 | Y | Y | partial | Y | ? | ? | N | N | Extracts concepts from free-text self-explanations, then mechanistically tests whether those concepts are causally influential. Evaluates post-hoc NL explanations; does not require a structured, typed claim format or score pre-intervention prediction accuracy. |
| Duan — *Pre-Intervention Prediction of Sparse Autoencoder Steering Side Effects* | 2026 | N | Y | Y | Y (steering) | N | N | Y | ? | Predicts SAE steering side effects *before* intervening, using decoder geometry / co-activation / direct-logit features — not from a model self-report. Strongest existing analogue to the "predict-before-you-patch" framing, but the predictor is an external estimator, not the model's own structured claim. |
| Mahale — *Causally Grounded Mechanistic Interpretability for LLMs with Faithful Natural-Language Explanations* | 2026 | Y | Y | ? | Y | ? | Y (sufficiency/comprehensiveness-style) | N | N | Activation patching -> causal heads -> NL explanation -> sufficiency/comprehensiveness eval. Very close in spirit; main gap is NL explanation format rather than a typed feature-claim format, and no pre-intervention prediction framing. |
| Marks et al. — *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in LMs* | 2024 | N | Y | Y | Y | partial | partial | N | N | Foundational for "claim graph over SAE features validated causally," but the graph is discovered by the *researchers*, not reported by the *model*, and there's no self-report-faithfulness framing at all. |
| Zhang & Nanda — *Towards Best Practices of Activation Patching in LMs: Metrics and Methods* | 2023/24 | N | N | N | Y | N | N | N | N | Methodological, not a self-report paper — but essential for justifying any necessity/sufficiency metric built on patching. Cite for metric-choice rigor, not as a comparable system. |
| Geiger et al. — *Causal Abstraction: A Theoretical Foundation for MI* | 2023 / JMLR 2025 | N | Y | N | Y (interchange interventions) | N | N | N | N | Formal framework for mapping high-level causal claims to low-level mechanisms. Useful for *defining* what a "structured claim" should formally commit to, not a competing system. |
| McCann — *Descriptive Collision in SAE Auto-Interpretability* | 2026 | N | N | Y | N | N | N | N | N | Important caution, not competing work: many SAE features can share one plausible-sounding explanation. Motivates why SELF-GROUND needs a *discriminative* test (does the claim predict an intervention outcome) rather than a plausibility check. |

## Faithfulness / CoT-faithfulness adjacent

| Paper | Year | Relevance |
|---|---|---|
| Turpin et al. — *LMs Don't Always Say What They Think* | 2023 | Foundational negative result: CoT explanations can rationalize biased outputs without naming the real cause. Motivates why text-only self-explanation evaluation is insufficient on its own. |
| Lanham et al. — *Measuring Faithfulness in CoT Reasoning* | 2023 | Intervenes on stated reasoning to test output dependence — closest CoT-side analogue to "does the claim predict the intervention outcome." |
| Madsen, Chandar, Reddy — *Are self-explanations from LLMs faithful?* | 2024 | Tests counterfactual / feature-attribution / redaction self-explanations; finds faithfulness is model- and task-dependent. Useful baseline framing for "faithfulness is not a fixed property of a model." |
| Jia, Benton, Easley — *Faithfulness as Information Flow* | 2026 | Formalizes faithfulness via information flow prompt -> CoT -> answer, with explicit sufficiency/necessity/completeness diagnostics. Closest *framework-level* analogue outside the SAE literature — worth comparing definitions side by side. |
| Paul, West, Bosselut, Faltings — *Making Reasoning Matter* (FRODO) | 2024 | Causal mediation analysis + training to improve reliance on intermediate reasoning. Relevant if SELF-GROUND ever moves from eval-only to training. |

## SAE validity / steering reliability (recency-heavy)

| Paper | Year | Relevance |
|---|---|---|
| Chanin — *Are Sparse Autoencoder Benchmarks Reliable?* | 2026 | Audits SAEBench-style metrics; relevant to choosing which SAE quality signals (if any) belong in the mechanism report. |
| McCann — *Descriptive Collision...* | 2026 | See above. |
| Dalili & Mahdavi — *Subspace-Aware SAEs* | 2026 | Argues single-direction SAE latents can fragment multidimensional concepts — relevant caveat for treating one `sae_N` ID as one concept. |
| "When Attribution Patching Lies..." | 2026 | Caution against using attribution patching as a cheap proxy for real activation patching — relevant if Phase 3+ ever needs to scale beyond exact patching for cost reasons. |
| Zhang & Nanda | 2023/24 | See above (methodology). |

## Causal abstraction / circuit discovery, broader context

| Paper | Year | Relevance |
|---|---|---|
| Geiger et al. — *Causal Abstraction* | 2023/2025 | See above. |
| Meng et al. — *Locating and Editing Factual Associations in GPT* (ROME) | 2022 | Classic causal tracing + editing; useful as the "what came before circuit-level claim graphs" anchor. |
| Méloux, Peyrard, Portet — *MI as Statistical Estimation: Variance Analysis of EAP-IG* | 2025 | Circuit-discovery methods can have high variance; relevant caution for any future move from manual feature selection to automated circuit discovery. |

## What this matrix is for, concretely

The wedge is whichever cell stays empty across this whole table. As of this
pass, the most defensible empty cell is: **a model emits a typed, structured
claim over named internal features (not free text) and that claim is scored
by whether it predicts the necessity/sufficiency/sign of a real activation
intervention — and the scoring is run against deterministic control feature
sets, not just "did the explanation sound right."** Re-derive this after
actually reading the top 5 papers in full; don't trust this paragraph as a
substitute for reading them.

## Next actions

- [ ] Read Li et al. 2025 and Duan 2026 in full; confirm or correct the Y/N/?
      cells above.
- [ ] Confirm whether Mahale 2026 already does something close enough to
      narrow SELF-GROUND's claimed novelty — this is the single biggest risk
      to the current framing.
- [ ] Re-verify every arXiv ID before the paper goes anywhere near a draft
      submission; several entries above came from an automated search pass
      and carry an explicit "verify" flag in the original research notes.
