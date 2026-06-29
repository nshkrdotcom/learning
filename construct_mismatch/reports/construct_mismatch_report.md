# Construct Mismatch Report

## 1. Abstract
This report tests whether direction, probe, and activation-patching methods appear to disagree because they operationalize different validity targets. The main artifact is a construct mismatch matrix over certainty/uncertainty and sentiment in GPT-2 Small.

## 2. Hypothesis
Method success and failure should be predictable from construct-validity mismatches. When lexical cues, semantic stance, target-token behavior, and causal intervention are decoupled, methods that look successful on ordinary examples should fail in different ways.

## 3. Prior-art Positioning
Sentiment analysis in GPT-style models is already well studied. Activation patching, linear probes, diff-in-means directions, sentiment steering, GPT-2 sentiment analysis, and the observation that probes can be predictive but non-causal are not claimed as novel here.

## 4. Why This Is Not Merely a Method Benchmark
The evaluation varies the operationalization of the construct: ordinary examples, lexical reversals, negation, quotation, contrast, format shifts, causal steering, and specificity. The comparison is about construct validity, not a single leaderboard score.

## 5. Why Sentiment Is a Baseline and Certainty Is Core
Sentiment is included as a familiar sanity check for the pipeline. Certainty/uncertainty is the stronger target construct because it is less saturated and more directly stresses the distinction between endorsed stance, lexical cue, and target-token behavior.

## 6. Model and Hardware
All experiments use GPT-2 Small through TransformerLens. Hardware is whatever device TransformerLens selected locally (`cuda`, `mps`, or `cpu`); no additional models are included.

## 7. Tokenization Constraints
GPT-2 leading spaces are part of the target tokens. Dataset construction validates target strings as single tokens before writing JSONL records.

| construct | raw_string | token_ids | n_tokens | usable_as_target |
| --- | --- | --- | --- | --- |
| certainty |  certain | 1728 | 1 | True |
| certainty |  uncertain | 8627 | 1 | True |
| certainty |  sure | 1654 | 1 | True |
| certainty |  unsure | 22147 | 1 | True |
| certainty |  definite | 21892 | 1 | True |
| certainty |  doubtful | 31608 | 1 | True |
| certainty |  confident | 6563 | 1 | True |
| certainty |  hesitant | 32848 | 1 | True |
| certainty |  clear | 1598 | 1 | True |
| certainty |  unclear | 10061 | 1 | True |
| certainty |  obvious | 3489 | 1 | True |
| certainty |  ambiguous | 27102 | 1 | True |
| sentiment |  great | 1049 | 1 | True |
| sentiment |  terrible | 7818 | 1 | True |
| sentiment |  good | 922 | 1 | True |
| sentiment |  bad | 2089 | 1 | True |
| sentiment |  excellent | 6275 | 1 | True |
| sentiment |  awful | 12659 | 1 | True |
| sentiment |  amazing | 4998 | 1 | True |
| sentiment |  poor | 3595 | 1 | True |
| sentiment |  wonderful | 7932 | 1 | True |
| sentiment |  disappointing | 17185 | 1 | True |

## 8. Dataset Design
Each construct has train, ordinary heldout, and decoupling splits. Records include paired class-A/class-B examples for patching where possible. The datasets are small and manually templated to prioritize interpretability over coverage.

## 9. Dataset Examples by Decoupling Axis
### certainty
- `lexical_reversal`: The memo mentioned "uncertain" early on, but the verified conclusion is ->  certain/ uncertain
- `negation`: The result was not uncertain; after the audit it was ->  certain/ uncertain
- `quotation`: A critic called the conclusion "uncertain", but the panel itself found it ->  certain/ uncertain
- `contrast`: At first the answer seemed uncertain, but after checking the details it was ->  certain/ uncertain
- `format_shift`: Conclusion:
Evidence status: verified
Confidence level: ->  certain/ uncertain
### sentiment
- `lexical_reversal`: The review quoted the word "terrible", but my final judgment was ->  great/ terrible
- `negation`: The movie was not terrible; it was ->  great/ terrible
- `quotation`: A critic called it "terrible", but I found it ->  great/ terrible
- `contrast`: Although people said it was terrible, I thought it was ->  great/ terrible
- `format_shift`: Review note: audience applauded
Final judgment: ->  great/ terrible

## 10. Behavior Check Results
| construct | split | decoupling_axis | n | accuracy | mean_signed_logit_diff | behavior_status |
| --- | --- | --- | --- | --- | --- | --- |
| certainty | decoupling | contrast | 12 | 0.75 | 0.6281798680623373 | usable |
| certainty | decoupling | format_shift | 12 | 0.5833333333333334 | 0.3093613982200622 | usable |
| certainty | decoupling | lexical_reversal | 12 | 0.5 | 0.1682002544403076 | behavior_absent_or_weak |
| certainty | decoupling | negation | 12 | 0.6666666666666666 | -0.1623414357503255 | behavior_absent_or_weak |
| certainty | decoupling | quotation | 12 | 0.5 | 0.1840436458587646 | behavior_absent_or_weak |
| certainty | heldout | ordinary | 36 | 0.6388888888888888 | 0.9092702865600586 | usable |
| certainty | train | ordinary | 60 | 0.6166666666666667 | 0.6284661372502645 | usable |
| sentiment | decoupling | contrast | 12 | 0.4166666666666667 | -0.4627400239308675 | behavior_absent_or_weak |
| sentiment | decoupling | format_shift | 12 | 0.5 | 0.5936665534973145 | behavior_absent_or_weak |
| sentiment | decoupling | lexical_reversal | 12 | 0.5833333333333334 | 0.4897017478942871 | usable |
| sentiment | decoupling | negation | 12 | 0.75 | 0.8223810990651449 | usable |
| sentiment | decoupling | quotation | 12 | 0.6666666666666666 | 0.658262292544047 | usable |
| sentiment | heldout | ordinary | 36 | 0.5555555555555556 | 0.6294815672768487 | usable |
| sentiment | train | ordinary | 60 | 0.65 | 0.966563606262207 | usable |

Strong disagreements or weak margins:
| id | construct | decoupling_axis | signed_logit_diff | flag | prompt |
| --- | --- | --- | --- | --- | --- |
| certainty_decoupling_contrast_005_class_b | certainty | contrast | -5.996954917907715 | strong_model_disagreement | The puzzle initially looked obvious, but after the contradiction it was |
| certainty_decoupling_format_shift_005_class_b | certainty | format_shift | -5.257821083068848 | strong_model_disagreement | Problem sheet<br>Without the answer key, the next step is |
| certainty_train_ordinary_002_class_b | certainty | ordinary | -5.225475311279297 | strong_model_disagreement | With measurements varying across trials, the conclusion is |
| certainty_decoupling_contrast_002_class_b | certainty | contrast | -5.103840827941895 | strong_model_disagreement | The opening paragraph sounded clear, but the appendix made the rule |
| certainty_heldout_ordinary_010_class_b | certainty | ordinary | -5.020306587219238 | strong_model_disagreement | With one premise unstated, the inference is |
| certainty_decoupling_negation_005_class_b | certainty | negation | -4.9258317947387695 | strong_model_disagreement | The solution was not obvious after the contradiction; it was |
| sentiment_decoupling_quotation_005_class_a | sentiment | quotation | -4.65622091293335 | strong_model_disagreement | The message repeated "disappointing", but the visit was |
| certainty_decoupling_format_shift_002_class_b | certainty | format_shift | -4.641802787780762 | strong_model_disagreement | Evidence status: checks conflict<br>The conclusion is |
| certainty_train_ordinary_012_class_b | certainty | ordinary | -4.366913795471191 | strong_model_disagreement | The instructions allow several options, so the choice is |
| certainty_decoupling_quotation_003_class_a | certainty | quotation | -4.169625282287598 | strong_model_disagreement | One blogger called the outcome "doubtful", but the official count made it |
| certainty_decoupling_quotation_005_class_b | certainty | quotation | -4.136313438415527 | strong_model_disagreement | The article quoted the word "obvious", but the final diagram made it |
| sentiment_decoupling_format_shift_006_class_b | sentiment | format_shift | -4.077325820922852 | strong_model_disagreement | Product note: defects returned<br>Customer reaction: |

## 11. Direction Results
### certainty
| baseline_type | split | decoupling_axis | layer | accuracy | mean_signed_projection |
| --- | --- | --- | --- | --- | --- |
| direction | train | ordinary | 0 | 0.6666666666666666 | 0.3885679853459199 |
| direction | train | ordinary | 1 | 0.6833333333333333 | 0.4526666708290577 |
| direction | train | ordinary | 2 | 0.65 | 0.6148707884053389 |
| direction | train | ordinary | 3 | 0.6666666666666666 | 0.8822981429596742 |
| direction | train | ordinary | 4 | 0.6166666666666667 | 1.1224711080392202 |
| direction | train | ordinary | 5 | 0.6833333333333333 | 1.4052687083681423 |
| direction | train | ordinary | 6 | 0.75 | 2.4321094272037347 |
| direction | train | ordinary | 7 | 0.75 | 3.425647475322088 |
| direction | train | ordinary | 8 | 0.8166666666666667 | 4.826433863242468 |
| direction | train | ordinary | 9 | 0.7833333333333333 | 6.739457738399506 |
| direction | train | ordinary | 10 | 0.7666666666666667 | 8.673081963260968 |
| direction | train | ordinary | 11 | 0.7 | 11.069556935628254 |
### sentiment
| baseline_type | split | decoupling_axis | layer | accuracy | mean_signed_projection |
| --- | --- | --- | --- | --- | --- |
| direction | train | ordinary | 0 | 0.5833333333333334 | 0.9591385463873544 |
| direction | train | ordinary | 1 | 0.55 | 1.1467033772418895 |
| direction | train | ordinary | 2 | 0.6833333333333333 | 1.549284310018023 |
| direction | train | ordinary | 3 | 0.7333333333333333 | 2.115068203707536 |
| direction | train | ordinary | 4 | 0.8 | 2.6673546895384788 |
| direction | train | ordinary | 5 | 0.8 | 3.3553797867149115 |
| direction | train | ordinary | 6 | 0.8666666666666667 | 4.543582183122635 |
| direction | train | ordinary | 7 | 0.8333333333333334 | 6.094395272930464 |
| direction | train | ordinary | 8 | 0.85 | 7.80354962448279 |
| direction | train | ordinary | 9 | 0.8333333333333334 | 10.60863128999869 |
| direction | train | ordinary | 10 | 0.8333333333333334 | 13.08279258410136 |
| direction | train | ordinary | 11 | 0.7166666666666667 | 15.674469590187073 |

## 12. Probe Results
### certainty
| split | decoupling_axis | layer | accuracy | direction_accuracy_reference |
| --- | --- | --- | --- | --- |
| train | ordinary | 0 | 1.0 | 0.6666666666666666 |
| train | ordinary | 1 | 1.0 | 0.6833333333333333 |
| train | ordinary | 2 | 1.0 | 0.65 |
| train | ordinary | 3 | 1.0 | 0.6666666666666666 |
| train | ordinary | 4 | 1.0 | 0.6166666666666667 |
| train | ordinary | 5 | 1.0 | 0.6833333333333333 |
| train | ordinary | 6 | 1.0 | 0.75 |
| train | ordinary | 7 | 1.0 | 0.75 |
| train | ordinary | 8 | 1.0 | 0.8166666666666667 |
| train | ordinary | 9 | 1.0 | 0.7833333333333333 |
| train | ordinary | 10 | 1.0 | 0.7666666666666667 |
| train | ordinary | 11 | 1.0 | 0.7 |
### sentiment
| split | decoupling_axis | layer | accuracy | direction_accuracy_reference |
| --- | --- | --- | --- | --- |
| train | ordinary | 0 | 1.0 | 0.5833333333333334 |
| train | ordinary | 1 | 1.0 | 0.55 |
| train | ordinary | 2 | 1.0 | 0.6833333333333333 |
| train | ordinary | 3 | 1.0 | 0.7333333333333333 |
| train | ordinary | 4 | 1.0 | 0.8 |
| train | ordinary | 5 | 1.0 | 0.8 |
| train | ordinary | 6 | 1.0 | 0.8666666666666667 |
| train | ordinary | 7 | 1.0 | 0.8333333333333334 |
| train | ordinary | 8 | 1.0 | 0.85 |
| train | ordinary | 9 | 1.0 | 0.8333333333333334 |
| train | ordinary | 10 | 1.0 | 0.8333333333333334 |
| train | ordinary | 11 | 1.0 | 0.7166666666666667 |

## 13. Patching Results
### certainty
| pair_id | decoupling_axis | top_layer | top_position | top_recovery | axis_top_site_stability |
| --- | --- | --- | --- | --- | --- |
| certainty_decoupling_contrast_001 | contrast | 6 | 7 | 1.142087370383012 | 0.5 |
| certainty_decoupling_contrast_002 | contrast | 6 | 6 | 1.2746268806772196 | 0.5 |
| certainty_decoupling_format_shift_001 | format_shift | 11 | 11 | 1.0 | 0.5 |
| certainty_decoupling_format_shift_002 | format_shift | 4 | 8 | -10.224888438254617 | 0.5 |
| certainty_decoupling_lexical_reversal_001 | lexical_reversal | 1 | 12 | 5.946460711725542 | 0.5 |
| certainty_decoupling_lexical_reversal_002 | lexical_reversal | 6 | 12 | -0.9114546972783752 | 0.5 |
| certainty_decoupling_negation_001 | negation | 2 | 4 | 1.3189986968512288 | 0.5 |
| certainty_decoupling_negation_002 | negation | 3 | 4 | 1.520593189625108 | 0.5 |
| certainty_decoupling_quotation_001 | quotation | 3 | 13 | 75.24043286954453 | 0.5 |
| certainty_decoupling_quotation_002 | quotation | 10 | 13 | 3.1877412492355672 | 0.5 |
| certainty_heldout_ordinary_001 | ordinary | 1 | 7 | 0.7291798019983237 | 0.5 |
| certainty_heldout_ordinary_002 | ordinary | 0 | 3 | 0.8944298739862556 | 0.5 |
### sentiment
| pair_id | decoupling_axis | top_layer | top_position | top_recovery | axis_top_site_stability |
| --- | --- | --- | --- | --- | --- |
| sentiment_decoupling_contrast_001 | contrast | 0 | 8 | -1.9659874598493372 | 0.5 |
| sentiment_decoupling_contrast_002 | contrast | 6 | 5 | -3.657623621945716 | 0.5 |
| sentiment_decoupling_format_shift_001 | format_shift | 1 | 6 | 3.773163867542885 | 0.5 |
| sentiment_decoupling_format_shift_002 | format_shift | 11 | 9 | 1.0 | 0.5 |
| sentiment_decoupling_lexical_reversal_001 | lexical_reversal | 0 | 8 | 2.3399828237366203 | 0.5 |
| sentiment_decoupling_lexical_reversal_002 | lexical_reversal | 9 | 13 | 1.4170770971321145 | 0.5 |
| sentiment_decoupling_negation_001 | negation | 9 | 7 | 1.0190784424798802 | 0.5 |
| sentiment_decoupling_negation_002 | negation | 11 | 7 | 1.0 | 0.5 |
| sentiment_decoupling_quotation_001 | quotation | 9 | 10 | 2.488664799968351 | 0.5 |
| sentiment_decoupling_quotation_002 | quotation | 10 | 11 | 1.0532888768640682 | 0.5 |
| sentiment_heldout_ordinary_001 | ordinary | 11 | 3 | 1.0 | 0.5 |
| sentiment_heldout_ordinary_002 | ordinary | 11 | 4 | 1.0 | 0.5 |

## 14. Construct Mismatch Matrix
| construct | method | causal_steering | contrast | format_shift | lexical_reversal | negation | ordinary_heldout | quotation | specificity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| certainty | direction | weak | weak | pass | weak | weak | pass | weak | pass |
| certainty | patching | not_applicable | pass | pass | pass | pass | pass | pass | weak |
| certainty | probe | not_applicable | fail | weak | fail | fail | pass | fail | not_applicable |
| sentiment | direction | fail | fail | weak | fail | pass | pass | weak | pass |
| sentiment | patching | not_applicable | pass | pass | pass | pass | pass | pass | weak |
| sentiment | probe | not_applicable | fail | weak | fail | fail | pass | fail | not_applicable |

Object classifications:
| construct | method | object_classification |
| --- | --- | --- |
| certainty | direction | robust_construct_variable |
| certainty | probe | predictive_noncausal_detector |
| certainty | patching | prompt_local_dependency |
| sentiment | direction | ordinary_only_proxy |
| sentiment | probe | predictive_noncausal_detector |
| sentiment | patching | prompt_local_dependency |

## 15. Failure-mode Taxonomy
- `ordinary_only_proxy`: ordinary heldout passes, but one or more decoupling axes fail.
- `causal_but_nonspecific_handle`: steering moves the target logits but causes high KL collateral disruption.
- `predictive_noncausal_detector`: a probe predicts class information without supporting a causal claim.
- `prompt_local_dependency`: patching works for individual prompt pairs but top sites are unstable.
- `no_reliable_object`: ordinary behavior or method signal is too weak to interpret.
- `behavior_absent_or_weak`: GPT-2 Small did not show enough target behavior to support MI analysis.

## 16. Ordinary Success That Failed Under Decoupling
These cases are identified by matrix rows where `ordinary_heldout` is `pass` or `weak` and at least one decoupling axis is `fail`. They should be read as construct-validity failures, not necessarily method bugs.

## 17. Causal Steering Worked but Was Nonspecific
Direction steering is summarized separately from prediction. When target-logit movement appears with high KL divergence, it is classified as nonspecific causal control rather than a clean construct handle.

## 18. Probes Detected Information That Steering Did Not Control
Probe success is treated as predictive evidence only. The report does not infer causal control from probe accuracy, even when it exceeds direction accuracy.

## 19. Patching Appeared Prompt-local
Patching top sites are evaluated for stability across examples. Unstable top sites are treated as prompt-local dependencies rather than stable construct variables.

## 20. Limitations
The dataset is small, manually templated, and English-only. Patching runs on a small subset for speed. GPT-2 Small target-token behavior may not align with human labels on all decoupled examples, and weak behavior is not converted into success language.

## 21. Recommended Next Experiment
Keep GPT-2 Small fixed and improve the certainty dataset with a second manual pass focused on format-shift and nonlexical uncertainty. Then rerun the same matrix to test whether failures persist after reducing ambiguous examples.
