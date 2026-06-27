# Held-Out Robustness Candidate Set

This candidate set was selected from prior multi-seed artifacts before held-out scoring.

- Source: `/home/home/p/g/n/learning/ml_research/local-mi-lab/reports/head_specific_induction_causality_v1/head_specific_multiseed_by_head.csv`
- Candidates: `16`
- Groups: `{'negative_control_no_effect': 3, 'negative_control_nonspecific': 3, 'prior_raw_attention_failed': 5, 'random_comparison_replicated': 1, 'replicated_candidate': 4}`

| candidate_id | head | group | prior status | prior gap | flags |
| --- | --- | --- | --- | --- | --- |
| heldout_cand_000 | L7H7 | random_comparison_replicated | replicated_head_specific_candidate | 0.0806 | random-comparison |
| heldout_cand_001 | L9H11 | replicated_candidate | replicated_head_specific_candidate | 0.0357 |  |
| heldout_cand_002 | L7H11 | replicated_candidate | replicated_head_specific_candidate | 0.0259 |  |
| heldout_cand_003 | L7H0 | replicated_candidate | replicated_head_specific_candidate | 0.0105 |  |
| heldout_cand_004 | L0H8 | replicated_candidate | replicated_head_specific_candidate | 0.0077 |  |
| heldout_cand_005 | L0H1 | prior_raw_attention_failed | no_effect | -0.1304 | raw-attention |
| heldout_cand_006 | L0H5 | prior_raw_attention_failed | no_effect | -0.0469 | raw-attention |
| heldout_cand_007 | L0H10 | prior_raw_attention_failed | nonspecific | -0.1086 | raw-attention |
| heldout_cand_008 | L11H8 | prior_raw_attention_failed | nonspecific | 0.0116 | raw-attention |
| heldout_cand_009 | L0H4 | prior_raw_attention_failed | no_effect | -0.0181 | raw-attention |
| heldout_cand_010 | L9H5 | negative_control_no_effect | no_effect | 0.0013 |  |
| heldout_cand_011 | L11H0 | negative_control_no_effect | no_effect | -0.0056 |  |
| heldout_cand_012 | L4H1 | negative_control_no_effect | no_effect | -0.0062 | random-comparison |
| heldout_cand_013 | L4H9 | negative_control_nonspecific | nonspecific | -0.0000 |  |
| heldout_cand_014 | L11H9 | negative_control_nonspecific | nonspecific | 0.0005 | random-comparison |
| heldout_cand_015 | L11H7 | negative_control_nonspecific | nonspecific | 0.0005 | random-comparison |

L7H7 remains flagged if it was a prior random-comparison candidate. The held-out run must not treat that label as evidence.
