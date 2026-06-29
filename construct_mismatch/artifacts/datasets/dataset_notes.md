# Dataset Quality Notes

The dataset is intentionally small and template-curated. Each example was written to make the target construct explicit while allowing the decoupling axis to stress a different validity target.

Sentiment is included as a familiar sanity-check construct. Certainty/uncertainty is the primary construct because the intended contribution is the controlled construct-validity matrix, not sentiment analysis.

All class targets used by the generated JSONL files were validated as single GPT-2 tokens with leading spaces. Entries whose target pair failed single-token validation are excluded before size checks.

Known caveat: several format-shift examples intentionally share a surface prompt across class roles and rely on the target-token contrast. They are useful for measuring target-token behavior but should not be overread as naturalistic semantic minimal pairs.

Counts:
- certainty:
  - train: n=60 (ordinary=60)
  - heldout: n=36 (ordinary=36)
  - decoupling: n=60 (lexical_reversal=12, negation=12, quotation=12, contrast=12, format_shift=12)
- sentiment:
  - train: n=60 (ordinary=60)
  - heldout: n=36 (ordinary=36)
  - decoupling: n=60 (lexical_reversal=12, negation=12, quotation=12, contrast=12, format_shift=12)
