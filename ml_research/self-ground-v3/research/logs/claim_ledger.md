# Claim Ledger

Claims are added manually or through `mwb ledger propose-claim <card-ref>` proposals.

Each claim entry uses:

```text
### C001 - Short claim title
```

followed by a required fenced `yaml` block with at least:

```yaml
claim_id: C001
status: no_real_evidence
allowed: []
forbidden: []
```
