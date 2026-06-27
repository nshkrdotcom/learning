# Evidence Graph

The evidence graph is the queryable provenance layer over file-backed workbench records.

It is rebuilt from `.mechanism/` files and then written to:

- `.mechanism/graph/evidence_edges.jsonl`
- `.mechanism/graph/graph_summary.json`
- SQLite table `evidence_edges`

SQLite is an operational index only. Deleting SQLite must not lose graph information after `mwb graph rebuild` has written the JSONL edge file.

## Edge Schema

Each edge is an `EvidenceEdge` domain object:

```json
{
  "wb_ref": "edge_...",
  "wb_type": "EvidenceEdge",
  "wb_version": "1",
  "src_ref": "unit_sae_12300",
  "dst_ref": "hyp_negation_detector",
  "relation": "depends_on",
  "source_ref": "hyp_negation_detector",
  "source_path": ".mechanism/hypotheses/hyp_negation_detector.json",
  "parents": ["unit_sae_12300", "hyp_negation_detector"],
  "metadata": {
    "src_type": "mechanistic_unit",
    "dst_type": "hypothesis"
  }
}
```

Allowed relations:

- `supports`
- `contradicts`
- `depends_on`
- `derived_from`
- `tested_by`
- `confounded_by`
- `fails_on`
- `generalizes_to`
- `cited_by`

Unknown relation names fail validation instead of being silently indexed.

## Rebuild

```bash
uv run mwb graph rebuild
```

The rebuild reads:

- `.mechanism/hypotheses/*.json`
- `.mechanism/sessions/*/cells.jsonl`
- `.mechanism/sessions/*/namespace_objects.jsonl`
- `.mechanism/sessions/*/artifacts.jsonl`
- `.mechanism/runs/*/run_manifest.json`
- `.mechanism/runs/*/verification_results.jsonl`
- `.mechanism/runs/*/blocker_report.json`
- `.mechanism/runs/*/mechanism_card.json`
- `.mechanism/runs/*/scientific_debt.json`
- `.mechanism/claims/*.json`

The rebuild also refreshes SQLite `evidence_edges`. `mwb repair-index` and `mwb rebuild-index` restore `evidence_edges` from `.mechanism/graph/evidence_edges.jsonl`.

## Queries

```bash
uv run mwb graph query claims-depending-on <unit-or-object-ref>
uv run mwb graph query controls-contradicting <run-ref>
uv run mwb graph query cells-producing <artifact-ref>
uv run mwb graph query debt-blocking <claim-ref>
```

These queries answer the first required Phase 12 graph questions:

- which claims depend on a mechanistic unit or intermediate object;
- which controls contradicted a run;
- which cells produced an artifact;
- which scientific debt records block a claim.

Query output is JSON with `status`, `query`, `ref`, `count`, and `results`. Each result includes the edge ref, relation, source path, metadata, and path where relevant.

## Claim Boundary

Graph edges record declared provenance and evidence relationships. They do not upgrade evidence tiers by themselves. A `supports` edge must be interpreted with its metadata, MechanismCard status, blockers, and scientific debt.
