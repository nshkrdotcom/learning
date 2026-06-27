# Adapter Guide

Mechanistic Workbench adapters are thin, audited boundaries over real external tools.
They do not replace backend libraries, and they do not make diagnostic metadata into
claim-bearing evidence.

Every adapter conformance command writes:

- `.mechanism/adapters/<adapter>/manifest.json`
- `.mechanism/adapters/<adapter>/backend_versions.json`
- `.mechanism/adapters/<adapter>/<adapter>_conformance.json`

`mwb repair-index` rebuilds `adapter_manifests` and `backend_versions` from those
file-backed records.

## Conformance Matrix

| Adapter | Dependency posture | Main purpose | Claim-bearing |
| --- | --- | --- | --- |
| `transformer-lens` | required P0 dependency | model load, hook identity, activation capture | yes, after full conformance pass |
| `saelens` | required P0 dependency | SAE identity, hook compatibility, feature refs | yes, after full conformance pass |
| `nnsight` | optional P1 dependency | HF-exact tracing/intervention target, nnterp naming when installed | no |
| `pyvene` | optional P1 dependency | serializable intervention-contract target | no |
| `neuronpedia` | read-only external metadata | feature metadata refs and external aliases | no |

Optional adapters can be useful for planning and metadata hygiene before their real
execution environments are installed. Their manifests explicitly set
`claim_bearing.supported=false`, and the claim gate blocks them even if a diagnostic
or API-contract check succeeds.

## Commands

P0 real conformance:

```bash
uv run mwb adapter conformance transformer-lens \
  --model EleutherAI/pythia-70m-deduped \
  --device cpu

uv run mwb adapter conformance saelens \
  --model EleutherAI/pythia-70m-deduped \
  --hook blocks.2.hook_resid_post \
  --device cpu
```

Optional diagnostic conformance:

```bash
uv run mwb adapter conformance nnsight \
  --model gpt2 \
  --module-path transformer.h.0.mlp \
  --device cpu \
  --dry-run

uv run mwb adapter conformance pyvene \
  --model gpt2 \
  --module-path transformer.h.0.mlp \
  --intervention-kind resample_ablation \
  --device cpu \
  --dry-run

uv run mwb adapter conformance neuronpedia \
  --model-id gemma-2-2b \
  --sae-id 20-gemmascope-res-16k \
  --feature-index 123 \
  --dry-run
```

When `nnsight` or `pyvene` is missing, non-dry conformance records a
`diagnostic_only` result with an explicit missing optional backend check. It does
not silently fall back to another backend.

## Artifact Pointers

`ctx.artifact.register(...)` and `ArtifactRegistry.register_path(...)` record
external artifact pointers without dereferencing large storage systems:

- Git LFS pointer files: backend `git_lfs`, target `oid`, and target `size`.
- DVC `.dvc` files: backend `dvc`, target path, content oid, and size when present.
- git-annex symlinks: backend `git_annex`, annex key, target, and parsed size when present.

Pointer records are stored in the normal `artifacts` table with
`materialized=false`. The pointer file or symlink itself is hashed locally, while
the external target identity is preserved in `pointer`.

## Claim Safety

An adapter is claim-bearing only when all of these are true:

- its capability manifest declares `claim_bearing.supported=true`;
- the required conformance result has `status=pass`;
- all required model, tensor, dictionary, artifact, and backend refs are present.

The gate rejects missing conformance, failed conformance, diagnostic-only
conformance, and manifests that explicitly opt out of claim-bearing support.
