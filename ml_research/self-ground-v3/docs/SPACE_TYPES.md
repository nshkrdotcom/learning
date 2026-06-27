# Space Types

The space type system prevents silent algebra and patching errors before they become evidence.

It covers:

- `TensorSpace`: typed read/write space for tensors.
- `TensorRef`: a typed tensor value linked to a `TensorSpace`.
- `SpaceTransform`: an explicit transform between spaces with provenance.
- `MechanisticUnitRef`: an addressable unit with valid and invalid operations.
- `SpaceCompatibilityReport`: deterministic pass/fail result for a proposed operation.

## Check

```bash
uv run mwb space check docs/fixtures/space_check_valid.json
```

The command writes:

```text
.mechanism/space_checks/latest_space_check.json
```

and indexes the report in SQLite table `space_checks`.

## TensorSpace

`TensorSpace` includes:

- model ref;
- backend;
- hook point;
- layer;
- stream kind;
- basis;
- normalization context;
- token-position semantics;
- dtype and shape.

Pre-LN and post-LN spaces are incompatible unless a matching `SpaceTransform` is declared.

## MechanisticUnitRef

Units record:

- unit kind;
- model ref;
- tensor/read/write spaces;
- dictionary ref for SAE features;
- feature/head/layer identity where relevant;
- valid operations;
- invalid operations.

An operation fails if it is in `invalid_operations`, or if `valid_operations` is non-empty and does not include the requested operation.

## Current Blockers

`mwb space check` currently blocks:

- `incompatible_dictionary`: comparing SAE features from different dictionaries.
- `normalization_context_mismatch`: using pre-LN and post-LN spaces without a declared transform.
- `missing_transform_provenance`: declaring a transform without provenance.
- `wrong_hook_point`: patching or ablation into a target hook that does not match the unit write space.
- `invalid_operation_for_unit`: requesting an operation the unit registry forbids.

## Transform Provenance

Transforms are explicit records:

```json
{
  "wb_ref": "xform_fold_ln",
  "from_space_ref": "space_pre",
  "to_space_ref": "space_post",
  "transform_kind": "fold_layernorm",
  "provenance_ref": "preflight_report_001"
}
```

The provenance ref is required because transformed evidence must remain explainable and reproducible.

## Claim Boundary

A passing space check only means the requested operation is structurally compatible. It does not establish causal evidence, validate controls, or promote claim status.
