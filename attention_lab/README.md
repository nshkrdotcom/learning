# Attention Lab

Attention Lab is a small GPT pretraining harness for local attention architecture
experiments. The standard-attention baseline is the control path; new mechanisms must
live behind the attention registry and prove themselves through manifest-checked,
verifier-checked runs.

## 1. Baseline Setup

Use `uv` only:

```bash
uv sync
uv run scripts/verify_cuda.py
```

Prepare and verify FineWeb-Edu 100M/4M shards:

```bash
uv run scripts/prepare_fineweb_edu.py \
  --out_dir data/fineweb_edu_100m \
  --train_tokens 100000000 \
  --val_tokens 4000000

uv run scripts/write_data_manifest.py \
  --data_root data/fineweb_edu_100m \
  --out data/fineweb_edu_100m/manifest.json

uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes
```

## 2. Verified Baseline Result

The completed historical baseline run is:

```text
runs/baseline_15m_fineweb100m_seed1
```

The name is historical. The actual model size is about 30M parameters:

```text
parameters_excluding_positional: 29938560
parameters_including_positional: 30331776
```

Use the accurate-size config for new runs:

```text
configs/baseline_30m_fineweb100m.yaml
```

Completed baseline result:

```text
final_val_loss: 4.081209182739258
best_val_loss: 4.081209182739258
final_val_perplexity: 59.2170307875361
median_tokens_per_sec: 107022.7422894312
peak_vram_allocated_mb: 3240.92431640625
bounded_hellaswag_accuracy_norm: 0.34
```

The run processed `3000 * 262144 = 786432000` token positions, meaning multiple
passes over the 100M-token train shard, not a unique 786M-token corpus.

## 3. Experiment Organization

Experiments are registered in:

```text
docs/experiments/experiments.yaml
```

The first planned experiment is:

```text
E001_cp_trilinear_attention
```

Directory convention:

```text
configs/experiments/<EXPERIMENT_ID>/
docs/experiments/<EXPERIMENT_ID>_plan.md
runs/experiments/<EXPERIMENT_ID>/
reports/experiments/<EXPERIMENT_ID>/
```

List and validate experiments:

```bash
uv run scripts/list_experiments.py
uv run scripts/list_experiments.py --id E001_cp_trilinear_attention
uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
```

## 4. Architecture Module Organization

Attention implementations live under:

```text
src/attention_lab/models/attention/
  standard.py
  trilinear_cp.py
  registry.py
```

Compatibility shims preserve old import paths:

```text
src/attention_lab/models/attention_standard.py
src/attention_lab/models/attention_trilinear_cp.py
src/attention_lab/models/attention_registry.py
```

The GPT block still calls:

```python
self.attn = build_attention(config)
```

The only implemented baseline attention type is:

```yaml
model:
  attention_type: standard
```

`trilinear_cp` and `cp_bilinear` experiment configs are planned skeletons and
intentionally fail as `experimental_unimplemented` until their modules are implemented.

## 5. Add A New Attention Variant

Follow:

```text
docs/architecture_variant_checklist.md
docs/architecture_experiment_contract.md
docs/pre_experiment_cleanup_checklist.md
```

Rules:

- Do not edit baseline files unless the experiment explicitly includes a
  standard-refactor control.
- Add new modules under `src/attention_lab/models/attention/`.
- Add registry and config validation support.
- Add tests for shape, causal masking, gradient flow, parameter count, and diagnostics.
- Keep dataset manifest, token budget, optimizer, LR schedule, seed, and eval cadence
  fixed for direct comparisons.

## 6. Validate An Experiment

```bash
uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
uv run scripts/inspect_model_config.py \
  --config configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml
```

The validator checks config parseability, runnable-vs-unimplemented status, unique run
directories, experiment run-dir containment, fixed baseline fields, dataset manifest,
and local historical baseline summary when present.

## 7. Run An Experiment

Runnable standard-control config:

```bash
uv run scripts/train.py \
  --config configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml \
  --overwrite

uv run scripts/verify_run.py \
  --run_dir runs/experiments/E001_cp_trilinear_attention/standard_30m_seed1 \
  --expect-complete-training \
  --expect-sample \
  --expect-data-manifest
```

CP configs are present as planned skeletons:

```text
configs/experiments/E001_cp_trilinear_attention/cp_bilinear_r8_30m_seed1.yaml
configs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1.yaml
configs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_lambda0_30m_seed1.yaml
```

They are not runnable until the architecture implementation pass.

## 8. Compare To Baseline

```bash
uv run scripts/compare_runs.py \
  --experiment E001_cp_trilinear_attention \
  --baseline runs/baseline_15m_fineweb100m_seed1 \
  --candidate runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1 \
  --json-out reports/experiments/E001_cp_trilinear_attention/comparison.json
```

Comparisons read `evals/run_summary.json`, add experiment metadata, require candidate
runs to live under the experiment run directory, and report deltas/ratios for loss,
perplexity, throughput, and VRAM when numeric values are available.

Future CP runs must emit diagnostics to:

```text
runs/experiments/E001_cp_trilinear_attention/<run_name>/evals/attention_diagnostics.jsonl
```

Schema:

```text
reports/schema/attention_diagnostics.schema.json
```

## 9. Known Limitations

- CP-bilinear attention is not implemented.
- CP-trilinear attention is not implemented.
- `torch.compile` is intentionally unsupported for baseline QC.
- DDP exists but is not the tested path.
- OpenAI Evals is not used for this training baseline.
- HellaSwag is optional bounded smoke/eval support, not the primary metric.
- HF export is still an honest stub, so `lm-evaluation-harness` remains deferred.
