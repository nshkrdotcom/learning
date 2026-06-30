# Attention Lab

Attention Lab is a small GPT pretraining harness for local attention architecture
experiments. It is intentionally built around a standard-attention baseline, real
FineWeb-Edu token shards, reproducible run artifacts, and one clean attention swap
point.

Active code lives under `src/attention_lab`. Imported `build-nanogpt` files at the
repository root are reference material.

## What This Repo Is

- A single-GPU-friendly GPT training harness.
- A reproducible standard-attention baseline for later architecture comparisons.
- A compact place to add new attention modules behind the existing registry.
- A local workflow using `uv`, YAML configs, `.npy` token shards, JSONL/CSV metrics,
  checkpoints, run summaries, and verifier scripts.

## What This Repo Is Not

- It is not a replacement with LitGPT, nanochat, nanoGPT, GPT-NeoX, TorchTitan, or any
  other upstream framework.
- It is not currently an HF-compatible export pipeline.
- It does not use OpenAI Evals for this stage.
- It does not treat HellaSwag as the primary metric.

Primary baseline metrics are next-token validation loss, perplexity, throughput,
allocated/reserved VRAM, and checkpoint reloadability.

## Current Verified Baseline

The completed local run is:

```text
runs/baseline_15m_fineweb100m_seed1
```

That run name is historical. The model is actually about 30M parameters:

```text
parameters_excluding_positional: 29938560
parameters_including_positional: 30331776
```

Use this accurate-size config for new runs:

```text
configs/baseline_30m_fineweb100m.yaml
```

The historical config remains for compatibility:

```text
configs/baseline_15m_fineweb100m.yaml
```

Completed baseline facts:

```text
final_val_loss: 4.081209182739258
best_val_loss: 4.081209182739258
final_val_perplexity: 59.2170307875361
median_tokens_per_sec: 107022.7422894312
peak_vram_allocated_mb: 3240.92431640625
bounded_hellaswag_accuracy_norm: 0.34
```

Interpretation note:

```text
3000 * 262144 = 786432000 token positions
```

This run made multiple passes over the 100M-token training shard. It is not a unique
786M-token corpus.

## Setup

Use `uv` only:

```bash
uv sync
```

Verify CUDA before training:

```bash
uv run scripts/verify_cuda.py
```

The verified local environment used PyTorch `2.11.0+cu128`, CUDA `12.8`, RTX 5060 Ti,
and bf16 support.

## Data Prep And Manifest

Prepare FineWeb-Edu 100M train / 4M validation token shards:

```bash
uv run scripts/prepare_fineweb_edu.py \
  --out_dir data/fineweb_edu_100m \
  --train_tokens 100000000 \
  --val_tokens 4000000
```

Write a manifest with shard stats and SHA256 hashes:

```bash
uv run scripts/write_data_manifest.py \
  --data_root data/fineweb_edu_100m \
  --out data/fineweb_edu_100m/manifest.json
```

Verify shards and hashes:

```bash
uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes
```

When `manifest.json` exists under the data root, training copies it to:

```text
runs/<run_name>/data_manifest.json
runs/<run_name>/data_manifest.sha256
```

## Sanity Run

The sanity run is a systems check, not architecture evidence:

```bash
uv run scripts/train.py --config configs/baseline_15m_fineweb100m_sanity.yaml --overwrite
uv run scripts/verify_run.py \
  --run_dir runs/baseline_15m_fineweb100m_sanity_seed1 \
  --expect-complete-training \
  --expect-sample
```

Expected initial validation loss is near `ln(50304) = 10.825`.

## Full Baseline Run

For new accurate-size naming:

```bash
./scripts/run_full_30m_baseline.sh
```

The completed historical run used the same model/data/training recipe:

```bash
./scripts/run_full_baseline.sh
```

`scripts/run_full_baseline.sh` is a historical completed-run reproducer. New runs
should use `scripts/run_full_30m_baseline.sh`.

## Run Verification

Verify a completed run:

```bash
uv run scripts/verify_run.py \
  --run_dir runs/baseline_15m_fineweb100m_seed1 \
  --expect-complete-training \
  --expect-sample \
  --expect-data-manifest
```

Run checkpoint reload eval, generation, bounded HellaSwag, summary, then verify all
artifacts:

```bash
uv run scripts/eval_loss.py \
  --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt \
  --data_root data/fineweb_edu_100m

uv run scripts/eval_generate.py \
  --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt \
  --prompt "The history of mathematics"

uv run scripts/eval_hellaswag.py \
  --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt \
  --max_examples 100

uv run scripts/summarize_run.py \
  --run_dir runs/baseline_15m_fineweb100m_seed1

uv run scripts/verify_run.py \
  --run_dir runs/baseline_15m_fineweb100m_seed1 \
  --expect-complete-training \
  --expect-sample \
  --expect-eval-loss \
  --expect-hellaswag \
  --expect-data-manifest
```

`eval_loss.py` verifies data-manifest compatibility when the checkpoint or run
directory has manifest provenance. Intentional cross-data evals must pass:

```bash
--allow-data-manifest-mismatch
```

## Run Summarization

```bash
uv run scripts/summarize_run.py --run_dir runs/baseline_15m_fineweb100m_seed1
```

This writes:

```text
runs/<run_name>/evals/run_summary.json
```

Compare two run summaries:

```bash
uv run scripts/compare_runs.py \
  --baseline runs/baseline_15m_fineweb100m_seed1 \
  --candidate runs/baseline_15m_fineweb100m_seed1
```

The machine-readable summary schema is:

```text
reports/schema/run_summary.schema.json
```

## Config Ladder

Standard-attention configs:

```text
configs/baseline_16m_fineweb100m.yaml
configs/baseline_30m_fineweb100m.yaml
configs/baseline_70m_fineweb300m.yaml
configs/baseline_125m_fineweb1b.yaml
```

The 70M and 125M configs are templates until their data roots are prepared and
manifested:

```text
data/fineweb_edu_300m/manifest.json
data/fineweb_edu_1b/manifest.json
```

Historical compatibility configs:

```text
configs/baseline_15m_fineweb100m.yaml
configs/baseline_124m_fineweb1b.yaml
```

`baseline_124m_fineweb1b.yaml` is a historical alias. Prefer the canonical
`baseline_125m_fineweb1b.yaml` name for new reports and runs.

Inspect exact model sizes:

```bash
uv run scripts/inspect_model_config.py --config configs/baseline_30m_fineweb100m.yaml
```

Current inspected sizes:

```text
16M tier: 16025856 excluding positional, 16288000 including positional
30M tier: 29938560 excluding positional, 30331776 including positional
70M tier: 69810688 excluding positional, 70334976 including positional
125M tier: 123587328 excluding positional, 124373760 including positional
```

## Architecture Experiment Contract

Read before adding new attention modules:

```text
docs/architecture_experiment_contract.md
```

Architecture variants must hold fixed the data manifest, tokenizer, shards, batch
construction, token budget, optimizer, LR schedule, seed policy, eval cadence,
checkpoint/eval scripts, and run verifier. Every variant must report parameter count,
parameter delta, final/best validation loss, perplexity, tokens/sec, allocated/reserved
VRAM, wall-clock runtime, checkpoint reload eval loss, and bounded HellaSwag when
requested.

The resolved pre-experiment cleanup checklist is:

```text
docs/pre_experiment_cleanup_checklist.md
```

## Adding A New Attention Module

The GPT block builds attention through:

```python
self.attn = build_attention(config)
```

Implemented baseline:

```yaml
model:
  attention_type: standard
```

Add future attention modules behind `attention_lab.models.attention_registry` without
changing trainer behavior. The `trilinear_cp` placeholder remains intentionally
unimplemented.

## HF Export And lm-eval

`scripts/export_hf.py` is an honest stub:

```bash
uv run scripts/export_hf.py --checkpoint path/to/ckpt_last.pt --out_dir exported_hf_model
```

It exits nonzero until HF export is implemented and verified. `lm-evaluation-harness`
integration is deferred until exported checkpoints load through HF APIs and logits are
verified against the internal model.

## Known Limitations

- `trilinear_cp` is not implemented.
- `torch.compile` is intentionally unsupported for baseline QC.
- DDP code exists, but single-GPU non-DDP behavior is the tested path.
- OpenAI Evals is not used for this training baseline.
- HellaSwag is optional bounded smoke/eval support, not the primary metric.
- HellaSwag eval JSON records the cached file path, source URL, and SHA256.
- `lm-evaluation-harness` is deferred until HF export exists.
