# Attention Lab

A modular GPT training harness for attention architecture experiments. The imported
`build-nanogpt` files at the repository root are reference material; active code lives
under `src/attention_lab`.

## Repository Rules

- Use `uv` only for environment creation, dependency installation, and execution.
- Do not run manual `pip install`, manual virtualenv setup, or ad hoc dependency commands.
- Keep generated datasets, HellaSwag downloads, and run outputs out of git.

## 1. Bootstrap

```bash
uv sync
```

Convenience wrapper:

```bash
./scripts/bootstrap.sh
```

## 2. Verify CUDA

```bash
uv run scripts/verify_cuda.py
```

If CUDA is unavailable, stop and fix the driver/PyTorch environment before training.
The baseline configs request CUDA and bf16.

## 3. Prepare FineWeb-Edu

```bash
uv run scripts/prepare_fineweb_edu.py \
  --out_dir data/fineweb_edu_100m \
  --train_tokens 100000000 \
  --val_tokens 4000000
```

Expected output:

```text
data/fineweb_edu_100m/
  edufineweb_val_000000.npy
  edufineweb_train_000001.npy
```

Verify the shards:

```bash
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m
```

## 4. Test And QC

```bash
uv run pytest
uv run ruff check .
```

The tests use tiny synthetic `.npy` token shards for unit and integration checks. They
are not evidence of real FineWeb-Edu model quality.

## 5. Sanity Run

```bash
uv run scripts/train.py --config configs/baseline_15m_fineweb100m_sanity.yaml --overwrite
```

Expected initial validation loss is near `ln(50304) = 10.825`. The sanity config runs
only 20 steps and is a systems check for CUDA training, bf16 autocast, metrics,
checkpointing, reloadability, and generation.

Verify the sanity run:

```bash
uv run scripts/verify_run.py \
  --run_dir runs/baseline_15m_fineweb100m_sanity_seed1 \
  --expect-complete-training \
  --expect-sample
```

## 6. Full Baseline Run

```bash
uv run scripts/train.py --config configs/baseline_15m_fineweb100m.yaml --overwrite
```

This is the first real 15M standard-attention baseline. It runs 3000 steps and should
not be described as complete until the command exits successfully and the run verifier
passes.

Verify the completed baseline:

```bash
uv run scripts/verify_run.py \
  --run_dir runs/baseline_15m_fineweb100m_seed1 \
  --expect-complete-training \
  --expect-sample
```

## 7. Reload And Eval

Validation loss from a checkpoint:

```bash
uv run scripts/eval_loss.py \
  --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt \
  --data_root data/fineweb_edu_100m
```

Generation:

```bash
uv run scripts/eval_generate.py \
  --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt \
  --prompt "The history of mathematics"
```

Bounded HellaSwag:

```bash
uv run scripts/eval_hellaswag.py \
  --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt \
  --max_examples 100
```

Verify a run after eval artifacts exist:

```bash
uv run scripts/verify_run.py \
  --run_dir runs/baseline_15m_fineweb100m_seed1 \
  --expect-complete-training \
  --expect-sample \
  --expect-eval-loss \
  --expect-hellaswag
```

## 8. Summarize A Run

```bash
uv run scripts/summarize_run.py --run_dir runs/baseline_15m_fineweb100m_seed1
```

This prints:

```text
run_dir
run_name
max_step
train_event_count
val_event_count
initial_val_loss
final_val_loss
best_val_loss
initial_val_perplexity
final_val_perplexity
median_tokens_per_sec
peak_vram_mb
checkpoint_count
```

It also writes:

```text
runs/<run_name>/evals/run_summary.json
```

## Run Directory Contract

A completed baseline run should contain:

```text
runs/<run_name>/
  config.yaml
  config_source.txt
  environment.txt
  git_commit.txt
  metrics.jsonl
  metrics.csv
  checkpoints/
    ckpt_step_*.pt
    ckpt_last.pt
  samples/
    sample_step_last.txt
  evals/
    val_loss.json
    hellaswag.json
    run_summary.json
```

Use `scripts/verify_run.py` rather than checking this manually.

## Attention Swap Point

The GPT block builds attention through:

```python
self.attn = build_attention(config)
```

The implemented baseline attention type is:

```yaml
model:
  attention_type: standard
```

Future attention modules should be added behind the registry without changing the
trainer contract.

## Not Implemented In Baseline

- `trilinear_cp` is not implemented. Its placeholder config lives under
  `configs/experimental/` with `status: experimental_unimplemented`, and the baseline
  config loader rejects it by default.
- `torch.compile` is intentionally unsupported for baseline QC. Config validation
  fails when `train.compile: true`.
- DDP code exists, but single-GPU non-DDP behavior is the tested baseline path.
- OpenAI Evals is not used for this baseline. Primary metrics are next-token
  validation loss, perplexity, throughput, peak VRAM, and checkpoint reloadability.
- `lm-evaluation-harness` is deferred until an HF-compatible export exists.

