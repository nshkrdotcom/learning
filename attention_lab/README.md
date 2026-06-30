# Attention Lab

A modular GPT training harness for attention architecture experiments. The imported
`build-nanogpt` files at the repository root are reference material; active code lives
under `src/attention_lab`.

## Local Completion Status

As of 2026-06-29, this repository has three distinct states:

1. Harness sanity verification: complete. Tests, ruff, CUDA verification, data
   verification, a 20-step CUDA sanity run, checkpoint reload, generation, bounded
   HellaSwag, run verification, and run summarization passed locally.
2. Full 15M baseline completion: complete for
   `configs/baseline_15m_fineweb100m.yaml`. The standard-attention baseline reached
   step 3000 on FineWeb-Edu 100M/4M token shards, and post-run evals passed.
3. Future architecture experiments: not implemented here. `trilinear_cp` remains an
   experimental placeholder and is excluded from baseline QC.

The full baseline run is recorded in `reports/baseline_harness_verification.md`; a
compact comparison reference lives in `reports/baseline_15m_fineweb100m_summary.md`.

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

This is the first real standard-attention baseline. It runs 3000 steps over the
prepared FineWeb-Edu shards. On the local RTX 5060 Ti 16GB, the completed run used the
config as committed, finished in about 2h08m, reached final validation loss `4.081209`,
final perplexity `59.217031`, median throughput `107022.74` tokens/sec, and PyTorch
peak allocated VRAM `3240.92` MB. A concurrent `nvidia-smi` sample during training
reported about 12 GB device memory in use.

Verify the completed baseline:

```bash
uv run scripts/verify_run.py \
  --run_dir runs/baseline_15m_fineweb100m_seed1 \
  --expect-complete-training \
  --expect-sample
```

Full post-run command sequence:

```bash
uv run scripts/train.py --config configs/baseline_15m_fineweb100m.yaml --overwrite

uv run scripts/verify_run.py \
  --run_dir runs/baseline_15m_fineweb100m_seed1 \
  --expect-complete-training \
  --expect-sample

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
  --expect-hellaswag
```

Long-running wrapper:

```bash
./scripts/run_full_baseline.sh
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

## Definition Of Done

The standard baseline is complete when:

- `uv run pytest` and `uv run ruff check .` pass.
- CUDA and FineWeb-Edu token shards verify locally.
- `configs/baseline_15m_fineweb100m.yaml` trains to step 3000.
- `verify_run.py` passes with complete-training, sample, eval-loss, and HellaSwag
  expectations.
- `eval_loss.py`, `eval_generate.py`, `eval_hellaswag.py --max_examples 100`, and
  `summarize_run.py` run against `ckpt_last.pt`.
- The verification report records the actual local results.

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
