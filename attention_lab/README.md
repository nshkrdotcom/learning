# Attention Lab

A modular GPT training harness for attention architecture experiments. The upstream
`build-nanogpt` files are kept at the repository root as reference material; active
training code lives under `src/attention_lab`.

## Rules for this repo

- Use `uv` only for environment creation, dependency installation, and execution.
- Do not run manual `pip install`, manual virtualenv setup, or ad hoc dependency commands.
- Keep generated datasets and run outputs out of git.

## Bootstrap

```bash
./scripts/bootstrap.sh
```

This runs:

```bash
uv sync
uv run scripts/verify_cuda.py
```

If CUDA is not available, stop and fix the driver/PyTorch environment before training.

## Prepare FineWeb-Edu 100M

```bash
./scripts/prepare_fineweb_edu_100m.sh
```

This creates:

```text
data/fineweb_edu_100m/
  edufineweb_val_000000.npy
  edufineweb_train_000001.npy
```

For a custom token budget:

```bash
uv run scripts/prepare_fineweb_edu.py \
  --out_dir data/fineweb_edu_100m \
  --train_tokens 100000000 \
  --val_tokens 4000000
```

Verify shards:

```bash
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m
```

## Sanity run

```bash
./scripts/run_sanity.sh
```

Equivalent command:

```bash
uv run scripts/train.py --config configs/baseline_15m_fineweb100m_sanity.yaml
```

Expected early validation loss is near `ln(50304) = 10.825` before training.

## Baseline run

```bash
uv run scripts/train.py --config configs/baseline_15m_fineweb100m.yaml
```

The run directory contains:

```text
runs/<run_name>/
  config.yaml
  config_source.txt
  environment.txt
  git_commit.txt
  metrics.jsonl
  metrics.csv
  checkpoints/
  samples/
  evals/
```

If a run directory already exists, the trainer stops instead of mixing outputs. Use
`--overwrite` only when you intentionally want to replace that run.

## Reload and eval

```bash
uv run scripts/eval_loss.py \
  --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt \
  --data_root data/fineweb_edu_100m

uv run scripts/eval_generate.py \
  --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt \
  --prompt "The history of mathematics"
```

Optional HellaSwag:

```bash
uv run scripts/eval_hellaswag.py \
  --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt
```

## Attention swap point

The GPT block builds attention through:

```python
self.attn = build_attention(config)
```

Change only this config field to select an implementation:

```yaml
model:
  attention_type: standard
```

`trilinear_cp` is wired into the registry as a deliberate placeholder so the trainer
interface is fixed before novel attention code is added.

