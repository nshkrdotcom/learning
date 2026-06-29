# Baseline Harness Verification

Date: 2026-06-29

Git commit at verification start: `f71e790381aa7982db51663639d342a8b09edb65`

## Machine And Environment

- Platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`
- Python: `3.12.2`
- PyTorch: `2.11.0+cu128`
- CUDA available: `True`
- CUDA version: `12.8`
- GPU: `NVIDIA GeForce RTX 5060 Ti`
- bf16 supported: `True`

## Commands Run

```bash
uv sync
uv run pytest
uv run ruff check .
uv run scripts/verify_cuda.py
uv run scripts/prepare_fineweb_edu.py --out_dir data/fineweb_edu_100m --train_tokens 100000000 --val_tokens 4000000
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m
uv run scripts/train.py --config configs/baseline_15m_fineweb100m_sanity.yaml --overwrite
uv run scripts/verify_run.py --run_dir runs/baseline_15m_fineweb100m_sanity_seed1 --expect-complete-training --expect-sample
uv run scripts/eval_loss.py --checkpoint runs/baseline_15m_fineweb100m_sanity_seed1/checkpoints/ckpt_last.pt --data_root data/fineweb_edu_100m
uv run scripts/eval_generate.py --checkpoint runs/baseline_15m_fineweb100m_sanity_seed1/checkpoints/ckpt_last.pt --prompt "The history of mathematics"
uv run scripts/eval_hellaswag.py --checkpoint runs/baseline_15m_fineweb100m_sanity_seed1/checkpoints/ckpt_last.pt --max_examples 100
uv run scripts/summarize_run.py --run_dir runs/baseline_15m_fineweb100m_sanity_seed1
uv run scripts/verify_run.py --run_dir runs/baseline_15m_fineweb100m_sanity_seed1 --expect-complete-training --expect-sample --expect-eval-loss --expect-hellaswag
```

The first data-prep attempt wrote both shards but exited with code `134` during Python
native-extension shutdown. `scripts/prepare_fineweb_edu.py` was patched to exit
explicitly after successful writes, and the same 100M/4M command was rerun
successfully with exit code `0`.

## CUDA Result

```text
torch: 2.11.0+cu128
cuda available: True
cuda version: 12.8
device: NVIDIA GeForce RTX 5060 Ti
bf16 supported: True
```

## FineWeb-Edu Data

Prepared and verified:

```text
data/fineweb_edu_100m/edufineweb_train_000001.npy (100000000,) uint16 0 50256
data/fineweb_edu_100m/edufineweb_val_000000.npy (4000000,) uint16 0 50256
```

## Test Results

```text
35 passed in 2.26s
```

## Ruff Result

```text
All checks passed!
```

## Sanity Run Result

Run directory:

```text
runs/baseline_15m_fineweb100m_sanity_seed1
```

Training completed for `20` steps. Summary:

```json
{
  "max_step": 20,
  "train_event_count": 20,
  "val_event_count": 3,
  "initial_val_loss": 10.908893585205078,
  "final_val_loss": 8.215901374816895,
  "best_val_loss": 8.215901374816895,
  "initial_val_perplexity": 54660.334833045774,
  "final_val_perplexity": 3699.3091863806258,
  "median_tokens_per_sec": 104956.08126630174,
  "peak_vram_mb": 3240.92431640625,
  "checkpoint_count": 1
}
```

Run verification with complete-training, sample, eval-loss, and HellaSwag expectations
passed.

## Eval Loss Result

```json
{
  "checkpoint": "runs/baseline_15m_fineweb100m_sanity_seed1/checkpoints/ckpt_last.pt",
  "split": "val",
  "steps": 5,
  "loss": 8.215901374816895,
  "perplexity": 3699.3091863806258
}
```

## Generation Result

Generation ran successfully from `ckpt_last.pt` with prompt:

```text
The history of mathematics
```

The sample was text-like but low quality and repetitive, which is expected after only
20 training steps.

## HellaSwag Bounded Result

```json
{
  "checkpoint": "runs/baseline_15m_fineweb100m_sanity_seed1/checkpoints/ckpt_last.pt",
  "split": "val",
  "num_total": 100,
  "num_correct_norm": 27,
  "accuracy_norm": 0.27
}
```

## Full Baseline Run

Not run in this verification pass. At the observed sanity-run throughput, the
3000-step 15M baseline is materially longer than a quick harness verification. Run:

```bash
uv run scripts/train.py --config configs/baseline_15m_fineweb100m.yaml --overwrite
```

Then verify:

```bash
uv run scripts/verify_run.py --run_dir runs/baseline_15m_fineweb100m_seed1 --expect-complete-training --expect-sample
```

## Known Limitations

- `trilinear_cp` remains unimplemented and is excluded from baseline QC.
- `torch.compile` is rejected by config validation for baseline QC.
- DDP is present but not part of the tested baseline completion path.
- OpenAI Evals is not used for this stage.
- `lm-evaluation-harness` is deferred until HF export exists.

