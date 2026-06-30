# Baseline Harness Verification

Date: 2026-06-29

Git commit at sanity verification start: `f71e790381aa7982db51663639d342a8b09edb65`

Git commit at full baseline run: `0760b275d46a5c920d79761609b59600d602f6f8`

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

Manifest written during pre-experiment hardening:

```text
data/fineweb_edu_100m/manifest.json
manifest sha256: 3302a779a89ee9f77a0c5717a963dd2744b5ee89dfef56b8c0d098cb61718f17
train shard sha256: 7bc89b5e75a6eba3e471c5434b03e98dd3be6aaa8ce043a9aae564bf51e25893
val shard sha256: efb01e4b8dad9ce4aa906ca8afbb36bd0329d4135e00741556eb4a70689f784c
```

## Test Results

```text
50 passed in 4.97s
```

## Ruff Result

```text
All checks passed!
```

Final QC after the full baseline documentation updates:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run scripts/verify_cuda.py
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes
uv run scripts/verify_run.py --run_dir runs/baseline_15m_fineweb100m_seed1 --expect-complete-training --expect-sample --expect-eval-loss --expect-hellaswag
uv run scripts/summarize_run.py --run_dir runs/baseline_15m_fineweb100m_seed1
uv run scripts/inspect_model_config.py --config configs/baseline_30m_fineweb100m.yaml
```

All final QC commands passed.

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

The full standard-attention baseline run completed on 2026-06-29. The config was not
changed from `configs/baseline_15m_fineweb100m.yaml`.

```bash
uv run scripts/train.py --config configs/baseline_15m_fineweb100m.yaml --overwrite
```

Post-run commands:

```bash
uv run scripts/verify_run.py --run_dir runs/baseline_15m_fineweb100m_seed1 --expect-complete-training --expect-sample
uv run scripts/eval_loss.py --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt --data_root data/fineweb_edu_100m
uv run scripts/eval_generate.py --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt --prompt "The history of mathematics"
uv run scripts/eval_hellaswag.py --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt --max_examples 100
uv run scripts/summarize_run.py --run_dir runs/baseline_15m_fineweb100m_seed1
uv run scripts/verify_run.py --run_dir runs/baseline_15m_fineweb100m_seed1 --expect-complete-training --expect-sample --expect-eval-loss --expect-hellaswag
```

Training completed:

```json
{
  "run_dir": "runs/baseline_15m_fineweb100m_seed1",
  "max_step": 3000,
  "train_event_count": 301,
  "val_event_count": 13,
  "initial_val_loss": 10.910149574279785,
  "final_val_loss": 4.081209182739258,
  "best_val_loss": 4.081209182739258,
  "initial_val_perplexity": 54729.03074804456,
  "final_val_perplexity": 59.2170307875361,
  "median_tokens_per_sec": 107022.7422894312,
  "peak_vram_mb": 3240.92431640625,
  "checkpoint_count": 3
}
```

The first metric timestamp was `2026-06-29T21:50:10+00:00`; the final checkpoint
event timestamp was `2026-06-29T23:58:11+00:00`, for about 2h08m wall-clock runtime.
PyTorch peak allocated VRAM was 3240.92 MB. A concurrent `nvidia-smi` sample during
training reported about 12 GB device memory in use.

Full-run verifier result:

```json
{
  "run_dir": "runs/baseline_15m_fineweb100m_seed1",
  "max_step": 3000,
  "train_event_count": 301,
  "val_event_count": 13,
  "checkpoint_event_count": 3,
  "ok": true
}
```

Full-run eval loss result:

```json
{
  "checkpoint": "runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt",
  "split": "val",
  "steps": 20,
  "loss": 4.081209182739258,
  "perplexity": 59.2170307875361
}
```

Full-run generation result:

```text
Prompt: The history of mathematics
Output file: runs/baseline_15m_fineweb100m_seed1/samples/sample_step_last.txt
```

The generated samples were text-like and topical, but still low-quality as expected
for this small baseline and token budget.

Full-run bounded HellaSwag result:

```json
{
  "checkpoint": "runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt",
  "split": "val",
  "num_total": 100,
  "num_correct_norm": 34,
  "accuracy_norm": 0.34
}
```

## Pre-Experiment Hardening Result

Completed on 2026-06-30 before starting novel attention work.

Added accurate-size naming:

```text
configs/baseline_15m_fineweb100m.yaml   historical completed run name
configs/baseline_30m_fineweb100m.yaml   accurate-size alias for new runs
```

Added standard-attention config ladder:

```text
configs/baseline_16m_fineweb100m.yaml
configs/baseline_30m_fineweb100m.yaml
configs/baseline_70m_fineweb300m.yaml
configs/baseline_125m_fineweb1b.yaml
```

Inspected model sizes:

```text
16M tier: 16025856 excluding positional, 16288000 including positional
30M tier: 29938560 excluding positional, 30331776 including positional
70M tier: 69810688 excluding positional, 70334976 including positional
125M tier: 123587328 excluding positional, 124373760 including positional
```

Added data-manifest workflow:

```bash
uv run scripts/write_data_manifest.py --data_root data/fineweb_edu_100m --out data/fineweb_edu_100m/manifest.json
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes
```

Added runtime-memory fields for new training rows:

```text
peak_vram_allocated_mb
peak_vram_reserved_mb
current_vram_allocated_mb
current_vram_reserved_mb
nvidia_smi_memory_mb
```

The historical run only has `peak_vram_mb`; summarization maps that to
`peak_vram_allocated_mb` for backward compatibility.

Added resume hardening:

```text
--overwrite and --resume are mutually exclusive
resume checks model config compatibility
resume checks B/T/total_batch_size compatibility
resume checks optimizer-relevant fields
resume checks data manifests when both run and data root have one
resume writes resume_from.txt
resume appends a resume event to metrics
```

Added comparison and export surfaces:

```text
docs/architecture_experiment_contract.md
docs/upstream_borrowing_audit.md
reports/schema/run_summary.schema.json
scripts/compare_runs.py
scripts/export_hf.py
```

`scripts/export_hf.py` is an honest nonzero stub. `lm-evaluation-harness` remains
deferred until HF export is implemented and verified.

## Known Limitations

- `trilinear_cp` remains unimplemented and is excluded from baseline QC.
- `torch.compile` is rejected by config validation for baseline QC.
- DDP is present but not part of the tested baseline completion path.
- OpenAI Evals is not used for this stage.
- `lm-evaluation-harness` is deferred until HF export exists.
