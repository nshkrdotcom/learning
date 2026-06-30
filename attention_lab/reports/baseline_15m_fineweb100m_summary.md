# Baseline 15M FineWeb-Edu 100M Summary

Date: 2026-06-29

Git commit at run time: `0760b275d46a5c920d79761609b59600d602f6f8`

## Model

- Config: `configs/baseline_15m_fineweb100m.yaml`
- Attention: `standard`
- Block size: `1024`
- Layers: `6`
- Heads: `6`
- Embedding size: `384`
- Dropout: `0.0`
- Bias: `false`
- Vocabulary size: `50304`
- Parameters, excluding positional embeddings: `29938560`
- Parameters, including positional embeddings: `30331776`

The run name preserves the project shorthand `15m`, but the current model code reports
the parameter counts above.

## Data

- Dataset source: FineWeb-Edu token shards prepared by `scripts/prepare_fineweb_edu.py`
- Train shard: `data/fineweb_edu_100m/edufineweb_train_000001.npy`
- Validation shard: `data/fineweb_edu_100m/edufineweb_val_000000.npy`
- Train shard size: `100000000` tokens
- Validation shard size: `4000000` tokens
- Tokenizer: GPT-2 BPE via `tiktoken`

## Training

- Device: `cuda`
- Dtype: `bfloat16`
- Compile: `false`
- Micro batch: `B=4`, `T=1024`
- Total batch size: `262144`
- Steps: `3000`
- Training token positions: `786432000`
- Optimizer: AdamW, betas `(0.9, 0.95)`, eps `1e-8`, fused on CUDA
- Weight decay: `0.1`
- Learning rate: `0.0006`
- Minimum learning rate: `0.00006`
- Warmup steps: `100`
- Validation cadence: every `250` steps, `20` validation batches
- Checkpoint cadence: every `1000` steps

## Result

- Initial validation loss: `10.910149574279785`
- Final validation loss: `4.081209182739258`
- Best validation loss: `4.081209182739258`
- Initial validation perplexity: `54729.03074804456`
- Final validation perplexity: `59.2170307875361`
- Median tokens/sec: `107022.7422894312`
- Peak VRAM from PyTorch allocated metric: `3240.92431640625` MB
- Checkpoints written: `3`
- Final checkpoint: `runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt`
- Final sample: `runs/baseline_15m_fineweb100m_seed1/samples/sample_step_last.txt`
- Eval loss JSON: `runs/baseline_15m_fineweb100m_seed1/evals/val_loss.json`
- HellaSwag JSON: `runs/baseline_15m_fineweb100m_seed1/evals/hellaswag.json`
- Run summary JSON: `runs/baseline_15m_fineweb100m_seed1/evals/run_summary.json`

## Bounded HellaSwag

- Split: `val`
- Examples: `100`
- Normalized correct: `34`
- Normalized accuracy: `0.34`

## Verification

`scripts/verify_run.py` passed with:

```text
--expect-complete-training --expect-sample --expect-eval-loss --expect-hellaswag
```
