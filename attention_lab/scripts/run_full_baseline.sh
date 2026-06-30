#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv run scripts/train.py --config configs/baseline_15m_fineweb100m.yaml --overwrite
uv run scripts/verify_run.py --run_dir runs/baseline_15m_fineweb100m_seed1 --expect-complete-training --expect-sample
uv run scripts/eval_loss.py --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt --data_root data/fineweb_edu_100m
uv run scripts/eval_generate.py --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt --prompt "The history of mathematics"
uv run scripts/eval_hellaswag.py --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt --max_examples 100
uv run scripts/summarize_run.py --run_dir runs/baseline_15m_fineweb100m_seed1
uv run scripts/verify_run.py --run_dir runs/baseline_15m_fineweb100m_seed1 --expect-complete-training --expect-sample --expect-eval-loss --expect-hellaswag
