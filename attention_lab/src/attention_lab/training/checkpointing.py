import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    out_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
    train_loss: float | None = None,
    val_loss: float | None = None,
) -> Path:
    checkpoint_dir = Path(out_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    step_path = checkpoint_dir / f"ckpt_step_{step:06d}.pt"
    last_path = checkpoint_dir / "ckpt_last.pt"
    tmp_path = step_path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, step_path)
    torch.save(checkpoint, last_path)
    return step_path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)

