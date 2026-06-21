from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    runpy.run_path(str(root / "scripts" / "run_real_residual_intervention.py"), run_name="__main__")
