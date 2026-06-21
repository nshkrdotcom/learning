from __future__ import annotations

import shutil
from pathlib import Path


class LocalArtifactStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.artifact_dir = self.run_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts: dict[str, str] = {}

    def register_artifact(self, name: str, path: Path) -> str:
        source = Path(path)
        if not source.exists():
            raise ValueError(f"artifact path does not exist: {source}")
        target = self.artifact_dir / name
        if source.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        self._artifacts[name] = str(target)
        return str(target)

    @property
    def artifacts(self) -> dict[str, str]:
        return dict(self._artifacts)
