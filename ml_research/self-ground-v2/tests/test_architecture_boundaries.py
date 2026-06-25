from __future__ import annotations

import ast
import tomllib
from pathlib import Path


def test_no_cli_entrypoint_or_typer_app_in_milestone_neg1() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "scripts" not in pyproject.get("project", {})
    for path in Path("src/mechledger").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert not (
                    node.func.attr == "Typer"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "typer"
                ), f"Typer app found in {path}"


def test_core_and_assessment_modules_have_no_heavy_ml_imports() -> None:
    banned = {"torch", "transformer_lens", "sae_lens", "numpy", "scipy"}
    for path in list(Path("src/mechledger/core").rglob("*.py")) + list(
        Path("src/mechledger/assessments").rglob("*.py")
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
                assert not names & banned, f"banned import in {path}: {names & banned}"
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in banned, (
                    f"banned import in {path}: {node.module}"
                )
