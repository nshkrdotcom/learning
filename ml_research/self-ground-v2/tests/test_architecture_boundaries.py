from __future__ import annotations

import ast
import tomllib
from pathlib import Path


def test_cli_entrypoint_exists_and_typer_stays_out_of_core() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["mechledger"] == "mechledger.cli:app"
    for path in Path("src/mechledger").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    node.func.attr == "Typer"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "typer"
                ):
                    assert path.match("src/mechledger/cli.py") or path.match(
                        "src/mechledger/commands/*.py"
                    ), f"Typer app found outside CLI modules in {path}"


def test_core_cli_sdk_and_assessments_have_no_heavy_ml_imports() -> None:
    banned = {"torch", "transformer_lens", "sae_lens", "numpy", "scipy", "pandas"}
    for path in Path("src/mechledger").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
                assert not names & banned, f"banned import in {path}: {names & banned}"
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in banned, (
                    f"banned import in {path}: {node.module}"
                )


def test_core_dependencies_do_not_add_heavy_numerical_stacks() -> None:
    banned = {"torch", "transformer-lens", "saelens", "numpy", "scipy", "pandas"}
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {
        item.split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].split(">", 1)[0].lower()
        for item in pyproject["project"]["dependencies"]
    }

    assert not dependencies & banned
