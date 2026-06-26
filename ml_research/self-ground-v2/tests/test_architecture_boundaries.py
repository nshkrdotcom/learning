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


def test_core_cli_sdk_and_assessments_have_no_heavy_ml_or_network_imports() -> None:
    banned = {
        "torch",
        "transformer_lens",
        "sae_lens",
        "numpy",
        "scipy",
        "pandas",
        "nnsight",
        "pyvene",
        "requests",
        "httpx",
        "aiohttp",
        "urllib3",
        "rdflib",
        "networkx",
    }
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


def test_core_dependencies_do_not_add_heavy_execution_or_graph_stacks() -> None:
    banned = {
        "torch",
        "transformer-lens",
        "saelens",
        "numpy",
        "scipy",
        "pandas",
        "nnsight",
        "pyvene",
        "requests",
        "httpx",
        "aiohttp",
        "rdflib",
        "networkx",
    }
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {
        item.split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].split(">", 1)[0].lower()
        for item in pyproject["project"]["dependencies"]
    }

    assert not dependencies & banned


def test_mechledger_does_not_define_execution_framework_schemas() -> None:
    banned_class_names = {
        "PatchSpec",
        "ActivationSource",
        "ActivationTarget",
        "InterventionSpec",
        "ModelExecutor",
        "ExecutionBackend",
    }
    banned_function_names = {
        "execute_intervention",
        "run_model",
        "load_transformer",
        "compute_activations",
        "compute_circuit_graph",
        "compute_weight_analysis",
    }
    for path in Path("src/mechledger").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert node.name not in banned_class_names, (
                    f"execution-framework schema found in {path}: {node.name}"
                )
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                assert node.name not in banned_function_names, (
                    f"execution-framework function found in {path}: {node.name}"
                )


def test_docs_state_core_execution_non_goals() -> None:
    readme = Path("README.md").read_text(encoding="utf-8").lower()
    usage = Path("docs/USAGE.md").read_text(encoding="utf-8").lower()
    docs = readme + "\n" + usage

    assert "does not execute" in docs
    assert "heavy ml dependencies" in docs or "heavy machine-learning dependencies" in docs
    assert "ro-crate" in docs and "not canonical" in docs
