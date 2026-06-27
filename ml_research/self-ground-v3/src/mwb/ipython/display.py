from __future__ import annotations

from typing import Any

from rich.console import Console

console = Console()


def display_card(
    card_or_ref: Any, *, format: str = "markdown", include_artifacts: bool = False
) -> Any:
    console.print(card_or_ref)
    return card_or_ref


def display_run(run_or_ref: Any, *, sections: list[str] | None = None) -> Any:
    console.print(run_or_ref)
    return run_or_ref


def display_features(
    features_or_ref: Any, *, top_k: int = 20, sort_by: str = "score", contrast: str | None = None
) -> Any:
    console.print(features_or_ref)
    return features_or_ref


def display_graph(graph_or_ref: Any = None, *, view: str = "lineage", depth: int = 2) -> Any:
    console.print(graph_or_ref)
    return graph_or_ref
