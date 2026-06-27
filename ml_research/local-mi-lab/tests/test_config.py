from __future__ import annotations

from local_mi_lab.config import load_config, resource_limit, selected_layers


def test_config_loads() -> None:
    config = load_config("configs/gpt2_small_induction.yaml")
    assert config["experiment"]["name"] == "gpt2_small_induction"


def test_default_model_is_gpt2_small() -> None:
    config = load_config("configs/gpt2_small_induction.yaml")
    assert config["model"]["name"] == "gpt2-small"


def test_resource_budgets_parse() -> None:
    config = load_config("configs/gpt2_small_induction.yaml")
    assert resource_limit(config, "max_disk_cache_gb") == 100
    assert resource_limit(config, "max_activation_cache_gb_per_run") == 20
    assert resource_limit(config, "max_ram_gb_per_run") == 64
    assert resource_limit(config, "max_gpu_vram_fraction") == 0.80


def test_auto_even_layers_parse() -> None:
    config = load_config("configs/gpt2_small_induction.yaml")
    assert selected_layers(config, n_layers=12) == [0, 2, 4, 7, 9, 11]
