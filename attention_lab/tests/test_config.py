from __future__ import annotations

from copy import deepcopy

import pytest

from attention_lab.training.config import load_config, validate_config


def test_valid_baseline_sanity_config_loads(repo_root):
    config = load_config(repo_root / "configs" / "baseline_15m_fineweb100m_sanity.yaml")
    assert config["run"]["name"] == "baseline_15m_fineweb100m_sanity_seed1"
    assert config["model"]["attention_type"] == "standard"


def test_baseline_config_ladder_loads(repo_root):
    config_names = [
        "baseline_15m_fineweb100m.yaml",
        "baseline_16m_fineweb100m.yaml",
        "baseline_30m_fineweb100m.yaml",
        "baseline_70m_fineweb300m.yaml",
        "baseline_125m_fineweb1b.yaml",
    ]
    for config_name in config_names:
        config = load_config(repo_root / "configs" / config_name)
        assert config["model"]["attention_type"] == "standard"
        assert config["train"]["total_batch_size"] % (config["train"]["B"] * config["train"]["T"]) == 0


def test_historical_15m_and_30m_alias_have_same_model_shape(repo_root):
    historical = load_config(repo_root / "configs" / "baseline_15m_fineweb100m.yaml")
    canonical = load_config(repo_root / "configs" / "baseline_30m_fineweb100m.yaml")
    assert historical["model"] == canonical["model"]
    assert historical["data"]["vocab_size"] == canonical["data"]["vocab_size"]


def test_missing_required_section_fails(tiny_config, tmp_path):
    config = tiny_config(tmp_path, tmp_path / "data")
    del config["data"]
    with pytest.raises(ValueError, match="missing sections"):
        validate_config(config, source=tmp_path / "bad.yaml")


def test_unknown_model_key_fails(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["model"]["not_a_real_key"] = 1
    with pytest.raises(ValueError, match="Unknown model config keys"):
        validate_config(config)


def test_unknown_train_key_fails(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["train"]["learningrate"] = 0.0006
    with pytest.raises(ValueError, match="Unknown train config keys"):
        validate_config(config)


def test_unknown_data_key_fails(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["data"]["toknizer"] = "gpt2"
    with pytest.raises(ValueError, match="Unknown data config keys"):
        validate_config(config)


def test_unknown_sample_key_fails(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["sample"]["sample_evey"] = 1
    with pytest.raises(ValueError, match="Unknown sample config keys"):
        validate_config(config)


def test_invalid_dtype_fails(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["train"]["dtype"] = "fp8"
    with pytest.raises(ValueError, match="train.dtype"):
        validate_config(config)


def test_n_embd_must_be_divisible_by_n_head(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["model"]["n_head"] = 3
    with pytest.raises(ValueError, match="n_embd"):
        validate_config(config)


def test_total_batch_size_must_be_divisible_by_micro_batch(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["train"]["total_batch_size"] = 17
    with pytest.raises(ValueError, match="total_batch_size"):
        validate_config(config)


def test_unimplemented_trilinear_config_not_runnable_baseline(repo_root):
    config_path = repo_root / "configs" / "experimental" / "trilinear_cp_15m_fineweb100m.yaml"
    with pytest.raises(ValueError, match="experimental"):
        load_config(config_path)


def test_compile_requires_explicit_experimental_flag(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["train"]["compile"] = True
    with pytest.raises(ValueError, match="compile"):
        validate_config(config)
