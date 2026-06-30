from __future__ import annotations

import pytest

from attention_lab.training.config import load_config
from attention_lab.training.validate_experiment import validate_experiment


def test_validate_e002_first_build_configs():
    result = validate_experiment("E002_multitrack_qkv_shift_register")
    assert result["ok"] is True
    assert result["config_count"] == 11
    assert result["runnable_config_count"] == 5
    assert result["unimplemented_config_count"] == 6


def test_e002_canonical_multi_qkv_configs_load_and_old_skeletons_remain_unimplemented(repo_root):
    config_dir = repo_root / "configs" / "experiments" / "E002_multitrack_qkv_shift_register"
    for name in (
        "standard_refactor_control_30m_seed1.yaml",
        "multi_qkv_static_3track_global_30m_seed1.yaml",
        "multi_qkv_train_rotation_3track_global_30m_seed1.yaml",
        "multi_qkv_position_rotation_3track_global_30m_seed1.yaml",
    ):
        assert load_config(config_dir / name)

    with pytest.raises(ValueError, match="experimental"):
        load_config(config_dir / "multi_qkv_softmix_3track_30m_seed1.yaml")
