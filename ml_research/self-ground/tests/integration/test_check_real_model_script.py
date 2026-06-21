from __future__ import annotations

import json
import subprocess

import pytest


@pytest.mark.integration
def test_check_real_model_script_writes_activation_artifact(tmp_path) -> None:
    out = tmp_path / "check_real_model.json"

    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/check_real_model.py",
            "--device",
            "cpu",
            "--out",
            str(out),
        ],
        check=True,
    )

    artifact = json.loads(out.read_text())
    assert artifact["status"] == "ok"
    assert artifact["activation_shape"][0] == 4
    assert artifact["activation_shape"][2] == 512
