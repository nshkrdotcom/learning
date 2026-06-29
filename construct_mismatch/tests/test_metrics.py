from __future__ import annotations

import numpy as np

from construct_mismatch.metrics import signed_logit_diff


def test_signed_logit_diff_sign_convention() -> None:
    raw = np.asarray([2.0, 2.0, -1.5, -1.5])
    labels = ["class_a", "class_b", "class_a", "class_b"]
    signed = signed_logit_diff(raw, labels)
    assert signed.tolist() == [2.0, -2.0, -1.5, 1.5]
