from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter

from self_ground.negation import (
    DECOY_MARKERS,
    FAMILIES,
    contains_negation_marker,
    generate_negation_pairs,
)


def test_generation_is_deterministic_and_has_exactly_four_families() -> None:
    first = generate_negation_pairs(per_family=3, seed=123)
    second = generate_negation_pairs(per_family=3, seed=123)

    assert [p.model_dump() for p in first] == [p.model_dump() for p in second]
    assert len(first) == 12

    counts = Counter(pair.template_family for pair in first)
    assert set(counts) == set(FAMILIES)
    assert len(counts) == 4
    assert all(count == 3 for count in counts.values())


def test_each_pair_has_four_clean_conditions_and_markers() -> None:
    pairs = generate_negation_pairs(per_family=4, seed=7)

    for pair in pairs:
        assert pair.x_pos
        assert pair.x_neg
        assert pair.x_para
        assert pair.x_decoy
        assert pair.changed_variable == "negation_presence"
        assert pair.held_constant
        assert 0.0 <= pair.control_purity <= 1.0

        assert contains_negation_marker(pair.x_pos)
        assert contains_negation_marker(pair.x_para)
        assert not contains_negation_marker(pair.x_neg)
        assert any(marker in pair.x_decoy.lower() for marker in DECOY_MARKERS)


def test_ids_are_stable_across_processes() -> None:
    code = (
        "from self_ground.negation import generate_negation_pairs; "
        "import json; "
        "print(json.dumps([p.id for p in generate_negation_pairs(per_family=2, seed=99)]))"
    )

    ids_one = json.loads(subprocess.check_output([sys.executable, "-c", code], text=True))
    ids_two = json.loads(subprocess.check_output([sys.executable, "-c", code], text=True))

    assert ids_one == ids_two
    assert len(set(ids_one)) == len(ids_one)
