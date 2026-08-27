"""
tests/test_golden_ids.py — citation safety.

Every Q.* / MC.* id in the published n<=4 census is frozen in
tests/golden/ids-n4.json. Any change to canonical forms, hashing, BFS
gluing, or the seed set that re-keys the database fails here first.

Regenerate ONLY for a deliberate, documented re-keying:
    python tests/test_golden_ids.py --regenerate
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "ids-n4.json")


def _snapshot(r4):
    return {
        "bound": 2,
        "max_vertices": 4,
        "quivers": sorted(r4.quivers),
        "classes": sorted(r4.classes),
        "membership": dict(sorted(r4.membership.items())),
        "class_sizes": {mc_id: mc.labeled_size for mc_id, mc in sorted(r4.classes.items())},
    }


def test_ids_match_golden(r4):
    if not os.path.exists(GOLDEN):
        pytest.fail(f"{GOLDEN} missing — run `python tests/test_golden_ids.py --regenerate`")
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    now = _snapshot(r4)
    assert now["quivers"] == golden["quivers"], "quiver IDs changed — this re-keys the database"
    assert now["classes"] == golden["classes"], "mutation-class IDs changed — this re-keys the database"
    assert now["membership"] == golden["membership"]
    assert now["class_sizes"] == golden["class_sizes"]


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        from qmd.core import run_generation
        snap = _snapshot(run_generation(max_vertices=4, bound=2))
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=0, sort_keys=True)
            f.write("\n")
        print(f"wrote {GOLDEN}: {len(snap['quivers'])} quivers, {len(snap['classes'])} classes")
    else:
        sys.exit(pytest.main([__file__, "-q"]))
