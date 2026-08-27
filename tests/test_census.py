"""
tests/test_census.py — qmd/census.py (exact counts, orderly generation, sampling)
and the parallel pipeline path.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qmd import census  # noqa: E402
from qmd.core import canonical_form, generate_seed_quivers, run_generation, to_matrix  # noqa: E402


@pytest.mark.parametrize("n,h,expected", [
    (1, 2, 1), (2, 2, 3), (3, 2, 25), (4, 2, 695),      # the published n<=4 census
    (3, 1, 7), (4, 1, 42), (5, 1, 582), (3, 10, 1561),
])
def test_count_quivers_known_values(n, h, expected):
    assert census.count_quivers(n, h) == expected


def test_count_matches_brute_force_small_cells():
    for n in (2, 3):
        for h in (1, 2, 3):
            seen = set()
            import itertools
            upper = [(i, j) for i in range(n) for j in range(i + 1, n)]
            for combo in itertools.product(range(-h, h + 1), repeat=len(upper)):
                rows = [[0] * n for _ in range(n)]
                for (i, j), v in zip(upper, combo):
                    rows[i][j], rows[j][i] = v, -v
                seen.add(canonical_form(to_matrix(rows)))
            assert len(seen) == census.count_quivers(n, h), (n, h)


def test_orderly_generation_equals_brute_force():
    brute = set(generate_seed_quivers(4, 2))
    for n in range(1, 5):
        cell = set(census.census_seeds(n, 2))
        assert cell == {m for m in brute if len(m) == n}
        assert len(cell) == census.count_quivers(n, 2)


@pytest.mark.parametrize("n,h", [(4, 3), (5, 1), (3, 5)])
def test_orderly_generation_count_and_canonicity(n, h):
    reps = list(census.generate_cell(n, h))
    assert len(reps) == census.count_quivers(n, h)
    assert all(census.is_key_minimal(m) for m in reps)
    assert len({canonical_form(m) for m in reps}) == len(reps)   # distinct classes


def test_key_minimal_form_is_isomorphism_invariant():
    import random
    rng = random.Random(11)
    for _ in range(100):
        n = rng.randint(2, 5)
        m = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                v = rng.randint(-2, 2)
                m[i][j], m[j][i] = v, -v
        m = to_matrix(m)
        p = list(range(n)); rng.shuffle(p)
        m2 = tuple(tuple(m[p[i]][p[j]] for j in range(n)) for i in range(n))
        assert census.key_minimal_form(m) == census.key_minimal_form(m2)


def test_orderly_generation_parallel_matches_serial():
    serial = census.census_seeds(4, 2)
    parallel = census.census_seeds(4, 2, workers=3)
    assert serial == parallel


def test_sample_cell_is_distinct_canonical_and_in_cell():
    s = census.sample_cell(5, 3, 50, seed=1)
    assert len(s) == 50 and len(set(s)) == 50
    assert all(canonical_form(m) == m for m in s)
    assert all(max(abs(x) for row in m for x in row) <= 3 for m in s)
    assert census.sample_cell(5, 3, 50, seed=1) == s          # deterministic


def test_parallel_run_generation_matches_serial():
    seeds = census.census_seeds(4, 2)
    a = run_generation(max_vertices=4, bound=2, ranks=[4], seeds=seeds)
    b = run_generation(max_vertices=4, bound=2, ranks=[4], seeds=seeds, workers=4)
    assert set(a.quivers) == set(b.quivers)
    assert set(a.classes) == set(b.classes)
    assert a.membership == b.membership
    for k in a.classes:
        assert a.classes[k].labeled_quivers == b.classes[k].labeled_quivers
        assert a.classes[k].exploration == b.classes[k].exploration
    assert (a.open_open_gluings, a.closed_closed_merges) == (b.open_open_gluings, b.closed_closed_merges)
