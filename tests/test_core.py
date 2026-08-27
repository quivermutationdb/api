"""
tests/test_core.py — core.py + canonicalize.py

Run with:  python -m pytest tests/ -q
(or `python tests/test_core.py`, which just delegates to pytest.)
"""
import os
import sys
from collections import defaultdict
from itertools import permutations as _perms

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qmd.canonicalize import (  # noqa: E402
    PermutationCanonicalizer,
    _apply_permutation,
    _lex_key,
    active_backend,
    are_isomorphic,
    canonical_backend,
    canonical_form,
    lexmin_form,
    verify_with_fallback,
)
from qmd.core import (  # noqa: E402
    _RawOrbit,
    _UnionFind,
    explore_mutation_class,
    generate_seed_quivers,
    is_bounded,
    is_skew_symmetric,
    max_edge,
    mutate,
    mutation_class_id,
    quiver_id,
    to_lists,
    to_matrix,
)

A2 = to_matrix([[0, 1], [-1, 0]])
A2_flip = to_matrix([[0, -1], [1, 0]])                       # isomorphic to A2
A3 = to_matrix([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])
A3_rev = to_matrix([[0, -1, 0], [1, 0, -1], [0, 1, 0]])      # A3 reversed — isomorphic
D4 = to_matrix([[0, 1, 1, 1], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0]])
D4_perm = to_matrix([[0, -1, 0, 0], [1, 0, 1, 1], [0, -1, 0, 0], [0, -1, 0, 0]])
zero2 = to_matrix([[0, 0], [0, 0]])
kronecker = to_matrix([[0, 2], [-2, 0]])

perm_canon = PermutationCanonicalizer()

ISO_PAIRS = [("A2", A2, A2_flip), ("A3", A3, A3_rev), ("D4", D4, D4_perm)]
NAMED = [("A2", A2), ("A2_flip", A2_flip), ("A3", A3), ("A3_rev", A3_rev),
         ("D4", D4), ("D4_perm", D4_perm), ("zero2", zero2), ("kronecker", kronecker)]


# ---------------------------------------------------------------------------
# 1. PermutationCanonicalizer
# ---------------------------------------------------------------------------

def test_apply_permutation_identity_and_reversal():
    assert _apply_permutation(A3, (0, 1, 2)) == A3
    rev = _apply_permutation(A3, (2, 1, 0))
    assert is_skew_symmetric(rev)
    assert rev == A3_rev


@pytest.mark.parametrize("label,a,b", ISO_PAIRS)
def test_canonical_form_collapses_isomorphic(label, a, b):
    assert perm_canon.canonical_form(a) == perm_canon.canonical_form(b)


def test_canonical_form_separates_non_isomorphic():
    cf = perm_canon.canonical_form
    assert cf(A2) != cf(A3)
    assert cf(A3) != cf(D4)
    assert cf(A2) != cf(zero2)


def test_canonical_form_idempotent_and_skew():
    cf_A3 = perm_canon.canonical_form(A3)
    assert perm_canon.canonical_form(cf_A3) == cf_A3
    assert is_skew_symmetric(cf_A3)
    assert is_skew_symmetric(perm_canon.canonical_form(D4))


def test_canonical_form_is_lex_min():
    """The ID key is the lex-min relabeling — the published definition."""
    for _, m in NAMED:
        n = len(m)
        brute = min((_apply_permutation(m, p) for p in _perms(range(n))), key=_lex_key)
        assert perm_canon.canonical_form(m) == brute


# ---------------------------------------------------------------------------
# 2. Module-level dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,m", NAMED)
def test_dispatch_agrees_with_permutation_backend(label, m):
    # Whatever backend is active, the ID key must be the lex-min form.
    assert canonical_form(m) == perm_canon.canonical_form(m)


def test_id_key_backend_is_always_lexmin():
    assert canonical_backend() == "lexmin"


def test_lexmin_branch_and_bound_matches_brute_force():
    """The production canonicalizer vs the n! oracle on random matrices."""
    import random
    rng = random.Random(4242)
    for _ in range(300):
        n = rng.randint(1, 6)
        m = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                w = rng.choice([0, 0, 0, 1, -1, 1, -1, 2, -2, 3])
                m[i][j], m[j][i] = w, -w
        m = to_matrix(m)
        assert lexmin_form(m) == perm_canon.canonical_form(m)
        assert verify_with_fallback(m)


def test_are_isomorphic():
    assert are_isomorphic(A2, A2_flip)
    assert are_isomorphic(A3, A3_rev)
    assert are_isomorphic(D4, D4_perm)
    assert not are_isomorphic(A2, A3)


@pytest.mark.skipif(active_backend() != "nauty", reason="nauty backend not active")
def test_nauty_certificates_agree_with_lexmin_equality():
    import random
    rng = random.Random(99)
    for _ in range(200):
        n = rng.randint(2, 6)
        m = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                w = rng.choice([0, 0, 1, -1, 2, -2])
                m[i][j], m[j][i] = w, -w
        m = to_matrix(m)
        perm = list(range(n)); rng.shuffle(perm)
        relabeled = _apply_permutation(m, tuple(perm))
        other = to_matrix([[0] * n for _ in range(n)])
        assert are_isomorphic(m, relabeled)
        assert are_isomorphic(m, other) == (canonical_form(m) == canonical_form(other))


# ---------------------------------------------------------------------------
# 3. Gadget encoding (weight-2 edges)
# ---------------------------------------------------------------------------

def test_weight2_isomorphism_and_separation():
    w2 = to_matrix([[0, 2, 0], [-2, 0, 1], [0, -1, 0]])
    w2_perm = _apply_permutation(w2, (1, 0, 2))
    assert perm_canon.canonical_form(w2) == perm_canon.canonical_form(w2_perm)
    assert canonical_form(w2) == perm_canon.canonical_form(w2)
    w2_diff = to_matrix([[0, 2, 0], [-2, 0, 2], [0, -2, 0]])
    assert canonical_form(w2) != canonical_form(w2_diff)


# ---------------------------------------------------------------------------
# 4. Matrix helpers
# ---------------------------------------------------------------------------

def test_matrix_helpers():
    assert to_matrix(to_lists(A3)) == A3
    assert is_skew_symmetric(A2) and is_skew_symmetric(D4)
    assert not is_skew_symmetric(to_matrix([[0, 1], [1, 0]]))
    assert is_bounded(A2, 2)
    assert not is_bounded(to_matrix([[0, 3], [-3, 0]]), 2)
    assert max_edge(kronecker) == 2


# ---------------------------------------------------------------------------
# 5. Mutation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,m", [("A2", A2), ("A3", A3), ("D4", D4), ("kronecker", kronecker)])
def test_mutation_is_involution_and_skew(label, m):
    for k in range(len(m)):
        assert mutate(mutate(m, k), k) == m
        assert is_skew_symmetric(mutate(m, k))


def test_mutation_small_cases():
    assert all(mutate(zero2, k) == zero2 for k in range(2))
    assert mutate(A2, 0) == A2_flip
    assert mutate(A2, 1) == A2_flip


def test_mutation_hand_computed_weight2():
    # 0 -(2)-> 1 -> 2; mutate at 1: reverse arrows at 1, add b02 += 2*1 = 2.
    m = to_matrix([[0, 2, 0], [-2, 0, 1], [0, -1, 0]])
    expected = to_matrix([[0, -2, 2], [2, 0, -1], [-2, 1, 0]])
    assert mutate(m, 1) == expected


# ---------------------------------------------------------------------------
# 6. ID generation
# ---------------------------------------------------------------------------

def test_quiver_id_iso_invariant_and_separating():
    assert quiver_id(A2) == quiver_id(A2_flip)
    assert quiver_id(A3) == quiver_id(A3_rev)
    assert quiver_id(D4) == quiver_id(D4_perm)
    assert quiver_id(A2) != quiver_id(A3)
    assert quiver_id(A3) != quiver_id(D4)


def test_id_formats():
    qid = quiver_id(A2)
    assert qid.startswith("Q.n2.") and len(qid) == len("Q.n2.") + 16
    assert quiver_id(D4).startswith("Q.n4.")
    mcid = mutation_class_id(canonical_form(A2))
    assert mcid.startswith("MC.n2.") and len(mcid) == len("MC.n2.") + 16
    assert mcid == mutation_class_id(canonical_form(A2))


# ---------------------------------------------------------------------------
# 7. canonical_class_rep / mutation_class_id
# ---------------------------------------------------------------------------

def test_canonical_class_rep_is_brute_force_min():
    r = explore_mutation_class(A3)
    manual = min((_apply_permutation(m, p) for m in r.labeled_quivers
                  for p in _perms(range(len(m)))), key=_lex_key)
    assert r.canonical_rep == manual


def test_mc_id_invariant_to_seed_and_orbit_member():
    r_A3 = explore_mutation_class(A3)
    assert r_A3.mc_id == explore_mutation_class(A3_rev).mc_id
    assert explore_mutation_class(D4).mc_id == explore_mutation_class(D4_perm).mc_id
    for other in r_A3.labeled_quivers:
        if other != A3:
            assert explore_mutation_class(other).mc_id == r_A3.mc_id
            break


# ---------------------------------------------------------------------------
# 8. BFS exploration
# ---------------------------------------------------------------------------

def test_bfs_known_sizes():
    r_A2 = explore_mutation_class(A2)
    assert r_A2.labeled_size == 2 and r_A2.distinct_quiver_count == 1 and not r_A2.is_open
    r_A3 = explore_mutation_class(A3)
    assert r_A3.labeled_size == 14 and r_A3.distinct_quiver_count == 4 and not r_A3.is_open
    r_D4 = explore_mutation_class(D4)
    assert r_D4.labeled_size == 50 and r_D4.distinct_quiver_count == 6 and not r_D4.is_open
    r_zero = explore_mutation_class(zero2)
    assert r_zero.labeled_size == 1 and r_zero.distinct_quiver_count == 1 and not r_zero.is_open


def test_bfs_data_model_consistency():
    r = explore_mutation_class(A3)
    assert len(r.quiver_ids) == len(r.labeled_quivers)
    assert all(quiver_id(m) == qid for m, qid in zip(r.labeled_quivers, r.quiver_ids))
    assert is_skew_symmetric(r.canonical_rep) and is_bounded(r.canonical_rep)
    assert all(is_bounded(m, 2) for m in r.labeled_quivers)
    assert all(is_bounded(m, 2) for m in explore_mutation_class(D4).labeled_quivers)


@pytest.mark.parametrize("m", [A2, A3])
def test_closed_class_is_mutation_closed(m):
    res = explore_mutation_class(m)
    assert not res.is_open
    orbit = set(res.labeled_quivers)
    for mem in res.labeled_quivers:
        for k in range(len(mem)):
            mu = mutate(mem, k)
            assert mu in orbit or not is_bounded(mu, 2)


def test_node_cap_semantics():
    """cap without a crossing -> truncated; cap after a crossing -> still bound."""
    r = explore_mutation_class(D4, node_cap=10)
    assert r.exploration == "truncated" and r.is_open and r.labeled_size == 10
    # A class that crosses |b_ij| <= 2 quickly: Kronecker-ish rank-3 seed.
    m = to_matrix([[0, 2, -1], [-2, 0, 2], [1, -2, 0]])
    full = explore_mutation_class(m)
    assert full.exploration == "bound"
    capped = explore_mutation_class(m, node_cap=3)
    assert capped.exploration == "bound", "a crossing proves infinitude; the cap must not hide it"
    assert capped.labeled_size <= 3 + 1


def test_open_class_has_boundary():
    res = explore_mutation_class(to_matrix([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]), bound=2)
    if res.is_open:
        assert len(res.boundary_quivers) > 0


# ---------------------------------------------------------------------------
# 9. Seed generation
# ---------------------------------------------------------------------------

def test_seed_generation():
    seeds = generate_seed_quivers(max_vertices=4, bound=2)
    assert all(is_skew_symmetric(s) for s in seeds)
    assert all(is_bounded(s, 2) for s in seeds)
    assert {len(s) for s in seeds} == {1, 2, 3, 4}
    assert all(canonical_form(s) == s for s in seeds)
    ids = [quiver_id(s) for s in seeds]
    assert len(ids) == len(set(ids))
    assert len([s for s in seeds if len(s) == 1]) == 1
    assert len([s for s in seeds if len(s) == 2]) == 3      # {0}, {±1}, {±2}


# ---------------------------------------------------------------------------
# 10. Full pipeline
# ---------------------------------------------------------------------------

def test_pipeline_structure(r3, r4):
    assert r3.quivers and r3.classes and r3.membership
    assert set(r3.membership) <= set(r3.quivers)
    assert set(r3.membership.values()) <= set(r3.classes)
    assert all(canonical_form(m) == m for m in r4.quivers.values())
    assert all(is_bounded(m, 2) for m in r4.quivers.values())
    for r in (r3, r4):
        assert all(quiver_id(m) == qid for qid, m in r.quivers.items())
        assert all(mutation_class_id(mc.canonical_rep) == mcid for mcid, mc in r.classes.items())
    assert all(is_skew_symmetric(mc.canonical_rep) and is_bounded(mc.canonical_rep)
               for mc in r4.classes.values())
    assert all(mc.labeled_size > 0 for mc in r4.classes.values())


def test_pipeline_known_results(r4):
    assert quiver_id(A3) in r4.quivers and quiver_id(D4) in r4.quivers
    a3 = r4.classes[r4.membership[quiver_id(A3)]]
    d4 = r4.classes[r4.membership[quiver_id(D4)]]
    assert a3.labeled_size == 14
    assert d4.labeled_size == 50
    assert any(not mc.is_open for mc in r4.classes.values())
    assert r4.total_gluings >= 0


def test_pipeline_totals_n4(r4):
    """The published n<=4, bound-2 census (also asserted by the golden-ID test)."""
    assert len(r4.quivers) == 724
    assert len(r4.classes) == 178
    assert sum(1 for mc in r4.classes.values() if not mc.is_open) == 24


# ---------------------------------------------------------------------------
# 11. Gluing counters and orbit-uniqueness theorem
# ---------------------------------------------------------------------------

def test_gluing_counters(r4):
    assert r4.closed_closed_merges == 0
    assert r4.closed_open_merges == 0
    assert r4.open_open_gluings >= 0
    assert r4.total_gluings == (r4.closed_closed_merges + r4.closed_open_merges
                                + r4.open_open_gluings)
    for mc in r4.classes.values():
        if mc.merged_orbit_count > 1:
            assert mc.is_open
        if not mc.is_open:
            assert mc.merged_orbit_count == 1


def _simulate_union(orbits):
    """Only the union-find step of run_generation; raises like the real thing."""
    uf = _UnionFind(len(orbits))
    qid_map = defaultdict(list)
    for idx, orb in enumerate(orbits):
        for q in orb.qid_set:
            qid_map[q].append(idx)
    for q, idxs in qid_map.items():
        for i in range(1, len(idxs)):
            a, b = idxs[0], idxs[i]
            if uf.union(a, b):
                a_open, b_open = orbits[a].is_open, orbits[b].is_open
                if not a_open and not b_open:
                    raise AssertionError(f"closed+closed merge on qid={q!r}")
                if not a_open or not b_open:
                    raise AssertionError(f"closed+open merge on qid={q!r}")


def _orbit(*ms, is_open):
    ids = [quiver_id(m) for m in ms]
    return _RawOrbit(labeled_quivers=list(ms), quiver_ids=ids, qid_set=set(ids),
                     is_open=is_open, boundary_quivers=[ms[0]] if is_open else [])


def test_union_find_rejects_closed_merges():
    m_shared = to_matrix([[0, 1], [-1, 0]])
    m_a = to_matrix([[0, 2], [-2, 0]])
    m_b = to_matrix([[0, -1], [1, 0]])   # same Q.* as m_shared
    closed_a = _orbit(m_shared, m_a, is_open=False)
    closed_b = _orbit(m_shared, m_b, is_open=False)
    open_o = _orbit(m_shared, is_open=True)
    with pytest.raises(AssertionError, match="closed\\+closed"):
        _simulate_union([closed_a, closed_b])
    with pytest.raises(AssertionError, match="closed\\+open"):
        _simulate_union([closed_a, open_o])
    _simulate_union([open_o, open_o])    # open+open is the one legal merge


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.dirname(__file__), "-q"]))
