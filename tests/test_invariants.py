"""
tests/test_invariants.py — invariants.py, dynkin.py, local_acyclicity.py,
class_properties.py.

These are the modules whose answers are published as *facts*; a wrong value
here is worse than a missing one, so each test pins a known mathematical
result rather than the code's current output.
"""
import os
import random
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qmd import dynkin, invariants  # noqa: E402
from qmd import local_acyclicity as la  # noqa: E402
from qmd.canonicalize import _apply_permutation  # noqa: E402
from qmd.core import explore_mutation_class, mutate, quiver_id, to_matrix  # noqa: E402
from qmd.invariants import _det_int, class_is_mutation_acyclic, representation_type  # noqa: E402


def _opposite(m):
    return tuple(tuple(-x for x in row) for row in m)


# ---------------------------------------------------------------------------
# Integer determinant (used by the Tits-form classification)
# ---------------------------------------------------------------------------

def _det_fraction(M):
    n = len(M)
    A = [[Fraction(x) for x in row] for row in M]
    det = Fraction(1)
    for c in range(n):
        piv = next((r for r in range(c, n) if A[r][c] != 0), None)
        if piv is None:
            return 0
        if piv != c:
            A[c], A[piv] = A[piv], A[c]
            det = -det
        det *= A[c][c]
        for r in range(c + 1, n):
            f = A[r][c] / A[c][c]
            for k in range(c, n):
                A[r][k] -= f * A[c][k]
    return int(det)


def test_det_int_matches_exact_arithmetic():
    rng = random.Random(20260827)
    for _ in range(500):
        n = rng.randint(1, 6)
        M = [[rng.randint(-4, 4) for _ in range(n)] for _ in range(n)]
        assert _det_int(M) == _det_fraction(M)


# ---------------------------------------------------------------------------
# Representation type (Tits form)
# ---------------------------------------------------------------------------

def test_representation_type_known_cases(A3, D4, kronecker):
    assert representation_type(A3) == "finite"
    assert representation_type(D4) == "finite"
    assert representation_type(kronecker) == "tame"           # affine A1~
    affine_a2 = to_matrix([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])  # oriented 3-cycle
    assert representation_type(affine_a2) is None            # not acyclic
    acyclic_a2_tilde = to_matrix([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
    assert representation_type(acyclic_a2_tilde) == "tame"   # acyclic A2~
    wild = to_matrix([[0, 3], [-3, 0]])
    assert representation_type(wild) == "wild"


# ---------------------------------------------------------------------------
# Dynkin classification
# ---------------------------------------------------------------------------

def test_dynkin_classify_known_types(A3, D4, kronecker, markov):
    assert dynkin.classify(explore_mutation_class(A3).canonical_rep) == "A3"
    assert dynkin.classify(explore_mutation_class(D4).canonical_rep) == "D4"
    a4 = to_matrix([[0, 1, 0, 0], [-1, 0, 1, 0], [0, -1, 0, 1], [0, 0, -1, 0]])
    assert dynkin.classify(explore_mutation_class(a4).canonical_rep) == "A4"
    assert dynkin.classify(explore_mutation_class(kronecker).canonical_rep) is None
    assert dynkin.classify(explore_mutation_class(markov).canonical_rep) is None


def test_dynkin_classify_disjoint_union():
    a1_a2 = to_matrix([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
    assert dynkin.classify(explore_mutation_class(a1_a2).canonical_rep) == "A1 + A2"


def test_dynkin_is_orientation_independent(D4):
    # Any orientation of D4 is mutation-equivalent to any other.
    flipped = to_matrix([[0, -1, 1, 1], [1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0]])
    assert dynkin.classify(explore_mutation_class(flipped).canonical_rep) == "D4"


# ---------------------------------------------------------------------------
# Per-quiver invariants
# ---------------------------------------------------------------------------

def test_quiver_invariants_a3(A3):
    # "Bipartite" is the cluster-algebra sense: every vertex is a source or a
    # sink. The linear orientation 0 -> 1 -> 2 is not; the alternating one is.
    qi = invariants.quiver_invariants(A3)
    assert qi["is_bipartite"] is False
    assert qi["representation_type"] == "finite"
    # Automorphisms must preserve orientation: reversing 0 -> 1 -> 2 flips
    # every arrow, so the linear A3 has a trivial symmetry group ...
    assert qi["symmetry_group"]["order"] == 1
    alternating = to_matrix([[0, 1, 0], [-1, 0, -1], [0, 1, 0]])   # 0 -> 1 <- 2
    assert invariants.is_bipartite(alternating) is True
    # ... while swapping the two sources of 0 -> 1 <- 2 is one.
    assert invariants.symmetry_group(alternating)["order"] == 2


def test_symmetry_group_markov(markov):
    # Cyclic relabelings fix the oriented 3-cycle: order 3.
    assert invariants.symmetry_group(markov)["order"] == 3


# ---------------------------------------------------------------------------
# Banff / Louise / P' — three-state searches
# ---------------------------------------------------------------------------

BUDGET = dict(max_depth=16, timeout=10, cap=8)


@pytest.mark.parametrize("kind", ["banff", "louise", "p_prime"])
def test_local_acyclicity_acyclic_is_true(A3, D4, kind):
    fn = getattr(la, f"{kind}_status")
    for m in (A3, D4):
        state, witness = fn(m, **BUDGET)
        assert state == "true" and witness is not None


@pytest.mark.parametrize("kind", ["banff", "louise", "p_prime"])
def test_local_acyclicity_invariant_under_opposite_quiver(kind):
    fn = getattr(la, f"{kind}_status")
    m = to_matrix([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])     # oriented 3-cycle
    assert fn(m, **BUDGET)[0] == fn(_opposite(m), **BUDGET)[0]


@pytest.mark.parametrize("kind", ["banff", "louise", "p_prime"])
def test_local_acyclicity_invariant_under_relabeling(D4, kind):
    fn = getattr(la, f"{kind}_status")
    m = mutate(D4, 0)
    assert fn(m, **BUDGET)[0] == fn(_apply_permutation(m, (3, 1, 0, 2)), **BUDGET)[0]


def test_local_acyclicity_never_false_when_truncated(markov):
    # With a tiny budget the search cannot finish; the answer must be unknown,
    # never a false "false".
    for fn in (la.banff_status, la.louise_status, la.p_prime_status):
        state, _ = fn(markov, max_depth=0, timeout=10, cap=2)
        assert state in ("unknown", "true")


# ---------------------------------------------------------------------------
# Mutation-acyclicity and the induced-subquiver fallback
# ---------------------------------------------------------------------------

def test_class_is_mutation_acyclic(A3, markov):
    r_a3 = explore_mutation_class(A3)
    assert class_is_mutation_acyclic(r_a3.labeled_quivers, r_a3.is_open) is True
    r_mk = explore_mutation_class(markov)
    assert not r_mk.is_open                       # Markov is its own closed class
    assert class_is_mutation_acyclic(r_mk.labeled_quivers, r_mk.is_open) is False


def test_subquiver_fallback_markov_heredity(r4, markov):
    from qmd.class_properties import resolve_mutation_acyclic
    infos = [(mc_id, len(mc.canonical_rep),
              class_is_mutation_acyclic(mc.labeled_quivers, mc.is_open),
              mc.labeled_quivers, mc.quiver_ids) for mc_id, mc in r4.classes.items()]
    resolved = resolve_mutation_acyclic(infos)
    base = {mc_id: b for mc_id, _n, b, _m, _q in infos}
    markov_mc = r4.membership[quiver_id(markov)]
    assert resolved[markov_mc] is False
    # Only ever upgrades unknown -> False; never overwrites a proved value.
    assert all(resolved[k] == base[k] or (base[k] is None and resolved[k] is False)
               for k in resolved)
    assert any(base[k] is None and resolved[k] is False for k in resolved)
