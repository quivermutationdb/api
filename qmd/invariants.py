"""
qmd/invariants.py

Exactly-computable quiver invariants.

Everything here is deterministic and exact (no floating point, no search
truncation): the results are correct, not heuristic.  Properties that are only
semidecidable live in qmd/local_acyclicity.py instead.
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Optional

from qmd.core import Matrix, is_acyclic


# ---------------------------------------------------------------------------
# Simple combinatorial invariants
# ---------------------------------------------------------------------------

def is_bipartite(m: Matrix) -> bool:
    """Every vertex is a source or a sink (alternating orientation)."""
    n = len(m)
    for i in range(n):
        has_out = any(m[i][j] > 0 for j in range(n))
        has_in = any(m[i][j] < 0 for j in range(n))
        if has_out and has_in:
            return False
    return True


def is_abundant(m: Matrix) -> bool:
    """Connected on >=2 vertices with |b_ij| >= 2 for every pair i != j."""
    n = len(m)
    if n < 2:
        return False
    return all(abs(m[i][j]) >= 2 for i in range(n) for j in range(i + 1, n))


def is_planar(m: Matrix) -> Optional[bool]:
    """
    Underlying multigraph is planar.  Every graph on <=4 vertices is planar
    (K5 / K3,3 need >=5), so this is exact for the current data; returns None
    (unknown) for larger n until a full planarity test is wired in.
    """
    if len(m) <= 4:
        return True
    return None


# ---------------------------------------------------------------------------
# Representation type of the path algebra kQ (acyclic Q only) via Tits form
# ---------------------------------------------------------------------------

def _det_int(mat: list[list[int]]) -> int:
    """Exact integer determinant (fraction-free Bareiss)."""
    n = len(mat)
    if n == 0:
        return 1
    M = [row[:] for row in mat]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            swap = next((i for i in range(k + 1, n) if M[i][k] != 0), None)
            if swap is None:
                return 0
            M[k], M[swap] = M[swap], M[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]


def _principal_submatrix(M: list[list[int]], idx: tuple[int, ...]) -> list[list[int]]:
    return [[M[a][b] for b in idx] for a in idx]


def _is_positive_definite(M: list[list[int]]) -> bool:
    """Sylvester's criterion: all leading principal minors > 0."""
    n = len(M)
    return all(_det_int([row[:k] for row in M[:k]]) > 0 for k in range(1, n + 1))


def _is_positive_semidefinite(M: list[list[int]]) -> bool:
    """A symmetric matrix is PSD iff every principal minor is >= 0."""
    n = len(M)
    for size in range(1, n + 1):
        for idx in combinations(range(n), size):
            if _det_int(_principal_submatrix(M, idx)) < 0:
                return False
    return True


def representation_type(m: Matrix) -> Optional[str]:
    """
    Representation type of the path algebra kQ: 'finite', 'tame', or 'wild'.

    Returns None when Q is not acyclic (kQ is then infinite-dimensional and the
    finite/tame/wild trichotomy does not apply).

    Uses the symmetric Tits form M = 2I - |B|:
      positive definite      -> finite (Dynkin)
      positive semidefinite  -> tame   (Euclidean / extended Dynkin)
      indefinite             -> wild
    Exact (integer determinants).
    """
    if not is_acyclic(m):
        return None
    n = len(m)
    M = [[2 if i == j else -abs(m[i][j]) for j in range(n)] for i in range(n)]
    if _is_positive_definite(M):
        return "finite"
    if _is_positive_semidefinite(M):
        return "tame"
    return "wild"


def is_representation_finite(m: Matrix) -> Optional[bool]:
    rt = representation_type(m)
    return None if rt is None else (rt == "finite")


def is_representation_tame(m: Matrix) -> Optional[bool]:
    rt = representation_type(m)
    return None if rt is None else (rt == "tame")


# ---------------------------------------------------------------------------
# Automorphism (symmetry) group of the labeled quiver
# ---------------------------------------------------------------------------

def _compose(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(p[q[i]] for i in range(len(p)))


def _element_order(p: tuple[int, ...]) -> int:
    ident = tuple(range(len(p)))
    x, k = p, 1
    while x != ident:
        x = _compose(p, x)
        k += 1
    return k


def _group_name(elements: list[tuple[int, ...]]) -> str:
    order = len(elements)
    if order == 1:
        return "1"
    abelian = all(_compose(a, b) == _compose(b, a) for a in elements for b in elements)
    if any(_element_order(p) == order for p in elements):
        return f"Z/{order}"
    if order == 4:
        return "Z/2 x Z/2"
    if order == 6:
        return "S_3" if not abelian else "Z/6"
    if order == 8 and not abelian:
        return "D_4"
    if order == 12 and not abelian:
        return "A_4"
    if order == 24 and not abelian:
        return "S_4"
    return f"order {order}"


def _generating_set(elements: list[tuple[int, ...]], n: int) -> list[list[int]]:
    """A small generating set found greedily by subgroup closure."""
    ident = tuple(range(n))
    gens: list[tuple[int, ...]] = []
    closure = {ident}
    for g in elements:
        if g in closure:
            continue
        gens.append(g)
        # recompute closure of gens
        closure = {ident}
        frontier = [ident]
        while frontier:
            x = frontier.pop()
            for h in gens:
                y = _compose(x, h)
                if y not in closure:
                    closure.add(y)
                    frontier.append(y)
        if len(closure) == len(elements):
            break
    return [list(g) for g in gens]


def symmetry_group(m: Matrix) -> dict:
    """
    Automorphism group of the labeled quiver: permutations sigma with
    b_{sigma(i)sigma(j)} = b_{ij} for all i, j.

    Returns {order, name, generators}.  Brute force over all n! permutations
    (exact; fine for small n — switch to nauty when scaling).
    """
    n = len(m)
    elements = [
        perm for perm in permutations(range(n))
        if all(m[perm[i]][perm[j]] == m[i][j] for i in range(n) for j in range(n))
    ]
    return {
        "order": len(elements),
        "name": _group_name(elements),
        "generators": _generating_set(elements, n),
    }


# ---------------------------------------------------------------------------
# Bundles used by the writer
# ---------------------------------------------------------------------------

def quiver_invariants(m: Matrix) -> dict:
    """All exact per-quiver invariants for one (canonical) quiver."""
    return {
        "is_bipartite": is_bipartite(m),
        "is_abundant": is_abundant(m),
        "is_planar": is_planar(m),
        "representation_type": representation_type(m),
        "symmetry_group": symmetry_group(m),
    }


def class_is_mutation_acyclic(members, is_open: bool) -> Optional[bool]:
    """
    True if any explored member is acyclic.  For a finite (closed) class the
    member list is complete, so a 'no' is exact (False); for an open class a
    'no' is only 'unknown' (None) since exploration is partial.
    """
    if any(is_acyclic(m) for m in members):
        return True
    return False if not is_open else None
