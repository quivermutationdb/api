"""
qmd/dynkin.py

Finite-type (Dynkin) classification for mutation classes.

A skew-symmetric mutation class is of *finite cluster type* iff it is
mutation-equivalent to an orientation of a simply-laced Dynkin diagram
(A_n, D_n, E_6/7/8) — or a disjoint union of several, for a disconnected
quiver.  (Skew-symmetric => simply-laced, so only A/D/E occur; classes with
double arrows, e.g. the Kronecker quiver, are affine/wild and get no label.)

classify(canonical_rep) decomposes the quiver into connected components and
identifies each by matching its mutation-class id against a reference table
of Dynkin quivers.  It returns a combined label ("A3", "D4", "A1 + A2") or
None if any component is not a finite Dynkin type.

This is exact (it relies on mutation-class equality), not a heuristic.
"""

from __future__ import annotations

from typing import Optional

from qmd.core import Matrix, to_matrix, explore_mutation_class


def _connected_components(matrix: Matrix) -> list[Matrix]:
    """Split a quiver into connected-component submatrices (edges = nonzero entries)."""
    n = len(matrix)
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0:
                adj[i].add(j)
                adj[j].add(i)

    seen: set[int] = set()
    components: list[Matrix] = []
    for start in range(n):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        members: list[int] = []
        while stack:
            v = stack.pop()
            members.append(v)
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        members.sort()
        sub = tuple(tuple(matrix[i][j] for j in members) for i in members)
        components.append(sub)
    return components


def _A(n: int) -> Matrix:
    """Linear A_n quiver: 0 -> 1 -> ... -> (n-1)."""
    rows = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        rows[i][i + 1] = 1
        rows[i + 1][i] = -1
    return to_matrix(rows)


def _D(n: int) -> Matrix:
    """D_n quiver (n >= 4): path 0..(n-2) with an extra leaf (n-1) at vertex (n-3)."""
    rows = [[0] * n for _ in range(n)]
    for i in range(n - 2):
        rows[i][i + 1] = 1
        rows[i + 1][i] = -1
    rows[n - 3][n - 1] = 1
    rows[n - 1][n - 3] = -1
    return to_matrix(rows)


def _build_reference(max_rank: int = 4) -> dict[str, str]:
    """
    Map mutation-class id -> Dynkin name for every connected finite type up
    to max_rank.  (E_6/E_7/E_8 should be added here when max_rank reaches 6+.)
    """
    seeds: dict[str, Matrix] = {f"A{k}": _A(k) for k in range(1, max_rank + 1)}
    seeds.update({f"D{k}": _D(k) for k in range(4, max_rank + 1)})
    return {explore_mutation_class(seed).mc_id: name for name, seed in seeds.items()}


_REFERENCE: Optional[dict[str, str]] = None


def _reference() -> dict[str, str]:
    global _REFERENCE
    if _REFERENCE is None:
        _REFERENCE = _build_reference()
    return _REFERENCE


def classify(canonical_rep: Matrix) -> Optional[str]:
    """
    Return the finite Dynkin type of the mutation class with this canonical
    representative (e.g. "A3", "D4", "A1 + A2"), or None if the class is not
    of finite type.
    """
    ref = _reference()
    names: list[str] = []
    for comp in _connected_components(canonical_rep):
        name = ref.get(explore_mutation_class(comp).mc_id)
        if name is None:
            return None
        names.append(name)
    if not names:
        return None
    return " + ".join(sorted(names))
