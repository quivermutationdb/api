"""
qmd/dynkin.py

Finite-type (Dynkin) classification for mutation classes.

A skew-symmetric mutation class is of *finite cluster type* iff it is
mutation-equivalent to an orientation of a simply-laced Dynkin diagram
(A_n, D_n, E_6/7/8; reference built to rank 8) — or a disjoint union of several, for a disconnected
quiver.  (Skew-symmetric => simply-laced, so only A/D/E occur; classes with
double arrows, e.g. the Kronecker quiver, are affine/wild and get no label.)

classify(canonical_rep) decomposes the quiver into connected components and
identifies each by matching its mutation-class id against a reference table
of Dynkin quivers.  It returns a combined label ("A3", "D4", "A1 + A2") or
None if any component is not a finite Dynkin type.

This is exact (it relies on mutation-class equality), not a heuristic.
"""

from __future__ import annotations

import json
import os
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


def _E(n: int) -> Matrix:
    """E_n quiver (n in 6..8): path 0..(n-2) with a leaf (n-1) attached to vertex 2 (arms 2, n-4, 1)."""
    assert 6 <= n <= 8
    rows = [[0] * n for _ in range(n)]
    for i in range(n - 2):
        rows[i][i + 1] = 1
        rows[i + 1][i] = -1
    rows[2][n - 1] = 1
    rows[n - 1][2] = -1
    return to_matrix(rows)


def _seeds_of_rank(k: int) -> dict[str, Matrix]:
    seeds = {f"A{k}": _A(k)}
    if k >= 4:
        seeds[f"D{k}"] = _D(k)
    if 6 <= k <= 8:
        seeds[f"E{k}"] = _E(k)
    return seeds


REFERENCE_CACHE = os.environ.get("QMD_DYNKIN_CACHE", os.path.join(
    os.path.dirname(__file__), "..", "dist", "dynkin-reference.json"))

_REFERENCE: dict[int, dict[str, str]] = {}


def reference_for(rank: int) -> dict[str, str]:
    """
    mc_id -> Dynkin name for every connected finite type of exactly this rank.
    E8's class is 25,080 labeled matrices, so the table is cached on disk
    (REFERENCE_CACHE) and computed once per process otherwise; call this in
    the parent before forking a worker pool.
    """
    if rank in _REFERENCE:
        return _REFERENCE[rank]
    cache: dict = {}
    if os.path.exists(REFERENCE_CACHE):
        try:
            with open(REFERENCE_CACHE, encoding="utf-8") as f:
                cache = json.load(f)
        except (OSError, ValueError):
            cache = {}
    key = str(rank)
    if key not in cache:
        cache[key] = {explore_mutation_class(seed).mc_id: name
                      for name, seed in _seeds_of_rank(rank).items()}
        try:
            os.makedirs(os.path.dirname(REFERENCE_CACHE), exist_ok=True)
            tmp = REFERENCE_CACHE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=1, sort_keys=True)
            os.replace(tmp, REFERENCE_CACHE)
        except OSError:
            pass
    _REFERENCE[rank] = cache[key]
    return cache[key]


def _build_reference(max_rank: int = 4) -> dict[str, str]:
    """Flat mc_id -> name table for all ranks up to max_rank (tests / tools)."""
    out: dict[str, str] = {}
    for k in range(1, max_rank + 1):
        out.update(reference_for(k))
    return out


def classify(canonical_rep: Matrix, mc_id: Optional[str] = None) -> Optional[str]:
    """
    Return the finite Dynkin type of the mutation class with this canonical
    representative (e.g. "A3", "D4", "A1 + A2"), or None if the class is not
    of finite type.

    Only meaningful for completely explored classes. If the quiver is
    connected and its class id is known, pass `mc_id` to avoid re-exploring
    the class (E8 alone is 25,080 matrices).
    """
    comps = _connected_components(canonical_rep)
    names: list[str] = []
    for comp in comps:
        ref = reference_for(len(comp))
        cid = mc_id if (mc_id is not None and len(comps) == 1) \
            else explore_mutation_class(comp).mc_id
        name = ref.get(cid)
        if name is None:
            return None
        names.append(name)
    if not names:
        return None
    return " + ".join(sorted(names))
