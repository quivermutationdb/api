"""
qmd/local_acyclicity.py

Banff / Louise / P' conditions (sufficient conditions for local acyclicity),
ported from the reference implementation onto qmd's core types.

These properties are only semidecidable, so each public entry point returns a
three-state result:

    "true"     a witness (mutation sequence + source-deletion certificate) was
               found  -> the quiver IS Banff / Louise / P'
    "false"    the search was provably exhaustive (the whole finite mutation
               class was covered with no depth / timeout / arrow-cap truncation
               anywhere) and no witness exists -> it is NOT
    "unknown"  the search was truncated before a witness or a proof -> open

A `truncated` flag is threaded through the entire computation (BFS + recursion);
a "false" is only reported when nothing was ever truncated, so it is a genuine
proof.  (This is the key correctness adaptation over a plain depth-limited BFS.)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

from qmd.core import Matrix, to_matrix, mutate, max_edge, is_acyclic


class _Ctx:
    """Shared search budget; records whether anything was truncated."""
    def __init__(self, max_depth: int, timeout: float, cap: int):
        self.max_depth = max_depth
        self.timeout = timeout
        self.cap = cap
        self.start = time.time()
        self.truncated = False

    def tick(self) -> None:
        if self.timeout is not None and time.time() - self.start > self.timeout:
            self.truncated = True
            raise TimeoutError


def _delete_vertex(q: Matrix, k: int) -> Matrix:
    idx = [i for i in range(len(q)) if i != k]
    return tuple(tuple(q[a][b] for b in idx) for a in idx)


def _mut(q: Matrix, k: int, ctx: _Ctx) -> Optional[Matrix]:
    r = mutate(q, k)
    if max_edge(r) > ctx.cap:
        ctx.truncated = True
        return None
    return r


def _is_mutation_acyclic(q: Matrix, ctx: _Ctx) -> Optional[list[int]]:
    """Return a mutation path to an acyclic quiver, or None (within budget)."""
    if is_acyclic(q):
        return []
    queue = deque([(q, 0, [])])
    seen = {q}
    while queue:
        ctx.tick()
        cur, depth, path = queue.popleft()
        if depth >= ctx.max_depth:
            ctx.truncated = True
            continue
        for k in range(len(cur)):
            mu = _mut(cur, k, ctx)
            if mu is None or mu in seen:
                continue
            seen.add(mu)
            new_path = path + [k]
            if is_acyclic(mu):
                return new_path
            queue.append((mu, depth + 1, new_path))
    return None


# --- recursive source-deletion conditions ---------------------------------

def _sources(q: Matrix) -> list[int]:
    n = len(q)
    return [k for k in range(n) if all(q[k][j] >= 0 for j in range(n))]


def _check_condition(q: Matrix, ctx: _Ctx, kind: str):
    """Banff/Louise/P' source-deletion condition on a single quiver."""
    if is_acyclic(q):
        return (True, "acyclic")
    n = len(q)
    for k in _sources(q):
        ok_k, w_k = _is_cond_class(_delete_vertex(q, k), ctx, kind)
        if not ok_k:
            continue
        if kind == "p_prime":
            return (True, {"source": k, "witness_k": w_k})
        # banff / louise need a neighbor j
        for j in range(n):
            adjacent = q[k][j] > 0 if kind == "banff" else (j != k and q[k][j] != 0)
            if not adjacent:
                continue
            ok_j, w_j = _is_cond_class(_delete_vertex(q, j), ctx, kind)
            if not ok_j:
                continue
            if kind == "banff":
                return (True, {"source": k, "neighbor": j,
                               "witness_k": w_k, "witness_j": w_j})
            # louise also needs Q\{k,j}
            q_kj = _delete_vertex(_delete_vertex(q, k), j - 1 if j > k else j)
            ok_kj, w_kj = _is_cond_class(q_kj, ctx, kind)
            if ok_kj:
                return (True, {"source": k, "neighbor": j, "witness_k": w_k,
                               "witness_j": w_j, "witness_kj": w_kj})
    return (False, None)


def _is_cond_class(q: Matrix, ctx: _Ctx, kind: str):
    """Is q mutation-equivalent to a quiver satisfying the `kind` condition?"""
    path = _is_mutation_acyclic(q, ctx)
    if path is not None:
        return (True, {"mutations": path, "condition": "acyclic"})

    ok, cw = _check_condition(q, ctx, kind)
    if ok:
        return (True, {"mutations": [], "condition": cw})

    queue = deque([(q, 0, [])])
    seen = {q}
    while queue:
        ctx.tick()
        cur, depth, path = queue.popleft()
        if depth >= ctx.max_depth:
            ctx.truncated = True
            continue
        for m in range(len(cur)):
            mu = _mut(cur, m, ctx)
            if mu is None or mu in seen:
                continue
            seen.add(mu)
            new_path = path + [m]
            ok, cw = _check_condition(mu, ctx, kind)
            if ok:
                return (True, {"mutations": new_path, "condition": cw})
            queue.append((mu, depth + 1, new_path))
    return (False, None)


def _status(q, kind: str, max_depth: int, timeout: float, cap: int):
    """Three-state wrapper: returns (state, witness)."""
    ctx = _Ctx(max_depth, timeout, cap)
    try:
        ok, witness = _is_cond_class(to_matrix([list(r) for r in q]), ctx, kind)
    except TimeoutError:
        return ("unknown", None)
    if ok:
        return ("true", witness)
    return ("false", None) if not ctx.truncated else ("unknown", None)


# --- public API ------------------------------------------------------------

def banff_status(q, max_depth=64, timeout=60, cap=32):
    return _status(q, "banff", max_depth, timeout, cap)


def louise_status(q, max_depth=64, timeout=60, cap=32):
    return _status(q, "louise", max_depth, timeout, cap)


def p_prime_status(q, max_depth=64, timeout=60, cap=32):
    return _status(q, "p_prime", max_depth, timeout, cap)
