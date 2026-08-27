"""
qmd/canonicalize.py

Canonical forms and isomorphism testing for quiver exchange matrices.

THE ID KEY IS LEX-MIN — AND ONLY LEX-MIN
-----------------------------------------
    canonical_form(B) := the row-major lexicographically smallest matrix
                         among all n! vertex relabelings of B.

Every published Q.* / MC.* identifier is a hash of this matrix
(qmd/core.py: quiver_id, mutation_class_id). It is a mathematical definition,
not an implementation detail: it must give the same answer on every machine,
with or without optional libraries installed. `tests/golden/ids-n4.json`
pins the result for the whole n<=4 census.

Implementation: a branch-and-bound search over relabelings (`lexmin_form`).
It places vertices one at a time and prunes a branch as soon as a lower bound
on the completed key exceeds the best key found so far. Exact for all n;
`PermutationCanonicalizer` is the O(n!) brute force kept as the reference
oracle for tests.

Nauty (optional) — isomorphism only, never the key
--------------------------------------------------
If `pynauty` is importable and not disabled, `are_isomorphic` compares nauty
certificates of a gadget-graph encoding instead of computing two lex-min
forms. Nauty's own canonical labeling is a *different* canonical form; it is
deliberately never used to produce IDs.

    QMD_NAUTY=0   force the pure-Python path even if pynauty is installed
    QMD_NAUTY=1   require pynauty (ImportError otherwise)

Gadget encoding for nauty
--------------------------
Quiver vertices are nodes 0..n-1. For each pair i<j with b_ij = w != 0 we
insert |w| gadget nodes and a directed path source -> g_1 -> ... -> g_|w| ->
target (source/target chosen by the sign). Original vertices and gadget
nodes get different colours so nauty never mixes them. Weight k is a path of
length k+1, so any integer weight is encoded faithfully.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from itertools import permutations
from typing import Optional, Protocol

# Matrix type (defined here to avoid circular import; re-exported by core.py)
Matrix = tuple[tuple[int, ...], ...]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class Canonicalizer(Protocol):
    def canonical_form(self, matrix: Matrix) -> Matrix: ...
    def certificate(self, matrix: Matrix) -> bytes: ...
    @property
    def name(self) -> str: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_permutation(matrix: Matrix, perm: tuple[int, ...]) -> Matrix:
    """Apply vertex relabeling sigma: B'[i][j] = B[sigma(i)][sigma(j)]."""
    n = len(matrix)
    return tuple(
        tuple(matrix[perm[i]][perm[j]] for j in range(n))
        for i in range(n)
    )


def _lex_key(matrix: Matrix) -> tuple[int, ...]:
    """Row-major flattening: the total order that defines 'lex-min'."""
    return tuple(x for row in matrix for x in row)


def _digest(matrix: Matrix) -> bytes:
    serialized = json.dumps([list(row) for row in matrix], separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).digest()


# ---------------------------------------------------------------------------
# Lex-min by branch and bound  (THE canonical form)
# ---------------------------------------------------------------------------

def lexmin_form(matrix: Matrix) -> Matrix:
    """
    Row-major lex-min matrix over all vertex relabelings — exact for all n.

    Search: choose the image of position 0, then 1, ... . With positions
    0..k-1 assigned, rows 0..k-1 of the relabeled matrix are known in columns
    0..k-1; the unknown tail of each such row is bounded below by sorting that
    row's remaining entries ascending. The concatenation of those bounded rows
    is a valid lower bound (in row-major lex order) on every completion of the
    branch, so a branch whose bound already exceeds the best key found is cut.
    """
    n = len(matrix)
    if n <= 1:
        return matrix

    # Twin vertices: (u v) is an automorphism of the matrix iff b_uv = 0 and
    # u, v have identical rows elsewhere. Swapping twins never changes the
    # key, so at every level only one representative per twin class needs to
    # be tried. This is what keeps sparse / symmetric quivers (isolated
    # vertices, symmetric leaves — everything in a finite-type class) from
    # exploding into thousands of identical branches.
    twin = list(range(n))            # union-find root per vertex

    def find(x: int) -> int:
        while twin[x] != x:
            twin[x] = twin[twin[x]]
            x = twin[x]
        return x

    for u in range(n):
        for v in range(u + 1, n):
            if matrix[u][v] != 0:
                continue
            if all(matrix[u][w] == matrix[v][w] for w in range(n) if w != u and w != v):
                twin[find(v)] = find(u)

    best_key: list[Optional[tuple[int, ...]]] = [_lex_key(matrix)]
    best_perm: list[tuple[int, ...]] = [tuple(range(n))]

    def dfs(prefix: list[int], remaining: list[int]) -> None:
        k = len(prefix)
        if k == n:
            key = _lex_key(_apply_permutation(matrix, tuple(prefix)))
            if key < best_key[0]:
                best_key[0] = key
                best_perm[0] = tuple(prefix)
            return
        # Lower bound on rows 0..k-1 of any completion of this prefix.
        bound: list[int] = []
        for r in prefix:
            row = matrix[r]
            bound.extend(row[c] for c in prefix)
            bound.extend(sorted(row[v] for v in remaining))
        if tuple(bound) > best_key[0][: len(bound)]:
            return
        # One candidate per twin class among the remaining vertices, in
        # ascending order of their entry in the first row (good keys first).
        seen_roots: set[int] = set()
        candidates = []
        for v in sorted(remaining, key=lambda v: (matrix[prefix[0]][v], v)):
            r = find(v)
            if r in seen_roots:
                continue
            seen_roots.add(r)
            candidates.append(v)
        for v in candidates:
            rest = [u for u in remaining if u != v]
            prefix.append(v)
            dfs(prefix, rest)
            prefix.pop()

    seen_roots: set[int] = set()
    for start in range(n):
        r = find(start)
        if r in seen_roots:
            continue
        seen_roots.add(r)
        dfs([start], [v for v in range(n) if v != start])

    return _apply_permutation(matrix, best_perm[0])


class LexMinCanonicalizer:
    """Branch-and-bound lex-min: the production canonicalizer."""

    @property
    def name(self) -> str:
        return "lexmin"

    def canonical_form(self, matrix: Matrix) -> Matrix:
        return lexmin_form(matrix)

    def certificate(self, matrix: Matrix) -> bytes:
        return _digest(self.canonical_form(matrix))


class PermutationCanonicalizer:
    """
    Brute-force lex-min over all n! relabelings. Same answer as
    LexMinCanonicalizer by definition; kept as the reference oracle for tests.
    Practical for n <= 7.
    """

    @property
    def name(self) -> str:
        return "permutation"

    def canonical_form(self, matrix: Matrix) -> Matrix:
        n = len(matrix)
        return min(
            (_apply_permutation(matrix, perm) for perm in permutations(range(n))),
            key=_lex_key,
        )

    def certificate(self, matrix: Matrix) -> bytes:
        return _digest(self.canonical_form(matrix))


# ---------------------------------------------------------------------------
# Optional nauty isomorphism certificates
# ---------------------------------------------------------------------------

def _build_gadget_graph(matrix: Matrix):
    """
    Encode a quiver exchange matrix as a coloured directed graph for pynauty
    (see the module docstring). Returns (pynauty.Graph, n, total_nodes).
    """
    import pynauty

    n = len(matrix)
    gadget_count = sum(
        abs(matrix[i][j])
        for i in range(n) for j in range(i + 1, n)
        if matrix[i][j] != 0
    )
    total = n + gadget_count

    adjacency: dict[int, set[int]] = {v: set() for v in range(total)}
    next_gadget = n
    for i in range(n):
        for j in range(i + 1, n):
            w = matrix[i][j]
            if w == 0:
                continue
            src, dst, weight = (i, j, w) if w > 0 else (j, i, -w)
            chain = [src] + list(range(next_gadget, next_gadget + weight)) + [dst]
            next_gadget += weight
            for a, b in zip(chain, chain[1:]):
                adjacency[a].add(b)

    vertex_coloring = [set(range(n)), set(range(n, total))]
    g = pynauty.Graph(
        number_of_vertices=total,
        directed=True,
        adjacency_dict=adjacency,
        vertex_coloring=vertex_coloring,
    )
    return g, n, total


class NautyCertifier:
    """
    Isomorphism certificates via nauty. NOT a canonicalizer: nauty's canonical
    labeling is a different canonical form from lex-min, so it must never be
    used to build IDs. `canonical_form` here delegates to lex-min.
    """

    @property
    def name(self) -> str:
        return "nauty"

    def canonical_form(self, matrix: Matrix) -> Matrix:
        return lexmin_form(matrix)

    def certificate(self, matrix: Matrix) -> bytes:
        import pynauty
        g, _, _ = _build_gadget_graph(matrix)
        return bytes(pynauty.certificate(g))


# ---------------------------------------------------------------------------
# Backend selection (isomorphism only) and module-level API
# ---------------------------------------------------------------------------

_lexmin = LexMinCanonicalizer()
_fallback = PermutationCanonicalizer()


def _select_iso_backend() -> Canonicalizer:
    """
    QMD_NAUTY unset -> use nauty for isomorphism if importable (announced on
    stderr); '0' -> never; '1' -> required.
    """
    pref = os.environ.get("QMD_NAUTY", "").strip()
    if pref == "0":
        return _lexmin
    try:
        import pynauty  # noqa: F401
        c = NautyCertifier()
        c.certificate(((0, 1), (-1, 0)))       # smoke-test the encoding
    except Exception as exc:
        if pref == "1":
            raise ImportError("QMD_NAUTY=1 but pynauty is unusable") from exc
        return _lexmin
    print("qmd.canonicalize: using nauty for isomorphism certificates "
          "(IDs remain lex-min)", file=sys.stderr)
    return c


_iso_backend: Canonicalizer = _select_iso_backend()


def canonical_form(matrix: Matrix) -> Matrix:
    """
    The canonical (unlabeled) representative of a quiver matrix: the row-major
    lex-min over all relabelings. Two matrices are isomorphic as quivers iff
    their canonical forms are equal. Independent of installed libraries.
    """
    return lexmin_form(matrix)


def are_isomorphic(a: Matrix, b: Matrix) -> bool:
    """True iff a and b represent the same unlabeled quiver."""
    if len(a) != len(b):
        return False
    return _iso_backend.certificate(a) == _iso_backend.certificate(b)


def canonical_backend() -> str:
    """Name of the ID-key backend. Always 'lexmin'."""
    return _lexmin.name


def active_backend() -> str:
    """Name of the isomorphism-certificate backend: 'nauty' or 'lexmin'."""
    return _iso_backend.name


def verify_with_fallback(matrix: Matrix) -> bool:
    """
    Cross-check the branch-and-bound lex-min against brute force (n <= 7),
    and — when nauty is active — that its certificates agree with lex-min
    equality on this matrix. Returns True iff everything agrees.
    """
    cf = lexmin_form(matrix)
    if cf != _fallback.canonical_form(matrix):
        return False
    if _iso_backend.name == "nauty":
        return _iso_backend.certificate(matrix) == _iso_backend.certificate(cf)
    return True
