"""
qmd/canonicalize.py

Graph canonicalization for quiver exchange matrices.

Architecture
------------
Canonicalization is exposed through a single function:

    canonical_form(matrix: Matrix) -> Matrix

Two matrices represent the same unlabeled quiver iff their canonical
forms are equal.  The implementation is selected at import time:

  1. NautyCanonicalizer  — uses pynauty (nauty/Traces), O(n log n) average.
                           Active when `pynauty` is importable.

  2. PermutationCanonicalizer — enumerates all n! vertex permutations,
                           exact but O(n! * n^2).  Used as a fallback and
                           for testing/verification.

Gadget encoding for nauty
--------------------------
Quiver exchange matrices are skew-symmetric integer matrices with entries
in {0, ±1, ±2, ...}.  Nauty operates on simple undirected or directed
graphs, so we encode the quiver as a directed graph using *gadget nodes*:

  - Original quiver vertices become nodes 0..n-1.
  - For each ordered pair (i, j) with b_ij > 0:
      Insert b_ij gadget nodes g_1, ..., g_{b_ij} numbered consecutively.
      Add directed edges:  i -> g_1 -> g_2 -> ... -> g_{b_ij} -> j.

Properties of this encoding:
  - Weight k is encoded as a directed path of length k+1 through k gadget
    nodes.  Weights 1 and 2 produce paths of length 2 and 3 respectively.
  - Direction encodes sign: b_ij > 0 means arrows from i toward j.
    Since B is skew-symmetric, b_ji = -b_ij is implicit; we never insert
    gadgets for the (j, i) direction separately.
  - Gadget nodes have in-degree 1 and out-degree 1 (internal) or
    in-degree 1 and out-degree 0 (terminal, which we instead wire to j).
    This degree signature distinguishes them from real quiver vertices.
  - Works for any positive integer weight without modification.

The canonical certificate produced by nauty is a hash of the canonical
adjacency structure.  We reconstruct the canonical matrix by applying the
permutation nauty returns for the *original* vertices only (gadget nodes
are projected out after canonicalization).
"""

from __future__ import annotations

import hashlib
import json
from itertools import permutations
from typing import Protocol

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
    return tuple(x for row in matrix for x in row)


# ---------------------------------------------------------------------------
# Implementation 1: PermutationCanonicalizer (fallback)
# ---------------------------------------------------------------------------

class PermutationCanonicalizer:
    """
    Canonicalizes by enumerating all n! vertex permutations and returning
    the lexicographically smallest result.

    Exact for all n.  Practical for n <= 6 (720 permutations).
    For n = 7: 5040 permutations.  For n = 8: 40320.
    Recommended only as a fallback or for testing.
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
        cf = self.canonical_form(matrix)
        serialized = json.dumps([list(row) for row in cf], separators=(',', ':'))
        return hashlib.sha256(serialized.encode()).digest()


# ---------------------------------------------------------------------------
# Implementation 2: NautyCanonicalizer
# ---------------------------------------------------------------------------

def _build_gadget_graph(matrix: Matrix):
    """
    Encode a quiver exchange matrix as a directed graph for pynauty,
    using the gadget-node construction described in the module docstring.

    Returns (pynauty.Graph, original_node_count, total_node_count).

    The adjacency sets passed to pynauty use its directed graph convention:
    adjacency_dict[u] = set of nodes that u has edges *to*.
    """
    import pynauty

    n = len(matrix)
    # Count total gadget nodes needed
    gadget_count = sum(
        matrix[i][j]
        for i in range(n) for j in range(i + 1, n)
        if matrix[i][j] > 0
    )
    total = n + gadget_count

    adjacency: dict[int, set[int]] = {v: set() for v in range(total)}
    next_gadget = n

    for i in range(n):
        for j in range(i + 1, n):
            w = matrix[i][j]
            if w == 0:
                continue
            # Positive w: directed path from i to j through w gadget nodes
            # Negative w: directed path from j to i through |w| gadget nodes
            src, dst, weight = (i, j, w) if w > 0 else (j, i, -w)
            chain = [src] + list(range(next_gadget, next_gadget + weight)) + [dst]
            next_gadget += weight
            for a, b in zip(chain, chain[1:]):
                adjacency[a].add(b)

    # Vertex colouring: original vertices all get colour 0,
    # gadget nodes get colour 1.  This prevents nauty from permuting
    # original vertices and gadget nodes interchangeably.
    vertex_coloring = [set(range(n)), set(range(n, total))]

    g = pynauty.Graph(
        number_of_vertices=total,
        directed=True,
        adjacency_dict=adjacency,
        vertex_coloring=vertex_coloring,
    )
    return g, n, total


def _nauty_canonical_perm(matrix: Matrix) -> tuple[int, ...]:
    """
    Run nauty on the gadget graph and return the canonical permutation
    for the original quiver vertices (length n).

    Nauty returns a permutation of ALL nodes (original + gadgets).
    We extract just the sub-permutation on the original n nodes,
    normalised to a permutation of {0,...,n-1}.
    """
    import pynauty

    n = len(matrix)
    if n == 1:
        return (0,)

    g, orig_n, _ = _build_gadget_graph(matrix)
    cert = pynauty.certificate(g)

    # pynauty.canon_label returns the canonical labeling permutation
    canon_lab = pynauty.canon_label(g)

    # canon_lab[v] = canonical label of node v.
    # We want the induced permutation on original vertices:
    # collect positions of original vertices in canonical order.
    orig_in_canon_order = sorted(range(orig_n), key=lambda v: canon_lab[v])
    # orig_in_canon_order[i] = which original vertex maps to position i
    perm = tuple(orig_in_canon_order)
    return perm


class NautyCanonicalizer:
    """
    Canonicalizes quiver exchange matrices via nauty/Traces (pynauty).

    Uses the gadget-node directed graph encoding so that edge weights
    and directions are faithfully represented.  Works for any bounded
    integer weight without modification.

    Requires:  pip install pynauty
    """

    @property
    def name(self) -> str:
        return "nauty"

    def canonical_form(self, matrix: Matrix) -> Matrix:
        perm = _nauty_canonical_perm(matrix)
        return _apply_permutation(matrix, perm)

    def certificate(self, matrix: Matrix) -> bytes:
        import pynauty
        g, _, _ = _build_gadget_graph(matrix)
        return pynauty.certificate(g)


# ---------------------------------------------------------------------------
# Auto-select and module-level singletons
# ---------------------------------------------------------------------------

def _make_canonicalizer() -> Canonicalizer:
    try:
        import pynauty  # noqa: F401
        c = NautyCanonicalizer()
        # Smoke-test: canonicalize a known 2x2 matrix
        m = ((0, 1), (-1, 0))
        c.canonical_form(m)
        return c
    except Exception:
        return PermutationCanonicalizer()


_canonicalizer: Canonicalizer = _make_canonicalizer()
_fallback: Canonicalizer = PermutationCanonicalizer()


def canonical_form(matrix: Matrix) -> Matrix:
    """
    Return the canonical (unlabeled) representative of a quiver matrix.

    Two matrices are isomorphic as quivers iff their canonical forms
    are equal.

    Uses nauty if available, otherwise falls back to full permutation
    enumeration.
    """
    return _canonicalizer.canonical_form(matrix)


def are_isomorphic(a: Matrix, b: Matrix) -> bool:
    """True iff matrices a and b represent the same unlabeled quiver."""
    if len(a) != len(b):
        return False
    return canonical_form(a) == canonical_form(b)


def active_backend() -> str:
    """Return the name of the active canonicalization backend."""
    return _canonicalizer.name


def verify_with_fallback(matrix: Matrix) -> bool:
    """
    Cross-check nauty result against the permutation fallback.
    Useful for testing and validating the gadget encoding.
    Returns True iff both backends agree.
    Only meaningful when nauty is active.
    """
    cf_primary = _canonicalizer.canonical_form(matrix)
    cf_fallback = _fallback.canonical_form(matrix)
    return cf_primary == cf_fallback
