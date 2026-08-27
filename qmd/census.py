"""
qmd/census.py

Bounded-height census of quivers: exact counts and orderly generation.

A "cell" (n, h) is the set of unlabeled quivers on n vertices whose exchange
matrix has every |b_ij| <= h. Two things live here:

  count_quivers(n, h)   exact number of isomorphism classes in the cell
                        (Burnside over S_n; no enumeration) — use it to decide
                        what is storable BEFORE generating anything.

  generate_cell(n, h)   one representative per isomorphism class, produced
                        by canonical augmentation (orderly generation): every
                        class is emitted exactly once, without ever listing
                        the (2h+1)^{n(n-1)/2} labeled matrices, and the work
                        parallelises over the parent (n-1)-quivers.

  sample_cell(n, h, k)  k distinct uniformly-drawn labeled matrices,
                        canonicalised and deduplicated — for cells too large
                        to enumerate (ML datasets), with the same downstream.

Generation canonical form vs. ID canonical form
-----------------------------------------------
The published IDs hash the *row-major* lex-min matrix (qmd/canonicalize.py).
Row-major order is not hereditary (a leading principal block of a row-major
lex-min matrix need not itself be lex-min), so it cannot drive canonical
augmentation. Generation therefore uses a second, *hereditary* canonical form
based on the "augmentation key"

    key(B) = (B[1][0]; B[2][0], B[2][1]; B[3][0], B[3][1], B[3][2]; ...)

i.e. the strictly-lower-triangular entries read row by row. The key of the
leading k-block is a prefix of key(B), so if B is key-minimal over all
relabelings then so is every leading principal block. Consequently:

    every (n, h)-quiver has exactly one key-minimal labeling, and its leading
    (n-1)-block is the key-minimal labeling of an (n-1, h)-quiver.

So extending every key-minimal (n-1)-matrix by every possible new row and
keeping exactly the key-minimal results enumerates the cell once. The emitted
representative is then converted to the ID form with canonical_form().
"""

from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction
from typing import Iterable, Iterator, Optional

from qmd.canonicalize import Matrix, canonical_form

# ---------------------------------------------------------------------------
# Exact counts (Burnside / Pólya)
# ---------------------------------------------------------------------------

def _partitions(n: int, largest: Optional[int] = None) -> Iterator[tuple[int, ...]]:
    """Integer partitions of n as non-increasing tuples (cycle types of S_n)."""
    if largest is None:
        largest = n
    if n == 0:
        yield ()
        return
    for first in range(min(n, largest), 0, -1):
        for rest in _partitions(n - first, first):
            yield (first,) + rest


def _class_size(cycle_type: tuple[int, ...]) -> int:
    """Number of permutations in S_n with this cycle type."""
    n = sum(cycle_type)
    denom = 1
    for length, mult in _multiplicities(cycle_type).items():
        denom *= (length ** mult) * math.factorial(mult)
    return math.factorial(n) // denom


def _multiplicities(cycle_type: tuple[int, ...]) -> dict[int, int]:
    out: dict[int, int] = {}
    for L in cycle_type:
        out[L] = out.get(L, 0) + 1
    return out


def _free_pair_orbits(cycle_type: tuple[int, ...]) -> int:
    """
    Number of orbits of the permutation on unordered vertex pairs on which a
    fixed skew-symmetric matrix may take an arbitrary value.

    Pairs inside one cycle of length L split by "distance" d = 1..floor(L/2):
    the orbit at d = L/2 (L even) maps (i, j) to (j, i), forcing b_ij = 0, so
    only floor((L-1)/2) orbits are free. Pairs across two cycles of lengths
    L1, L2 form gcd(L1, L2) orbits, none of which is ever flipped.
    """
    free = 0
    for L in cycle_type:
        free += (L - 1) // 2
    for a, b in itertools.combinations(range(len(cycle_type)), 2):
        free += math.gcd(cycle_type[a], cycle_type[b])
    return free


def count_quivers(n: int, h: int) -> int:
    """Exact number of unlabeled quivers on n vertices with all |b_ij| <= h."""
    if n <= 1:
        return 1
    values = 2 * h + 1
    total = Fraction(0)
    for ct in _partitions(n):
        total += _class_size(ct) * Fraction(values) ** _free_pair_orbits(ct)
    total /= math.factorial(n)
    assert total.denominator == 1
    return int(total)


def count_connected_quivers(n: int, h: int) -> int:
    """
    Exact number of *connected* unlabeled quivers on n vertices with all
    |b_ij| <= h. Every quiver is a multiset of connected components, so the
    all-quiver counts a_k and connected counts c_k satisfy the Euler
    transform  sum a_k x^k = prod_k (1 - x^k)^(-c_k); invert it.
    """
    a = [count_quivers(k, h) for k in range(0, n + 1)]      # a_0 = 1
    # Standard inversion: n*a_n = sum_{k=1..n} b_k a_{n-k}, b_k = sum_{d|k} d*c_d.
    c = [0] * (n + 1)
    b = [0] * (n + 1)
    for m in range(1, n + 1):
        total = m * a[m] - sum(b[k] * a[m - k] for k in range(1, m))
        b[m] = total
        c[m] = (b[m] - sum(d * c[d] for d in range(1, m) if m % d == 0)) // m
    return c[n]


# ---------------------------------------------------------------------------
# Hereditary canonical form (augmentation key) with branch and bound
# ---------------------------------------------------------------------------

def _key_rows(matrix: Matrix, perm: list[int]) -> list[tuple[int, ...]]:
    """Augmentation key of the relabeled matrix, one tuple per vertex level."""
    return [tuple(matrix[perm[i]][perm[j]] for j in range(i)) for i in range(len(perm))]


def is_key_minimal(matrix: Matrix) -> bool:
    """
    True iff `matrix` is the key-minimal labeling of its quiver — i.e. no
    relabeling has a strictly smaller augmentation key.

    Depth-first over partial permutations; at level i the key rows 1..i of the
    candidate are known and compared with the matrix's own rows: a larger
    prefix is pruned, a smaller one is a counterexample, equal continues.
    """
    n = len(matrix)
    own = [tuple(matrix[i][j] for j in range(i)) for i in range(n)]

    def dfs(perm: list[int], remaining: list[int]) -> bool:
        i = len(perm)
        if i == n:
            return True                         # equal key all the way: not smaller
        for v in remaining:
            row = tuple(matrix[v][perm[j]] for j in range(i))
            if row > own[i]:
                continue                        # this branch is larger
            perm.append(v)
            rest = [u for u in remaining if u != v]
            if row < own[i]:
                return False                    # strictly smaller key found
            if not dfs(perm, rest):
                return False
            perm.pop()
        return True

    return dfs([], list(range(n)))


def key_minimal_form(matrix: Matrix) -> Matrix:
    """The key-minimal labeling of a quiver (used by sample_cell and tests)."""
    n = len(matrix)
    best: list[Optional[list[int]]] = [None]
    best_key: list[list[tuple[int, ...]]] = [[]]

    def dfs(perm: list[int], remaining: list[int], rows: list[tuple[int, ...]]) -> None:
        i = len(perm)
        if best[0] is not None and rows > best_key[0][:i]:
            return
        if i == n:
            if best[0] is None or rows < best_key[0]:
                best[0], best_key[0] = list(perm), list(rows)
            return
        for v in sorted(remaining, key=lambda v: tuple(matrix[v][perm[j]] for j in range(i))):
            row = tuple(matrix[v][perm[j]] for j in range(i))
            perm.append(v); rows.append(row)
            dfs(perm, [u for u in remaining if u != v], rows)
            perm.pop(); rows.pop()

    dfs([], list(range(n)), [])
    p = best[0] or list(range(n))
    return tuple(tuple(matrix[p[i]][p[j]] for j in range(n)) for i in range(n))


# ---------------------------------------------------------------------------
# Orderly generation (canonical augmentation)
# ---------------------------------------------------------------------------

def _extend(parent: Matrix, new_row: tuple[int, ...]) -> Matrix:
    n = len(parent) + 1
    rows = [list(r) + [-new_row[i]] for i, r in enumerate(parent)]
    rows.append(list(new_row) + [0])
    return tuple(tuple(r) for r in rows)


def children(parent: Matrix, h: int) -> list[Matrix]:
    """All key-minimal one-vertex extensions of a key-minimal parent."""
    k = len(parent)
    out: list[Matrix] = []
    for new_row in itertools.product(range(-h, h + 1), repeat=k):
        # Cheap necessary condition: the new (last) key row must not be smaller
        # than the parent's last key row when the two vertices are swapped —
        # otherwise the swap gives a smaller key. Full test follows.
        child = _extend(parent, new_row)
        if is_key_minimal(child):
            out.append(child)
    return out


def _children_job(args: tuple[Matrix, int]) -> list[Matrix]:
    return children(*args)


def generate_cell(n: int, h: int, workers: int = 1,
                  progress=None) -> Iterator[Matrix]:
    """
    One key-minimal representative per isomorphism class of (n, h)-quivers,
    level by level from the single 1-vertex quiver. With workers > 1 the
    extension of each level's parents is spread over a process pool.
    """
    level: list[Matrix] = [((0,),)]
    for k in range(2, n + 1):
        nxt: list[Matrix] = []
        if workers > 1 and len(level) > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(workers) as pool:
                for kids in pool.imap_unordered(_children_job, [(p, h) for p in level], chunksize=8):
                    nxt.extend(kids)
        else:
            for p in level:
                nxt.extend(children(p, h))
        nxt.sort()
        level = nxt
        if progress:
            progress(k, len(level))
    yield from level


def census_seeds(n: int, h: int, workers: int = 1, progress=None,
                 connected_only: bool = True) -> list[Matrix]:
    """
    The cell (n, h) as ID-canonical (lex-min) matrices, sorted — run_generation
    seeds. Generation must pass through disconnected intermediates (the
    canonical block of a connected quiver need not be connected), so the
    connected filter is applied to the finished level only.
    """
    from qmd.core import is_connected
    reps = generate_cell(n, h, workers=workers, progress=progress)
    seeds = [canonical_form(m) for m in reps if not connected_only or is_connected(m)]
    seeds.sort()
    return seeds


# ---------------------------------------------------------------------------
# Sampling (for cells too large to enumerate)
# ---------------------------------------------------------------------------

def sample_cell(n: int, h: int, k: int, seed: int = 0, connected_only: bool = False) -> list[Matrix]:
    """
    k distinct quivers drawn by sampling labeled matrices uniformly from the
    cell (each upper entry uniform in [-h, h]) and canonicalising. Not uniform
    over isomorphism classes (symmetric quivers are under-represented, exactly
    as they are among labeled matrices) — document this with any ML dataset.
    """
    from qmd.core import is_connected
    rng = random.Random(seed)
    out: set[Matrix] = set()
    upper = [(i, j) for i in range(n) for j in range(i + 1, n)]
    attempts = 0
    while len(out) < k and attempts < 50 * k + 1000:
        attempts += 1
        rows = [[0] * n for _ in range(n)]
        for i, j in upper:
            v = rng.randint(-h, h)
            rows[i][j], rows[j][i] = v, -v
        m = tuple(tuple(r) for r in rows)
        if connected_only and not is_connected(m):
            continue
        out.add(canonical_form(m))
    return sorted(out)
