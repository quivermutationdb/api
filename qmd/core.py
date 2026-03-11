"""
qmd/core.py

Core data structures and algorithms for the Quiver Mutation Database.

Data model
----------
Quiver (Q.*)
    An *unlabeled* quiver — one record per isomorphism class of
    skew-symmetric integer matrix.  canonical_form() is called before
    hashing, so isomorphic matrices always map to the same Q.* id.

MutationClass (MC.*)
    The union of all bounded BFS orbits that share at least one unlabeled
    quiver (Q.* id).  Sharing is transitive: if orbit A and orbit B share
    a quiver, and orbit B and orbit C share a quiver, then A, B, and C all
    collapse into one mutation class.

    labeled_quivers is the union of every labeled matrix from every
    merged orbit.

    is_open is True if ANY constituent orbit hit the |b_ij| > bound
    boundary.

    The MC.* id is hashed from canonical_class_rep() computed over the
    full union of labeled matrices.

Membership
    Every labeled matrix in the union maps to exactly one MC.* id and
    (via canonical_form) to one Q.* id.

Key theorem (orbit uniqueness under shared quivers)
    If orbit A and orbit B share any quiver Q, and orbit A is closed,
    then orbit B is also closed and A = B (as labeled sets, up to a
    global vertex relabeling).

    Proof: Since A is closed, A contains every matrix reachable from Q
    by any sequence of bounded mutations.  BFS from Q is deterministic
    given the bound, so B must contain the same set — meaning B = A and
    B is also closed.

    Consequence: the ONLY valid gluing case is open + open.
    The other two cases are both bugs:

      closed + closed sharing a quiver
          Canonicalization bug: the same mutation class produced two
          different canonical reps and therefore two different mc_ids.

      closed + open sharing a quiver
          BFS bug: the open orbit failed to fully explore a class that
          is bounded.  It should have terminated as closed.

    Both cases raise AssertionError immediately in the pipeline.

Gluing breakdown (tracked in GenerationResult)
    closed_closed_merges  -- must always be 0 (canonicalization bug if not)
    closed_open_merges    -- must always be 0 (BFS bug if not)
    open_open_gluings     -- the only valid case; two partial explorations
                             of the same unbounded class joined together

Gluing algorithm
    Union-Find (disjoint set union) over raw BFS orbits.  Each orbit
    starts as its own component.  For every pair of orbits that share a
    Q.* id, their components are unioned.  This is O(alpha(n)) per
    union and handles transitivity automatically.  After all unions,
    each component is reduced to a single MutationClassResult by merging
    labeled matrices, recomputing canonical_class_rep, and re-deriving
    the MC.* id.
"""

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import permutations, product

from qmd.canonicalize import (
    canonical_form,
    are_isomorphic,
    active_backend,
    verify_with_fallback,
    _apply_permutation,
    _lex_key,
)


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Matrix = tuple[tuple[int, ...], ...]


# ---------------------------------------------------------------------------
# Exchange matrix helpers
# ---------------------------------------------------------------------------

def to_matrix(rows: list[list[int]]) -> Matrix:
    """Convert a list-of-lists to an immutable Matrix."""
    return tuple(tuple(row) for row in rows)


def to_lists(matrix: Matrix) -> list[list[int]]:
    """Convert a Matrix back to a list-of-lists (for serialisation)."""
    return [list(row) for row in matrix]


def is_skew_symmetric(matrix: Matrix) -> bool:
    n = len(matrix)
    return all(matrix[i][j] == -matrix[j][i] for i in range(n) for j in range(n))


def max_edge(matrix: Matrix) -> int:
    """Return the maximum absolute value of any entry."""
    return max(abs(matrix[i][j])
               for i in range(len(matrix))
               for j in range(len(matrix[i])))


def is_bounded(matrix: Matrix, bound: int = 2) -> bool:
    """True iff every entry satisfies |b_ij| <= bound."""
    return max_edge(matrix) <= bound


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutate(matrix: Matrix, k: int) -> Matrix:
    """
    Return the matrix obtained by mutating at vertex k.

    Standard skew-symmetric mutation rule:
        b'_ij = -b_ij                                          if i==k or j==k
        b'_ij = b_ij + (|b_ik|*b_kj + b_ik*|b_kj|) / 2      otherwise
    """
    n = len(matrix)
    rows = [list(row) for row in matrix]
    for i in range(n):
        for j in range(n):
            if i == k or j == k:
                rows[i][j] = -matrix[i][j]
            else:
                b_ik = matrix[i][k]
                b_kj = matrix[k][j]
                rows[i][j] = (matrix[i][j]
                               + (abs(b_ik) * b_kj + b_ik * abs(b_kj)) // 2)
    return to_matrix(rows)


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def _hash_matrix(matrix: Matrix, prefix: str) -> str:
    """
    Deterministic SHA-256 hash of a canonical matrix, truncated to 16 hex chars.
    Serialisation: compact JSON, row-major order.
    Callers must pass an already-canonical matrix.
    """
    serialized = json.dumps(to_lists(matrix), separators=(',', ':'))
    h = hashlib.sha256(serialized.encode()).hexdigest()[:16]
    return f"{prefix}.n{len(matrix)}.{h}"


def quiver_id(matrix: Matrix) -> str:
    """
    Stable identifier for an *unlabeled* quiver (isomorphism class).
    Format: Q.n{vertices}.{sha256[:16]}

    Calls canonical_form() before hashing — isomorphic matrices always
    produce the same id.
    """
    return _hash_matrix(canonical_form(matrix), "Q")


def mutation_class_id(canonical_rep: Matrix) -> str:
    """
    Stable identifier for an unlabeled mutation class.
    Format: MC.n{vertices}.{sha256[:16]}

    canonical_rep must be the output of canonical_class_rep() computed
    over the FULL merged labeled orbit.
    """
    return _hash_matrix(canonical_rep, "MC")


# ---------------------------------------------------------------------------
# Mutation class canonical representative
# ---------------------------------------------------------------------------

def canonical_class_rep(labeled_matrices: list[Matrix]) -> Matrix:
    """
    Canonical representative of a mutation class.

    Returns the lex-min matrix over ALL vertex relabelings of ALL labeled
    matrices in the orbit.  Invariant under:
      (a) which quiver in the orbit BFS started from,
      (b) which vertex labeling the seed carries, and
      (c) which subset of orbits have been merged so far (adding more
          matrices can only decrease or maintain the minimum).
    """
    return min(
        (_apply_permutation(m, perm)
         for m in labeled_matrices
         for perm in permutations(range(len(m)))),
        key=_lex_key,
    )


# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set Union)
# ---------------------------------------------------------------------------

class _UnionFind:
    """
    Union-Find with path compression and union by rank.

    Nodes are integer indices.  Used to group raw BFS orbits that share
    at least one unlabeled quiver id.
    """

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank   = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path halving
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Union the sets containing x and y.  Returns True if they were distinct."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        return True

    def components(self, indices: list[int]) -> dict[int, list[int]]:
        """
        Return a dict mapping root -> list of members for all given indices.
        """
        groups: dict[int, list[int]] = defaultdict(list)
        for i in indices:
            groups[self.find(i)].append(i)
        return dict(groups)


# ---------------------------------------------------------------------------
# BFS mutation-class explorer  (single orbit, pre-gluing)
# ---------------------------------------------------------------------------

@dataclass
class _RawOrbit:
    """
    The direct output of a single BFS from one seed.
    Multiple raw orbits may be glued into one MutationClassResult.
    """
    labeled_quivers : list[Matrix]
    quiver_ids      : list[str]       # parallel to labeled_quivers
    qid_set         : set[str]        # fast membership test
    is_open         : bool
    boundary_quivers: list[Matrix]


def _bfs_orbit(seed: Matrix, bound: int = 2) -> _RawOrbit:
    """
    BFS over all matrices reachable from `seed` by bounded mutation.
    Returns the raw labeled orbit without any gluing.
    """
    assert is_skew_symmetric(seed), "Seed must be skew-symmetric"
    assert is_bounded(seed, bound), \
        f"Seed already violates the bound |b_ij| <= {bound}"

    visited : set[Matrix] = set()
    boundary: set[Matrix] = set()
    queue   : deque[Matrix] = deque([seed])
    visited.add(seed)
    is_open = False

    while queue:
        current = queue.popleft()
        at_boundary = False
        for k in range(len(current)):
            mutated = mutate(current, k)
            if not is_bounded(mutated, bound):
                is_open = True
                at_boundary = True
            else:
                if mutated not in visited:
                    visited.add(mutated)
                    queue.append(mutated)
        if at_boundary:
            boundary.add(current)

    labeled = list(visited)
    qids    = [quiver_id(m) for m in labeled]
    return _RawOrbit(
        labeled_quivers  = labeled,
        quiver_ids       = qids,
        qid_set          = set(qids),
        is_open          = is_open,
        boundary_quivers = list(boundary),
    )


# ---------------------------------------------------------------------------
# MutationClassResult  (post-gluing)
# ---------------------------------------------------------------------------

@dataclass
class MutationClassResult:
    """
    A fully-merged mutation class.

    Attributes
    ----------
    labeled_quivers : list[Matrix]
        Union of all labeled matrices from every merged BFS orbit.
        For A3 this is 14 entries.  For a glued class it is the union
        of the contributing orbits' labeled matrices (duplicates removed).

    quiver_ids : list[str]
        Parallel to labeled_quivers.  labeled_quivers[i] maps to
        quiver_ids[i].  Multiple entries may share the same Q.* id
        (same unlabeled quiver under different vertex labelings).

    canonical_rep : Matrix
        Lex-min over all vertex relabelings of all labeled_quivers.
        Recomputed after gluing.

    mc_id : str
        MC.* id derived from canonical_rep.

    is_open : bool
        True if ANY constituent orbit hit the |b_ij| > bound boundary.

    boundary_quivers : list[Matrix]
        Union of boundary_quivers from all constituent orbits.

    merged_orbit_count : int
        Number of raw BFS orbits that were glued to form this class.
        1 means no gluing occurred.
    """
    labeled_quivers    : list[Matrix]
    quiver_ids         : list[str]
    canonical_rep      : Matrix
    mc_id              : str
    is_open            : bool
    boundary_quivers   : list[Matrix] = field(default_factory=list)
    merged_orbit_count : int = 1

    @property
    def labeled_size(self) -> int:
        return len(self.labeled_quivers)

    @property
    def distinct_quiver_count(self) -> int:
        return len(set(self.quiver_ids))


def _merge_orbits(orbits: list[_RawOrbit]) -> MutationClassResult:
    """
    Merge a list of _RawOrbit objects (already determined to belong to
    the same mutation class) into a single MutationClassResult.

    Deduplicates labeled matrices across orbits (a matrix appearing in
    two orbits is stored once).  Recomputes canonical_rep and mc_id
    from the merged labeled set.
    """
    seen_labeled : set[Matrix] = set()
    merged_labeled  : list[Matrix] = []
    merged_qids     : list[str]    = []
    merged_boundary : list[Matrix] = []
    is_open = False

    for orbit in orbits:
        is_open = is_open or orbit.is_open
        for m, qid in zip(orbit.labeled_quivers, orbit.quiver_ids):
            if m not in seen_labeled:
                seen_labeled.add(m)
                merged_labeled.append(m)
                merged_qids.append(qid)
        for b in orbit.boundary_quivers:
            if b not in seen_labeled:
                merged_boundary.append(b)

    canon_rep = canonical_class_rep(merged_labeled)
    mc_id_str = mutation_class_id(canon_rep)

    return MutationClassResult(
        labeled_quivers    = merged_labeled,
        quiver_ids         = merged_qids,
        canonical_rep      = canon_rep,
        mc_id              = mc_id_str,
        is_open            = is_open,
        boundary_quivers   = merged_boundary,
        merged_orbit_count = len(orbits),
    )


# ---------------------------------------------------------------------------
# Public BFS entry point  (single seed, no pipeline context)
# ---------------------------------------------------------------------------

def explore_mutation_class(seed: Matrix, bound: int = 2) -> MutationClassResult:
    """
    Explore the mutation class of a single seed via bounded BFS.

    This is the single-seed entry point used in tests and interactive work.
    It does NOT perform cross-orbit gluing (that requires the full pipeline
    context).  Use run_generation() for complete gluing behaviour.

    Parameters
    ----------
    seed  : starting exchange matrix (skew-symmetric and bounded)
    bound : maximum allowed |b_ij| at each step (default 2)
    """
    orbit = _bfs_orbit(seed, bound)
    return _merge_orbits([orbit])


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------

def generate_seed_quivers(max_vertices: int = 4, bound: int = 2) -> list[Matrix]:
    """
    Enumerate one representative per isomorphism class of skew-symmetric
    {0, +-1, ..., +-bound}-matrices of size n x n, for n in [1, max_vertices].

    Returns canonical-form matrices — one per distinct unlabeled quiver.
    """
    seen  : set[Matrix]   = set()
    seeds : list[Matrix]  = []

    for n in range(1, max_vertices + 1):
        upper = [(i, j) for i in range(n) for j in range(i + 1, n)]
        for combo in product(range(-bound, bound + 1), repeat=len(upper)):
            rows = [[0] * n for _ in range(n)]
            for (i, j), v in zip(upper, combo):
                rows[i][j] =  v
                rows[j][i] = -v
            m  = to_matrix(rows)
            cf = canonical_form(m)
            if cf not in seen:
                seen.add(cf)
                seeds.append(cf)

    return seeds


# ---------------------------------------------------------------------------
# Full pipeline with union-find gluing
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """
    Output of the full generation pipeline.

    quivers    : dict  quiver_id -> canonical Matrix
                 One entry per distinct unlabeled quiver.

    classes    : dict  mc_id -> MutationClassResult
                 One entry per merged mutation class.

    membership : dict  quiver_id -> mc_id

    Gluing counters (see module docstring for the orbit uniqueness theorem):

    closed_closed_merges : int
        Merges between two closed orbits sharing a quiver.
        MUST be 0.  Non-zero means a canonicalization bug.

    closed_open_merges : int
        Merges between a closed orbit and an open orbit sharing a quiver.
        MUST be 0.  Non-zero means a BFS bug (open orbit should have
        terminated as closed).

    open_open_gluings : int
        Merges between two open orbits sharing a quiver.
        The only valid gluing case.

    total_gluings : property
        Sum of all three counters.  Only open_open_gluings should be > 0.
    """
    quivers              : dict[str, Matrix]              = field(default_factory=dict)
    classes              : dict[str, MutationClassResult] = field(default_factory=dict)
    membership           : dict[str, str]                 = field(default_factory=dict)
    closed_closed_merges : int = 0
    closed_open_merges   : int = 0
    open_open_gluings    : int = 0

    @property
    def total_gluings(self) -> int:
        return self.closed_closed_merges + self.closed_open_merges + self.open_open_gluings


def run_generation(max_vertices: int = 4, bound: int = 2) -> GenerationResult:
    """
    Full four-phase generation pipeline.

    Phase 1 — Seed enumeration
        One canonical seed per unlabeled quiver isomorphism class.

    Phase 2 — BFS per seed
        Run _bfs_orbit() for each seed not already covered.
        Collect all raw orbits; record which Q.* ids appear in each.

    Phase 3 — Union-Find gluing
        Build a union-find over orbit indices.  For each Q.* id that
        appears in more than one orbit, union those orbits.  Transitivity
        is handled automatically.  After all unions, collect the connected
        components and merge each component into a single MutationClassResult
        via _merge_orbits().

    Phase 4 — Consistency assertions
        Every mc_id key must equal mutation_class_id(canonical_rep).
        Every quiver_id key must equal quiver_id(stored matrix).
    """
    result = GenerationResult()
    seeds  = generate_seed_quivers(max_vertices, bound)

    # --- Phase 2: BFS ---
    # We run BFS for every seed whose quiver_id is not yet covered.
    # We cannot skip seeds based on membership yet because gluing hasn't
    # happened; a seed might be in a merged class but we need its orbit
    # to perform the merge.
    raw_orbits   : list[_RawOrbit] = []
    covered_qids : set[str]        = set()  # qids seen in any orbit so far

    for seed in seeds:
        seed_qid = quiver_id(seed)
        if seed_qid in covered_qids:
            continue
        orbit = _bfs_orbit(seed, bound)
        raw_orbits.append(orbit)
        covered_qids.update(orbit.qid_set)

    # --- Phase 3: Union-Find gluing ---
    n_orbits = len(raw_orbits)
    uf = _UnionFind(n_orbits)

    # Build an inverted index: qid -> list of orbit indices containing it
    qid_to_orbits: dict[str, list[int]] = defaultdict(list)
    for idx, orbit in enumerate(raw_orbits):
        for qid in orbit.qid_set:
            qid_to_orbits[qid].append(idx)

    # Union all orbits that share a qid.
    # By the orbit uniqueness theorem, the ONLY valid case is open+open.
    # closed+closed and closed+open both indicate bugs and raise immediately.
    for qid, orbit_indices in qid_to_orbits.items():
        for i in range(1, len(orbit_indices)):
            a, b = orbit_indices[0], orbit_indices[i]
            if uf.union(a, b):
                a_open = raw_orbits[a].is_open
                b_open = raw_orbits[b].is_open

                if not a_open and not b_open:
                    raise AssertionError(
                        f"closed+closed merge on qid={qid!r} "
                        f"(orbits {a} and {b}): two closed orbits share a "
                        f"quiver, which violates the orbit uniqueness theorem. "
                        f"This indicates a canonicalization bug."
                    )
                elif not a_open or not b_open:
                    raise AssertionError(
                        f"closed+open merge on qid={qid!r} "
                        f"(orbits {a} and {b}): a closed orbit and an open "
                        f"orbit share a quiver, which means the open orbit "
                        f"failed to fully explore a bounded mutation class. "
                        f"This indicates a BFS bug."
                    )
                else:
                    # open + open: the only valid gluing case
                    result.open_open_gluings += 1

    # Collect components and merge
    components = uf.components(list(range(n_orbits)))
    for root, members in components.items():
        orbits_in_component = [raw_orbits[i] for i in members]
        mc_result = _merge_orbits(orbits_in_component)

        result.classes[mc_result.mc_id] = mc_result

        for m, qid in zip(mc_result.labeled_quivers, mc_result.quiver_ids):
            if qid not in result.quivers:
                result.quivers[qid] = canonical_form(m)
            result.membership[qid] = mc_result.mc_id

    # --- Phase 4: Consistency assertions ---
    for mc_id_key, mc_res in result.classes.items():
        computed = mutation_class_id(mc_res.canonical_rep)
        assert computed == mc_id_key, (
            f"mc_id mismatch: key={mc_id_key!r}, computed={computed!r}"
        )
    for qid_key, m in result.quivers.items():
        assert quiver_id(m) == qid_key, (
            f"quiver_id mismatch: key={qid_key!r}, computed={quiver_id(m)!r}"
        )

    return result
