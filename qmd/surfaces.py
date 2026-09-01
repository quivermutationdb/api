"""
qmd/surfaces.py — marked surfaces, their ideal triangulations, and the quivers
those triangulations carry (Fomin–Shapiro–Thurston).

Every quiver from a triangulated marked surface is mutation-finite, and the
flip graph of a surface is connected, so ONE triangulation per surface yields
the whole mutation class. This module turns that into a table

    mc_id -> surface name

for every surface class of a given rank — the surface analogue of
`qmd.dynkin.reference_for`, and the reason a rank never has to be classified
by hand.

Combinatorial model
-------------------
A triangulation is `t` triangles glued along sides. Triangle T owns sides
3T, 3T+1, 3T+2 in counter-clockwise order, and side 3T+k runs from corner k to
corner (k+1) mod 3 of T — so a side's index doubles as the index of the corner
it starts at. A `matching` pairs glued sides; unmatched sides are boundary
segments. Gluing is orientation-reversing on the shared side, which is exactly
what keeps the result an *oriented* surface:

    gluing 3A+i to 3B+j  identifies  corner_i(A) ~ corner_{j+1}(B)
                               and   corner_{i+1}(A) ~ corner_j(B)

Genus, boundary components and punctures then follow from Euler's formula, so
nothing here needs any topology beyond V - E + F.

Marked points are the corner classes. Those met by an unmatched side lie on the
boundary; the rest are punctures. For an ideal triangulation a boundary
component with m segments carries exactly m marked points, so the number of
boundary marked points must come out equal to the number of unmatched sides —
that identity is asserted, and a matching that violates it describes a pinched
complex rather than a surface and is rejected.
"""

from __future__ import annotations

import json
import os
import random
from typing import Iterator, Optional

from qmd.core import (
    Matrix, _bfs_unlabeled, _lex_key, canonical_form, is_connected,
    mutation_class_id, to_matrix,
)

# n = 6g + 3b + 3p + c - 6, with c = total marked points on the boundary.
# Rearranged for the triangle count: every triangle has 3 sides, each arc is
# shared by two of them and each boundary segment by one, so 3t = 2n + c.


def _find(parent: list, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list, a: int, b: int) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def corner_classes(t: int, matching: dict) -> list:
    """Union-find over the 3t corners; corner 3T+k is where side 3T+k starts."""
    parent = list(range(3 * t))
    for a, b in matching.items():
        if a > b:
            continue
        A, i = divmod(a, 3)
        B, j = divmod(b, 3)
        _union(parent, 3 * A + i, 3 * B + (j + 1) % 3)
        _union(parent, 3 * A + (i + 1) % 3, 3 * B + j)
    return parent


def boundary_cycles(t: int, matching: dict) -> list:
    """The unmatched sides, grouped into the boundary components they trace."""
    seen: set = set()
    cycles = []
    for start in range(3 * t):
        if start in matching or start in seen:
            continue
        cycle = []
        s = start
        while True:
            cycle.append(s)
            seen.add(s)
            T, k = divmod(s, 3)
            nxt = 3 * T + (k + 1) % 3      # next side ccw out of this side's end corner
            while nxt in matching:         # cross glued sides until the boundary again
                B, j = divmod(matching[nxt], 3)
                nxt = 3 * B + (j + 1) % 3
            s = nxt
            if s == start:
                break
            if s in seen:                  # not a disjoint union of cycles: degenerate
                return []
        cycles.append(cycle)
    return cycles


def _triangles_connected(t: int, matching: dict) -> bool:
    parent = list(range(t))
    for a, b in matching.items():
        _union(parent, a // 3, b // 3)
    return len({_find(parent, x) for x in range(t)}) == 1


def signature(t: int, matching: dict) -> Optional[tuple]:
    """
    (g, b, p, sorted marked-points-per-boundary-component), or None if this
    gluing is not an ideal triangulation of a connected oriented surface.
    """
    n = len(matching) // 2
    c = 3 * t - 2 * n
    if not _triangles_connected(t, matching):
        return None
    for a, b in matching.items():                  # no self-folded triangles
        if a // 3 == b // 3:
            return None
    cycles = boundary_cycles(t, matching)
    if sum(len(cy) for cy in cycles) != c:
        return None
    parent = corner_classes(t, matching)
    classes = {_find(parent, x) for x in range(3 * t)}
    on_boundary = {_find(parent, s) for cy in cycles for s in cy}
    if len(on_boundary) != c:                      # pinched boundary
        return None
    v, e, f = len(classes), n + c, t
    chi = v - e + f
    b = len(cycles)
    if (2 - b - chi) % 2:
        return None
    g = (2 - b - chi) // 2
    if g < 0:
        return None
    p = len(classes) - len(on_boundary)
    return (g, b, p, tuple(sorted(len(cy) for cy in cycles)))


def quiver_of(t: int, matching: dict) -> Optional[Matrix]:
    """
    The adjacency quiver: inside each triangle an arrow runs from each arc to
    the next one counter-clockwise, and opposite arrows cancel.
    """
    arcs = sorted({min(s, matching[s]) for s in matching})
    if not arcs:
        return None
    idx = {a: i for i, a in enumerate(arcs)}
    n = len(arcs)
    rows = [[0] * n for _ in range(n)]
    for T in range(t):
        for k in range(3):
            s, s2 = 3 * T + k, 3 * T + (k + 1) % 3
            if s not in matching or s2 not in matching:
                continue
            a, b = idx[min(s, matching[s])], idx[min(s2, matching[s2])]
            if a == b:
                continue
            rows[a][b] += 1
            rows[b][a] -= 1
    m = to_matrix(rows)
    return m if is_connected(m) else None


def enumerate_signatures(n: int) -> set:
    """
    Every (g, b, p, boundary distribution) with n arcs.

    Excludes the cases with no ideal triangulation: a closed surface needs at
    least one puncture to have any marked point at all, and a sphere needs four
    (with three it has only 3 arcs and the theory degenerates).
    """
    out = set()
    for g in range(0, n // 6 + 2):
        for b in range(0, n + 2):
            for p in range(0, n + 2):
                c = n - (6 * g + 3 * b + 3 * p - 6)
                if c < 0 or c < b:
                    continue
                if b == 0:
                    if c or p == 0 or (g == 0 and p < 4):
                        continue
                    out.add((g, 0, p, ()))
                    continue
                for dist in _compositions(c, b):
                    out.add((g, b, p, dist))
    return out


def _compositions(total: int, parts: int) -> Iterator[tuple]:
    """Sorted multisets of `parts` positive integers summing to `total`."""
    def rec(remaining: int, slots: int, least: int, acc: tuple):
        if slots == 0:
            if remaining == 0:
                yield acc
            return
        for v in range(least, remaining - slots + 2):
            yield from rec(remaining - v, slots - 1, v, acc + (v,))
    yield from rec(total, parts, 1, ())


_ORDINAL = {0: "", 1: "once-", 2: "twice-", 3: "thrice-"}


def _punct(p: int) -> str:
    return _ORDINAL.get(p, f"{p}-times-") if p else ""


def name_of(sig: tuple) -> str:
    """A readable name; the tuple itself stays the machine-readable key."""
    g, b, p, m = sig
    marks = ",".join(str(x) for x in m)
    if b == 0:
        base = "sphere" if g == 0 else ("torus" if g == 1 else f"genus-{g} surface")
        return f"{p}-punctured {base}" if g == 0 else f"{_punct(p)}punctured {base}"
    if g == 0 and b == 1:
        return f"{_punct(p)}punctured {m[0]}-gon" if p else f"{m[0]}-gon"
    if g == 0 and b == 2:
        ann = f"annulus({marks})"
        return f"{_punct(p)}punctured {ann}" if p else ann
    if g == 0 and b == 3 and p == 0:
        return f"pair of pants({marks})"
    base = "torus" if g == 1 else f"genus-{g} surface"
    plural = "s" if b > 1 else ""
    tail = f"{base} with {b} boundary component{plural} ({marks} marked)"
    return f"{_punct(p)}punctured {tail}" if p else tail


def classical_name(sig: tuple) -> Optional[str]:
    """
    The Dynkin-style name where the surface has one.

    A polygon is type A, a once-punctured polygon type D, a twice-punctured
    polygon affine D, and an annulus affine A. The arc counts line up: a
    twice-punctured m-gon has m + 3 arcs and D~k has k + 1 vertices, hence
    k = m + 2. These are checked against qmd.dynkin for the finite cases.
    """
    g, b, p, m = sig
    if g or b != 1 and b != 2:
        return None
    if b == 1 and p == 0:
        return f"A{m[0] - 3}"
    if b == 1 and p == 1:
        return f"D{m[0]}"
    if b == 1 and p == 2:
        return f"D~{m[0] + 2}"
    if b == 2 and p == 0:
        return f"A~({m[0]},{m[1]})"
    return None


def _sample(t: int, pairs: int, rng: random.Random) -> dict:
    sides = list(range(3 * t))
    rng.shuffle(sides)
    chosen = sides[:2 * pairs]
    return {chosen[i ^ 1]: chosen[i] for i in range(len(chosen))}


def triangulations_for(n: int, *, samples: int = 200_000, seed: int = 0,
                       log=None) -> dict:
    """
    One triangulation per realisable surface with n arcs: {signature: (t, matching)}.

    Sampling rather than exhaustive enumeration — the number of matchings on 3t
    sides is astronomical by n = 8, while the number of *surfaces* is tiny, so
    random gluings saturate the signature set long before the budget runs out.
    Seeded, hence reproducible. Coverage is checked by the caller against
    enumerate_signatures, so a miss is reported rather than silently dropped.
    """
    rng = random.Random(seed)
    found: dict = {}
    wanted = enumerate_signatures(n)
    for c in range(0, n + 4):
        if (2 * n + c) % 3:
            continue
        t = (2 * n + c) // 3
        if t <= 0:
            continue
        stale = 0
        for _ in range(samples):
            matching = _sample(t, n, rng)
            if len(matching) != 2 * n:
                break
            sig = signature(t, matching)
            if sig is None or sig in found:
                stale += 1
                if stale > samples // 4 and found:
                    break
                continue
            if quiver_of(t, matching) is None:
                continue
            found[sig] = (t, dict(matching))
            stale = 0
            if log:
                log(f"    n={n} t={t} c={c}: {name_of(sig)}")
        if wanted <= set(found):
            break
    return found


REFERENCE_CACHE = os.environ.get("QMD_SURFACE_CACHE", os.path.join(
    os.path.dirname(__file__), "..", "dist", "surface-reference.json"))

_REFERENCE: dict = {}


def reference_for(rank: int, *, samples: int = 200_000, seed: int = 0,
                  log=None) -> dict:
    """
    mc_id -> surface name for every surface mutation class of this rank.

    Cached on disk, and computed once per process otherwise; call it in the
    parent before forking a worker pool.
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
        # Distinct surfaces can share a mutation class — the once-punctured
        # triangle is A3, and the twice-punctured monogon is the (2,2) annulus.
        # Those coincidences are real mathematics, so keep every name rather
        # than letting whichever was sampled first win.
        names: dict = {}
        for sig, (t, matching) in sorted(triangulations_for(
                rank, samples=samples, seed=seed, log=log).items()):
            q = quiver_of(t, matching)
            orbit = _bfs_unlabeled(q, 2, None)
            mc_id = mutation_class_id(min(orbit.members, key=_lex_key))
            names.setdefault(mc_id, []).append(name_of(sig))
        cache[key] = {mc: " = ".join(v) for mc, v in names.items()}
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


SURFACE_MAX_RANK = int(os.environ.get("QMD_SURFACE_MAX_RANK", 8))


def reference_for_rank(rank: int) -> dict:
    """
    `reference_for` with a guard.

    Building the table explores every surface class of the rank, which is
    quick through rank 8 (~85 s there, cached afterwards) but grows with the
    class sizes. Above the guard the table comes back empty rather than
    stalling an export; raise QMD_SURFACE_MAX_RANK when a higher rank is
    actually wanted.
    """
    if rank < 3 or rank > SURFACE_MAX_RANK:
        return {}
    return reference_for(rank)


def classify(canonical_rep: Matrix, mc_id: Optional[str] = None) -> Optional[str]:
    """
    The surface this mutation class comes from, or None if it comes from none
    (E6, X6, X7 and the other exceptional classes, or any mutation-infinite
    class).

    Only meaningful for completely explored classes. Pass `mc_id` when it is
    already known to avoid re-exploring the class.
    """
    if not is_connected(canonical_rep):
        return None
    table = reference_for_rank(len(canonical_rep))
    if not table:
        return None
    if mc_id is None:
        orbit = _bfs_unlabeled(canonical_rep, 2, None)
        if orbit.is_open:
            return None
        mc_id = mutation_class_id(min(orbit.members, key=_lex_key))
    return table.get(mc_id)


def seed_quivers(rank: int) -> list:
    """
    One canonical seed per surface class of this rank, for the generator.

    A census cell is bounded (rank 7 and 8 are taken at |b_ij| <= 1), so a
    mutation-finite class whose every member carries a double arrow would never
    be seeded and would vanish from the dataset. Feeding the surface quivers in
    as curated seeds removes that whole failure mode: every surface class is
    present because it was constructed, not because the cell happened to
    contain it.
    """
    out = []
    if rank < 3 or rank > SURFACE_MAX_RANK:
        return out
    for _sig, (t, matching) in sorted(triangulations_for(rank).items()):
        q = quiver_of(t, matching)
        if q is not None:
            out.append(canonical_form(q))
    return out
