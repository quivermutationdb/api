"""
qmd/class_properties.py

Pure (database-free) resolution of per-class properties, used by the D1
export path (qmd/d1_export.py). Extracted from the legacy Postgres writer
(qmd/crud.py, removed post-migration) so generation runs on a bare box with
no database driver installed.
"""

from __future__ import annotations

from typing import Optional

from qmd.core import quiver_id

TRI = {"true": True, "false": False, "unknown": None}


def la_bounds(is_open: bool) -> dict:
    """Search budget: thorough for finite classes (they certify), bounded for open ones."""
    return dict(max_depth=8, timeout=3, cap=8) if is_open \
        else dict(max_depth=64, timeout=15, cap=8)


# ---------------------------------------------------------------------------
# Mutation-acyclicity resolution (subquiver fallback)
# ---------------------------------------------------------------------------
#
# A BFS over a bounded mutation class can prove is_mutation_acyclic = True (it
# found an acyclic member) and, for a closed class, prove False (the whole class
# was explored, none acyclic).  For an *open* class with no acyclic member found
# it can only say "unknown".
#
# The subquiver fallback upgrades such an "unknown" to a definite False:
# mutation-acyclicity is hereditary under induced subquivers, so if any member
# of the class has a proper induced subquiver that is already known NOT to be
# mutation-acyclic, the whole class is not mutation-acyclic.  The base case is
# the Markov quiver (rank 3, its own closed class -> proved False), and
# rank-ordered resolution lets that propagate upward.

def _delete_vertex(m, k):
    """Induced subquiver obtained by deleting vertex k (rows/cols)."""
    idx = [i for i in range(len(m)) if i != k]
    return tuple(tuple(m[a][b] for b in idx) for a in idx)


def _has_known_non_ma_subquiver(matrices, mut_by_qid: dict) -> bool:
    """
    True if some member has an induced subquiver already known to be NOT
    mutation-acyclic.  By heredity a not-mutation-acyclic subquiver of any size
    forces a not-mutation-acyclic (n-1)-subquiver, so deleting a single vertex
    is enough to find a witness.
    """
    from qmd.dynkin import _connected_components
    for m in matrices:
        n = len(m)
        if n < 4:          # the smallest non-mutation-acyclic quiver is rank 3 (Markov)
            continue
        for k in range(n):
            sub = _delete_vertex(m, k)
            # The census stores connected quivers only, so a disconnected
            # subquiver is looked up component by component: it is not
            # mutation-acyclic iff some component is not.
            for comp in _connected_components(sub):
                if len(comp) >= 3 and mut_by_qid.get(quiver_id(comp)) is False:
                    return True
    return False


def resolve_mutation_acyclic(class_infos,
                             known: Optional[dict] = None) -> dict:
    """
    Resolve is_mutation_acyclic for every class in ascending rank order.

    class_infos: iterable of (mc_id, n_vertices, base_state, matrices, qids),
    where base_state is the BFS result (True / False / None) and matrices /
    qids are the class's explored members and their unlabeled quiver ids.

    `known` seeds the quiver-level state map with already-resolved quivers
    from lower ranks — this is what lets a per-rank (checkpointed) run reach
    the same answers as a single all-ranks run, since the subquiver fallback
    looks one rank down.

    Returns {mc_id: True | False | None}.  Only "unknown" (None) values can be
    upgraded — to False — and never the reverse, so the pass is sound.
    """
    final: dict = {}
    mut_by_qid: dict = dict(known) if known else {}
    for mc_id, _n, base, matrices, qids in sorted(class_infos, key=lambda c: c[1]):
        state = base
        if state is None and _has_known_non_ma_subquiver(matrices, mut_by_qid):
            state = False
        final[mc_id] = state
        for qid in qids:
            mut_by_qid[qid] = state
    return final
