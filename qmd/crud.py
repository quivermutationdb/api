"""
qmd/crud.py

Database read/write operations.

Reads translate internal ORM rows into the frontend-facing shapes
(qmd_id, num_vertices, exchange_matrix, class_size, ...).  Writes compute
and persist the per-quiver invariants and the per-class labeled orbit.
"""

import json
from typing import Optional

from sqlalchemy.orm import Session

from qmd import dynkin, invariants, models
from qmd import local_acyclicity as la
from qmd.core import (
    GenerationResult,
    canonical_form,
    to_matrix,
    to_lists,
    quiver_id,
    max_edge,
    is_acyclic,
    is_connected,
)

_TRI = {"true": True, "false": False, "unknown": None}


def _la_bounds(is_open: bool) -> dict:
    """Search budget: thorough for finite classes (they certify), bounded for open ones."""
    return dict(max_depth=8, timeout=3, cap=8) if is_open \
        else dict(max_depth=64, timeout=15, cap=8)


# ---------------------------------------------------------------------------
# Serialisers (ORM row -> frontend dict)
# ---------------------------------------------------------------------------

def _class_size(mc: Optional[models.MutationClass]) -> Optional[int]:
    """Labeled orbit size for closed classes; None (=> ∞) for open / unknown."""
    if mc is None or mc.is_open:
        return None
    return mc.labeled_size


def _distinct_quivers(labeled_quivers: list, canonical_qid: Optional[str]) -> list[dict]:
    """
    Collapse the full labeled orbit into one entry per distinct unlabeled
    quiver (Q.* id).

    The labeled orbit stores every vertex-labeling of every quiver in the
    class, so the same Q.* id appears many times.  Here we group by qmd_id,
    count the labelings, and expose each quiver's canonical form (so the
    figure matches its quiver page).  Exactly one entry — the class canonical
    representative — is flagged is_canonical.
    """
    groups: dict[str, dict] = {}
    order: list[str] = []
    for entry in labeled_quivers:
        qid = entry["qmd_id"]
        g = groups.get(qid)
        if g is None:
            groups[qid] = {
                "qmd_id": qid,
                "matrix": to_lists(canonical_form(to_matrix(entry["matrix"]))),
                "labeling_count": 1,
                "is_canonical": qid == canonical_qid,
            }
            order.append(qid)
        else:
            g["labeling_count"] += 1

    items = [groups[qid] for qid in order]
    # Canonical first, then most-labeled, then id for a stable order.
    items.sort(key=lambda d: (not d["is_canonical"], -d["labeling_count"], d["qmd_id"]))
    return items


def _quiver_list_item(q: models.Quiver, mc: Optional[models.MutationClass]) -> dict:
    return {
        "qmd_id": q.quiver_id,
        "num_vertices": q.n_vertices,
        "dynkin_type": mc.dynkin_type if mc else None,
        "representation_type": q.representation_type,
        "max_edge": q.max_edge,
        "is_acyclic": q.is_acyclic,
        "is_connected": q.is_connected,
        "is_bipartite": q.is_bipartite,
        "is_open": mc.is_open if mc else False,
        "class_size": _class_size(mc),
        "exchange_matrix": q.canonical_matrix,   # labeled rows override this per labeling
        "mc_id": q.mc_id,
    }


def _labeling_count(q: models.Quiver, mc: Optional[models.MutationClass]) -> int:
    """How many labeled matrices in the class map to this unlabeled quiver."""
    if mc is None or not mc.labeled_quivers:
        return 1
    c = sum(1 for e in mc.labeled_quivers if e.get("qmd_id") == q.quiver_id)
    return c if c else 1


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------

_SORT_COLUMNS = {
    "qmd_id": models.Quiver.quiver_id,
    "num_vertices": models.Quiver.n_vertices,
    "class_size": models.MutationClass.labeled_size,
    "max_edge": models.Quiver.max_edge,
}


def _filtered_quivers(
    db: Session,
    *,
    rank: Optional[int] = None,
    is_open: Optional[bool] = None,
    dynkin_type: Optional[str] = None,
    representation_type: Optional[str] = None,
    max_edge: Optional[int] = None,
    is_acyclic: Optional[bool] = None,
    is_connected: Optional[bool] = None,
    is_simply_laced: Optional[bool] = None,
    is_mutation_finite: Optional[bool] = None,
    orbit_min: Optional[int] = None,
    orbit_max: Optional[int] = None,
):
    """Build a (Quiver, MutationClass) query with the given filters applied."""
    Q, MC = models.Quiver, models.MutationClass
    query = db.query(Q, MC).outerjoin(MC, Q.mc_id == MC.mc_id)

    if rank is not None:
        query = query.filter(Q.n_vertices == rank)
    if max_edge is not None:
        query = query.filter(Q.max_edge == max_edge)
    if is_acyclic is not None:
        query = query.filter(Q.is_acyclic == is_acyclic)
    if is_connected is not None:
        query = query.filter(Q.is_connected == is_connected)
    if is_simply_laced is not None:
        query = query.filter(Q.max_edge <= 1) if is_simply_laced \
            else query.filter(Q.max_edge > 1)
    if is_open is not None:
        query = query.filter(MC.is_open == is_open)
    if is_mutation_finite is not None:
        query = query.filter(MC.is_open == (not is_mutation_finite))
    if dynkin_type is not None:
        query = query.filter(MC.dynkin_type == dynkin_type)
    if representation_type is not None:
        query = query.filter(Q.representation_type == representation_type)
    if orbit_min is not None:
        query = query.filter(MC.labeled_size >= orbit_min)
    if orbit_max is not None:
        query = query.filter(MC.labeled_size <= orbit_max)

    return query


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def list_quivers(
    db: Session,
    *,
    filters: dict,
    scope: str = "distinct",
    sort: str = "num_vertices",
    direction: str = "asc",
    offset: int = 0,
    limit: int = 50,
) -> tuple[list[dict], int, int, int]:
    """
    Return (items, total, distinct_total, labeled_total) for a filtered,
    sorted, paginated quiver listing.

    scope:
      "distinct"  — one row per unlabeled quiver (exchange_matrix = canonical).
      "labelings" — one row per labeled matrix in the quiver's class (same
                    qmd_id repeats; exchange_matrix is each labeling).

    Both totals are always returned so the UI can show either count (and the
    download dialog can update without a second request).
    """
    query = _filtered_quivers(db, **filters)
    col = _SORT_COLUMNS.get(sort, models.Quiver.n_vertices)
    col = col.desc() if direction == "desc" else col.asc()
    rows = query.order_by(col, models.Quiver.quiver_id).all()

    distinct_total = len(rows)
    labeled_total = sum(_labeling_count(q, mc) for q, mc in rows)

    if scope == "labelings":
        expanded: list[dict] = []
        for q, mc in rows:
            base = _quiver_list_item(q, mc)
            labs = ([e for e in (mc.labeled_quivers or [])
                     if e.get("qmd_id") == q.quiver_id] if mc else [])
            if labs:
                for e in labs:
                    item = dict(base)
                    item["exchange_matrix"] = e["matrix"]
                    expanded.append(item)
            else:
                expanded.append(base)
        items = expanded[offset:offset + limit]
        total = labeled_total
    else:
        items = [_quiver_list_item(q, mc) for q, mc in rows[offset:offset + limit]]
        total = distinct_total

    return items, total, distinct_total, labeled_total


def get_quiver_detail(db: Session, qid: str) -> Optional[dict]:
    q = db.get(models.Quiver, qid)
    if q is None:
        return None
    mc = db.get(models.MutationClass, q.mc_id) if q.mc_id else None
    return {
        "qmd_id": q.quiver_id,
        "label": mc.label if mc else None,
        "num_vertices": q.n_vertices,
        "exchange_matrix": q.canonical_matrix,
        "dynkin_type": mc.dynkin_type if mc else None,
        "is_open": mc.is_open if mc else False,
        "is_acyclic": q.is_acyclic,
        "is_connected": q.is_connected,
        "max_edge": q.max_edge,
        "is_bipartite": q.is_bipartite,
        "is_abundant": q.is_abundant,
        "is_planar": q.is_planar,
        "representation_type": q.representation_type,
        "symmetry_group": q.symmetry_group,
        "class_size": _class_size(mc),
        "mc_id": q.mc_id,
        "tags": [],
    }


def get_class_detail(db: Session, mc_id: str) -> Optional[dict]:
    mc = db.get(models.MutationClass, mc_id)
    if mc is None:
        return None
    return {
        "mc_id": mc.mc_id,
        "label": mc.label,
        "num_vertices": mc.n_vertices,
        "dynkin_type": mc.dynkin_type,
        "is_open": mc.is_open,
        "labeled_size": mc.labeled_size,
        "distinct_quiver_count": mc.distinct_quiver_count,
        "merged_orbit_count": mc.merged_orbit_count,
        "canonical_matrix": mc.canonical_rep,
        "canonical_qid": mc.canonical_qid,
        "distinct_quivers": _distinct_quivers(mc.labeled_quivers, mc.canonical_qid),
        "labeled_quivers": mc.labeled_quivers,   # [{qmd_id, matrix}, ...]
        "is_finite_confirmed": mc.is_finite_confirmed,
        "is_infinite_confirmed": mc.is_infinite_confirmed,
        "is_infinite_expected": mc.is_infinite_expected,
        "size_of_explored_mutation_class": mc.labeled_size,
        "size_of_explored_frontier": mc.size_of_explored_frontier,
        "is_mutation_acyclic": mc.is_mutation_acyclic,
        "is_banff": mc.is_banff,
        "is_louise": mc.is_louise,
        "is_p_prime": mc.is_p_prime,
        "is_locally_acyclic": mc.is_locally_acyclic,
        "provenance": mc.provenance,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

# Column order for dataset exports (CSV / Excel).  Quiver-level columns first,
# then the full mutation-class statistics.  Stable so downloads are diffable.
EXPORT_COLUMNS = [
    # --- quiver (per unlabeled quiver / per labeling) ---
    "qmd_id", "num_vertices", "exchange_matrix", "representation_type",
    "max_edge", "is_acyclic", "is_connected", "is_bipartite", "is_abundant",
    "is_planar", "symmetry_order", "symmetry_name",
    # --- mutation-class statistics ---
    "mc_id", "dynkin_type", "is_open", "class_size", "labeled_size",
    "distinct_quiver_count", "merged_orbit_count",
    "is_finite_confirmed", "is_infinite_confirmed", "is_infinite_expected",
    "size_of_explored_frontier", "is_mutation_acyclic", "is_locally_acyclic",
    "is_banff", "is_louise", "is_p_prime",
]


def _export_row(q: models.Quiver, mc: Optional[models.MutationClass]) -> dict:
    """Flat export dict for one quiver + its mutation-class statistics.

    exchange_matrix defaults to the quiver's canonical form; callers exporting
    individual labelings override it per labeling.
    """
    sym = q.symmetry_group or {}
    return {
        "qmd_id": q.quiver_id,
        "num_vertices": q.n_vertices,
        "exchange_matrix": json.dumps(q.canonical_matrix, separators=(",", ":")),
        "representation_type": q.representation_type,
        "max_edge": q.max_edge,
        "is_acyclic": q.is_acyclic,
        "is_connected": q.is_connected,
        "is_bipartite": q.is_bipartite,
        "is_abundant": q.is_abundant,
        "is_planar": q.is_planar,
        "symmetry_order": sym.get("order"),
        "symmetry_name": sym.get("name"),
        # --- mutation-class statistics ---
        "mc_id": q.mc_id,
        "dynkin_type": mc.dynkin_type if mc else None,
        "is_open": mc.is_open if mc else False,
        "class_size": _class_size(mc),
        "labeled_size": mc.labeled_size if mc else None,
        "distinct_quiver_count": mc.distinct_quiver_count if mc else None,
        "merged_orbit_count": mc.merged_orbit_count if mc else None,
        "is_finite_confirmed": mc.is_finite_confirmed if mc else None,
        "is_infinite_confirmed": mc.is_infinite_confirmed if mc else None,
        "is_infinite_expected": mc.is_infinite_expected if mc else None,
        "size_of_explored_frontier": mc.size_of_explored_frontier if mc else None,
        "is_mutation_acyclic": mc.is_mutation_acyclic if mc else None,
        "is_locally_acyclic": mc.is_locally_acyclic if mc else None,
        "is_banff": mc.is_banff if mc else None,
        "is_louise": mc.is_louise if mc else None,
        "is_p_prime": mc.is_p_prime if mc else None,
    }


def export_rows(
    db: Session,
    *,
    filters: dict,
    scope: str = "distinct",
    sort: str = "num_vertices",
    direction: str = "asc",
) -> list[dict]:
    """
    Materialise every quiver matching `filters` (no pagination) as flat,
    export-ready dicts keyed by EXPORT_COLUMNS.  The exchange matrix is
    serialised as compact JSON so it fits one cell.

    scope:
      "distinct"  — one row per unlabeled quiver (exchange_matrix = canonical).
      "labelings" — one row per labeled matrix in the quiver's mutation class
                    (the same qmd_id repeats, exchange_matrix is each labeling).
    """
    query = _filtered_quivers(db, **filters)
    col = _SORT_COLUMNS.get(sort, models.Quiver.n_vertices)
    col = col.desc() if direction == "desc" else col.asc()
    rows = query.order_by(col, models.Quiver.quiver_id).all()

    out: list[dict] = []
    for q, mc in rows:
        base = _export_row(q, mc)
        if scope == "labelings" and mc is not None:
            labelings = [e for e in (mc.labeled_quivers or [])
                         if e.get("qmd_id") == q.quiver_id]
            if labelings:
                for e in labelings:
                    row = dict(base)
                    row["exchange_matrix"] = json.dumps(e["matrix"], separators=(",", ":"))
                    out.append(row)
                continue
        out.append(base)
    return out


def log_download(
    db: Session,
    *,
    fmt: str,
    row_count: int,
    filters: dict,
    email: Optional[str] = None,
    name: Optional[str] = None,
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
    referer: Optional[str] = None,
) -> None:
    """Record one dataset export.  Best-effort: never block a download on a log failure."""
    db.add(models.Download(
        fmt=fmt,
        row_count=row_count,
        filters=filters or None,
        email=email or None,
        name=name or None,
        ip=ip,
        user_agent=user_agent,
        referer=referer,
    ))
    db.commit()


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def upsert_generation_result(db: Session, result: GenerationResult) -> None:
    """
    Write (or overwrite) all quivers and mutation classes from a
    GenerationResult.  Computes per-quiver invariants and persists each
    class's full labeled orbit and canonical quiver id.

    Finite (closed) classes are labeled with their Dynkin type via qmd.dynkin;
    open/affine classes are left null.
    """
    # Mutation classes first (quivers FK -> mutation_classes).
    for mc_id, mc_res in result.classes.items():
        labeled = [
            {"qmd_id": qid, "matrix": to_lists(m)}
            for m, qid in zip(mc_res.labeled_quivers, mc_res.quiver_ids)
        ]
        is_open = mc_res.is_open
        rep = mc_res.canonical_rep
        dtype = None if is_open else dynkin.classify(rep)

        bounds = _la_bounds(is_open)
        b_state, b_w = la.banff_status(rep, **bounds)
        l_state, l_w = la.louise_status(rep, **bounds)
        p_state, p_w = la.p_prime_status(rep, **bounds)
        provenance = {
            "is_banff":   {"state": b_state, "witness": b_w},
            "is_louise":  {"state": l_state, "witness": l_w},
            "is_p_prime": {"state": p_state, "witness": p_w},
        }
        if is_open:
            provenance["is_infinite_confirmed"] = {
                "method": "Derksen-Owen: a bounded mutation reached |b_ij| >= 3"
            }

        db.merge(models.MutationClass(
            mc_id                 = mc_id,
            n_vertices            = len(rep),
            canonical_rep         = to_lists(rep),
            is_open               = is_open,
            labeled_size          = mc_res.labeled_size,
            distinct_quiver_count = mc_res.distinct_quiver_count,
            merged_orbit_count    = mc_res.merged_orbit_count,
            boundary_quivers      = [to_lists(m) for m in mc_res.boundary_quivers],
            canonical_qid         = quiver_id(rep),
            labeled_quivers       = labeled,
            dynkin_type           = dtype,
            label                 = dtype,
            is_finite_confirmed   = not is_open,
            is_infinite_confirmed = is_open,
            is_infinite_expected  = False,
            size_of_explored_frontier = len(mc_res.boundary_quivers),
            is_mutation_acyclic   = invariants.class_is_mutation_acyclic(
                                        mc_res.labeled_quivers, is_open),
            is_banff              = _TRI[b_state],
            is_louise             = _TRI[l_state],
            is_p_prime            = _TRI[p_state],
            is_locally_acyclic    = True if b_state == "true" else None,
            provenance            = provenance,
        ))

    # Quivers.
    for qid, matrix in result.quivers.items():
        qi = invariants.quiver_invariants(matrix)
        db.merge(models.Quiver(
            quiver_id        = qid,
            n_vertices       = len(matrix),
            canonical_matrix = to_lists(matrix),
            max_edge         = max_edge(matrix),
            is_acyclic       = is_acyclic(matrix),
            is_connected     = is_connected(matrix),
            is_bipartite     = qi["is_bipartite"],
            is_abundant      = qi["is_abundant"],
            is_planar        = qi["is_planar"],
            representation_type = qi["representation_type"],
            symmetry_group   = qi["symmetry_group"],
            mc_id            = result.membership.get(qid),
        ))

    db.commit()
