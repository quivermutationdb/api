"""
qmd/schemas.py

Pydantic schemas for the public API.

Field names match the *frontend's* expected contract (qmd_id, num_vertices,
exchange_matrix, class_size, ...), which deliberately differs from the
internal ORM column names (quiver_id, n_vertices, canonical_matrix, ...).
The CRUD layer is responsible for translating ORM rows into these shapes.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

Matrix = list[list[int]]


# ---------------------------------------------------------------------------
# Quiver
# ---------------------------------------------------------------------------

class QuiverListItem(BaseModel):
    """One row in the Browse / Search tables."""
    qmd_id: str
    num_vertices: int
    dynkin_type: Optional[str] = None
    representation_type: Optional[str] = None
    max_edge: Optional[int] = None
    is_acyclic: Optional[bool] = None
    is_connected: Optional[bool] = None
    is_bipartite: Optional[bool] = None
    is_open: bool
    class_size: Optional[int] = None       # None => unbounded (rendered as ∞)
    mc_id: Optional[str] = None


class QuiverDetail(BaseModel):
    """The full quiver page."""
    qmd_id: str
    label: Optional[str] = None
    num_vertices: int
    exchange_matrix: Matrix
    dynkin_type: Optional[str] = None
    is_open: bool
    is_acyclic: bool
    is_connected: bool
    max_edge: int
    is_bipartite: Optional[bool] = None
    is_abundant: Optional[bool] = None
    is_planar: Optional[bool] = None
    representation_type: Optional[str] = None
    symmetry_group: Optional[dict] = None
    class_size: Optional[int] = None
    mc_id: Optional[str] = None
    tags: list[str] = []


# ---------------------------------------------------------------------------
# Mutation class
# ---------------------------------------------------------------------------

class LabeledMember(BaseModel):
    qmd_id: str
    matrix: Matrix


class DistinctQuiver(BaseModel):
    """One distinct unlabeled quiver in the class (collapses all its labelings)."""
    qmd_id: str
    matrix: Matrix              # the quiver's canonical form (matches the quiver page)
    labeling_count: int         # how many labeled orbit matrices map to this quiver
    is_canonical: bool = False  # true for exactly one: the class canonical representative


class ClassDetail(BaseModel):
    """The full mutation-class page."""
    mc_id: str
    label: Optional[str] = None
    num_vertices: int
    dynkin_type: Optional[str] = None
    is_open: bool
    labeled_size: int
    distinct_quiver_count: int
    merged_orbit_count: int
    canonical_matrix: Matrix
    canonical_qid: Optional[str] = None
    distinct_quivers: list[DistinctQuiver] = []
    labeled_quivers: list[LabeledMember] = []
    is_finite_confirmed: Optional[bool] = None
    is_infinite_confirmed: Optional[bool] = None
    is_infinite_expected: Optional[bool] = None
    size_of_explored_mutation_class: Optional[int] = None
    size_of_explored_frontier: Optional[int] = None
    is_mutation_acyclic: Optional[bool] = None
    is_banff: Optional[bool] = None
    is_louise: Optional[bool] = None
    is_p_prime: Optional[bool] = None
    is_locally_acyclic: Optional[bool] = None
    provenance: Optional[dict] = None


# ---------------------------------------------------------------------------
# List envelope
# ---------------------------------------------------------------------------

class QuiverListResponse(BaseModel):
    items: list[QuiverListItem]
    total: int
