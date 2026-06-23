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
    class_size: Optional[int] = None
    mc_id: Optional[str] = None
    tags: list[str] = []


# ---------------------------------------------------------------------------
# Mutation class
# ---------------------------------------------------------------------------

class LabeledMember(BaseModel):
    qmd_id: str
    matrix: Matrix


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
    labeled_quivers: list[LabeledMember] = []


# ---------------------------------------------------------------------------
# List envelope
# ---------------------------------------------------------------------------

class QuiverListResponse(BaseModel):
    items: list[QuiverListItem]
    total: int
