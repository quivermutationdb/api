"""
qmd/models.py

SQLAlchemy ORM models.

Tables
------
mutation_classes
    One row per merged mutation class (MC.* id).

quivers
    One row per unlabeled quiver isomorphism class (Q.* id).
    Each quiver belongs to exactly one mutation class.
"""

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, String, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class MutationClass(Base):
    __tablename__ = "mutation_classes"

    mc_id                = Column(String, primary_key=True)
    n_vertices           = Column(Integer, nullable=False)
    canonical_rep        = Column(JSONB, nullable=False)
    is_open              = Column(Boolean, nullable=False)
    labeled_size         = Column(Integer, nullable=False)
    distinct_quiver_count = Column(Integer, nullable=False)
    merged_orbit_count   = Column(Integer, nullable=False, default=1)
    # boundary_quivers stored as a JSON array of matrices
    boundary_quivers     = Column(JSONB, nullable=False, default=list)
    # Q.* id of the canonical representative (quiver_id of canonical_rep)
    canonical_qid        = Column(String, nullable=True)
    # Full labeled orbit: JSON array of {"qmd_id": str, "matrix": [[int]]}
    labeled_quivers      = Column(JSONB, nullable=False, default=list)
    dynkin_type          = Column(String, nullable=True)
    label                = Column(String, nullable=True)

    # Finiteness trichotomy (mutually exclusive) + frontier size
    is_finite_confirmed  = Column(Boolean, nullable=True)
    is_infinite_confirmed = Column(Boolean, nullable=True)
    is_infinite_expected = Column(Boolean, nullable=True)
    size_of_explored_frontier = Column(Integer, nullable=True)

    # Three-state (true / false / null = unknown) class properties
    is_mutation_acyclic  = Column(Boolean, nullable=True)
    is_banff             = Column(Boolean, nullable=True)
    is_louise            = Column(Boolean, nullable=True)
    is_p_prime           = Column(Boolean, nullable=True)
    is_locally_acyclic   = Column(Boolean, nullable=True)

    # Per-property method / witness for the semidecidable properties
    provenance           = Column(JSONB, nullable=True)

    quivers = relationship("Quiver", back_populates="mutation_class",
                           cascade="all, delete-orphan")


class Quiver(Base):
    __tablename__ = "quivers"

    quiver_id        = Column(String, primary_key=True)
    n_vertices       = Column(Integer, nullable=False)
    canonical_matrix = Column(JSONB, nullable=False)
    # Computed invariants (stored so the API can filter on them)
    max_edge         = Column(Integer, nullable=False, default=0)
    is_acyclic       = Column(Boolean, nullable=False, default=True)
    is_connected     = Column(Boolean, nullable=False, default=True)
    is_bipartite     = Column(Boolean, nullable=True)
    is_abundant      = Column(Boolean, nullable=True)
    is_planar        = Column(Boolean, nullable=True)        # null = unknown (n > 4)
    representation_type = Column(String, nullable=True)      # 'finite'/'tame'/'wild'; null = n/a (cyclic)
    symmetry_group   = Column(JSONB, nullable=True)          # {order, name, generators}
    mc_id            = Column(String,
                              ForeignKey("mutation_classes.mc_id", ondelete="SET NULL"),
                              nullable=True, index=True)

    mutation_class = relationship("MutationClass", back_populates="quivers")


class Download(Base):
    """
    One row per dataset export.  Records what cut was downloaded, in what
    format, how many rows, and (optionally) who — so usage can be tracked
    without requiring site accounts.
    """
    __tablename__ = "downloads"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now(),
                         nullable=False, index=True)
    fmt         = Column(String, nullable=False)            # 'csv' | 'xlsx'
    row_count   = Column(Integer, nullable=False)
    filters     = Column(JSONB, nullable=True)              # the applied cut (non-null filters)
    email       = Column(String, nullable=True, index=True) # optional, self-reported
    name        = Column(String, nullable=True)             # optional name / affiliation
    ip          = Column(String, nullable=True)
    user_agent  = Column(String, nullable=True)
    referer     = Column(String, nullable=True)
