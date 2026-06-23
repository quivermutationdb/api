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

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
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
    # null until a Dynkin/type classifier is built (Phase 3)
    dynkin_type          = Column(String, nullable=True)
    label                = Column(String, nullable=True)

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
    mc_id            = Column(String,
                              ForeignKey("mutation_classes.mc_id", ondelete="SET NULL"),
                              nullable=True, index=True)

    mutation_class = relationship("MutationClass", back_populates="quivers")
