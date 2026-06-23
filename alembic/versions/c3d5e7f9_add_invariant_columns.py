"""add invariant columns (quiver + class, three-state)

Revision ID: c3d5e7f9
Revises: f2a4b8c1
Create Date: 2026-06-23
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "c3d5e7f9"
down_revision = "f2a4b8c1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # quivers
    op.add_column("quivers", sa.Column("is_bipartite", sa.Boolean(), nullable=True))
    op.add_column("quivers", sa.Column("is_abundant", sa.Boolean(), nullable=True))
    op.add_column("quivers", sa.Column("is_planar", sa.Boolean(), nullable=True))
    op.add_column("quivers", sa.Column("representation_type", sa.String(), nullable=True))
    op.add_column("quivers", sa.Column("symmetry_group", JSONB(), nullable=True))
    op.create_index("ix_quivers_representation_type", "quivers", ["representation_type"])

    # mutation_classes
    op.add_column("mutation_classes", sa.Column("is_finite_confirmed", sa.Boolean(), nullable=True))
    op.add_column("mutation_classes", sa.Column("is_infinite_confirmed", sa.Boolean(), nullable=True))
    op.add_column("mutation_classes", sa.Column("is_infinite_expected", sa.Boolean(), nullable=True))
    op.add_column("mutation_classes", sa.Column("size_of_explored_frontier", sa.Integer(), nullable=True))
    op.add_column("mutation_classes", sa.Column("is_mutation_acyclic", sa.Boolean(), nullable=True))
    op.add_column("mutation_classes", sa.Column("is_banff", sa.Boolean(), nullable=True))
    op.add_column("mutation_classes", sa.Column("is_louise", sa.Boolean(), nullable=True))
    op.add_column("mutation_classes", sa.Column("is_p_prime", sa.Boolean(), nullable=True))
    op.add_column("mutation_classes", sa.Column("is_locally_acyclic", sa.Boolean(), nullable=True))
    op.add_column("mutation_classes", sa.Column("provenance", JSONB(), nullable=True))
    op.create_index("ix_mutation_classes_is_banff", "mutation_classes", ["is_banff"])
    op.create_index("ix_mutation_classes_is_finite_confirmed", "mutation_classes", ["is_finite_confirmed"])


def downgrade() -> None:
    op.drop_index("ix_mutation_classes_is_finite_confirmed", table_name="mutation_classes")
    op.drop_index("ix_mutation_classes_is_banff", table_name="mutation_classes")
    for col in ["provenance", "is_locally_acyclic", "is_p_prime", "is_louise", "is_banff",
                "is_mutation_acyclic", "size_of_explored_frontier", "is_infinite_expected",
                "is_infinite_confirmed", "is_finite_confirmed"]:
        op.drop_column("mutation_classes", col)
    op.drop_index("ix_quivers_representation_type", table_name="quivers")
    for col in ["symmetry_group", "representation_type", "is_planar", "is_abundant", "is_bipartite"]:
        op.drop_column("quivers", col)
