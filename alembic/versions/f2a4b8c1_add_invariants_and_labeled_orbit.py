"""add quiver invariants and class labeled orbit

Revision ID: f2a4b8c1
Revises: 69b1dcec
Create Date: 2026-06-23
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "f2a4b8c1"
down_revision = "69b1dcec"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # quivers: computed invariants (stored so the API can filter on them)
    op.add_column("quivers",
        sa.Column("max_edge", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("quivers",
        sa.Column("is_acyclic", sa.Boolean(), nullable=False, server_default=sa.text("true")))
    op.add_column("quivers",
        sa.Column("is_connected", sa.Boolean(), nullable=False, server_default=sa.text("true")))
    op.create_index("ix_quivers_n_vertices", "quivers", ["n_vertices"])
    op.create_index("ix_quivers_max_edge", "quivers", ["max_edge"])

    # mutation_classes: canonical quiver id + full labeled orbit
    op.add_column("mutation_classes",
        sa.Column("canonical_qid", sa.String(), nullable=True))
    op.add_column("mutation_classes",
        sa.Column("labeled_quivers", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")))
    # dynkin_type / label are null until a classifier is built (Phase 3);
    # added now so search can filter on dynkin_type correctly.
    op.add_column("mutation_classes",
        sa.Column("dynkin_type", sa.String(), nullable=True))
    op.add_column("mutation_classes",
        sa.Column("label", sa.String(), nullable=True))
    op.create_index("ix_mutation_classes_n_vertices", "mutation_classes", ["n_vertices"])
    op.create_index("ix_mutation_classes_is_open", "mutation_classes", ["is_open"])


def downgrade() -> None:
    op.drop_index("ix_mutation_classes_is_open", table_name="mutation_classes")
    op.drop_index("ix_mutation_classes_n_vertices", table_name="mutation_classes")
    op.drop_column("mutation_classes", "label")
    op.drop_column("mutation_classes", "dynkin_type")
    op.drop_column("mutation_classes", "labeled_quivers")
    op.drop_column("mutation_classes", "canonical_qid")
    op.drop_index("ix_quivers_max_edge", table_name="quivers")
    op.drop_index("ix_quivers_n_vertices", table_name="quivers")
    op.drop_column("quivers", "is_connected")
    op.drop_column("quivers", "is_acyclic")
    op.drop_column("quivers", "max_edge")
