"""create quivers and mutation_classes tables

Revision ID: 69b1dcec
Revises:
Create Date: 2026-03-11
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "69b1dcec"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "mutation_classes",
        sa.Column("mc_id", sa.String(), primary_key=True, nullable=False),
        sa.Column("n_vertices", sa.Integer(), nullable=False),
        sa.Column("canonical_rep", JSONB(), nullable=False),
        sa.Column("is_open", sa.Boolean(), nullable=False),
        sa.Column("labeled_size", sa.Integer(), nullable=False),
        sa.Column("distinct_quiver_count", sa.Integer(), nullable=False),
        sa.Column("merged_orbit_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("boundary_quivers", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
    )

    op.create_table(
        "quivers",
        sa.Column("quiver_id", sa.String(), primary_key=True, nullable=False),
        sa.Column("n_vertices", sa.Integer(), nullable=False),
        sa.Column("canonical_matrix", JSONB(), nullable=False),
        sa.Column(
            "mc_id",
            sa.String(),
            sa.ForeignKey("mutation_classes.mc_id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_quivers_mc_id", "quivers", ["mc_id"])


def downgrade() -> None:
    op.drop_index("ix_quivers_mc_id", table_name="quivers")
    op.drop_table("quivers")
    op.drop_table("mutation_classes")
