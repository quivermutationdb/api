"""drop is_locally_acyclic from mutation_classes

Revision ID: f1a3c5d7
Revises: e7f9b1c3
Create Date: 2026-06-24
"""

from alembic import op
import sqlalchemy as sa

revision = "f1a3c5d7"
down_revision = "e7f9b1c3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("mutation_classes", "is_locally_acyclic")


def downgrade() -> None:
    op.add_column(
        "mutation_classes",
        sa.Column("is_locally_acyclic", sa.Boolean(), nullable=True),
    )
