"""add labeling_count to quivers

Revision ID: e7f9b1c3
Revises: d4e6f8a2
Create Date: 2026-06-24
"""

from alembic import op
import sqlalchemy as sa

revision = "e7f9b1c3"
down_revision = "d4e6f8a2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("quivers", sa.Column("labeling_count", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("quivers", "labeling_count")
