"""add downloads tracking table

Revision ID: d4e6f8a2
Revises: c3d5e7f9
Create Date: 2026-06-24
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "d4e6f8a2"
down_revision = "c3d5e7f9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "downloads",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("fmt", sa.String(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("filters", JSONB(), nullable=True),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("ip", sa.String(), nullable=True),
        sa.Column("user_agent", sa.String(), nullable=True),
        sa.Column("referer", sa.String(), nullable=True),
    )
    op.create_index("ix_downloads_created_at", "downloads", ["created_at"])
    op.create_index("ix_downloads_email", "downloads", ["email"])


def downgrade() -> None:
    op.drop_index("ix_downloads_email", table_name="downloads")
    op.drop_index("ix_downloads_created_at", table_name="downloads")
    op.drop_table("downloads")
