"""Add meeting_date and submission columns to ops.jobs

Revision ID: 004
Revises: 003
Create Date: 2026-06-28
"""

import sqlalchemy as sa
from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("meeting_date", sa.Date, nullable=False), schema="ops")
    op.add_column("jobs", sa.Column("submission", sa.JSON, nullable=False), schema="ops")
    op.add_column("jobs", sa.Column("raw_blob_key", sa.Text, nullable=False), schema="ops")


def downgrade() -> None:
    op.drop_column("jobs", "raw_blob_key", schema="ops")
    op.drop_column("jobs", "submission", schema="ops")
    op.drop_column("jobs", "meeting_date", schema="ops")
