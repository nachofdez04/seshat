"""Add publish_results table

Revision ID: 005
Revises: 004
Create Date: 2026-07-22
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Append-only audit trail: one row per publish, latest wins on retrieval.
    op.create_table(
        "publish_results",
        sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column("job_id", sa.Text, nullable=False),
        sa.Column("branch", sa.Text, nullable=False),
        sa.Column("commit_sha", sa.Text, nullable=False),
        sa.Column("pr_url", sa.Text, nullable=False, server_default=""),
        sa.Column("compare_url", sa.Text, nullable=False, server_default=""),
        sa.Column("files", postgresql.ARRAY(sa.Text), nullable=False),
        sa.Column("published_at", sa.TIMESTAMP(timezone=True), nullable=False),
        schema="ops",
    )
    op.create_index("ix_publish_results_job_id", "publish_results", ["job_id"], schema="ops")


def downgrade() -> None:
    op.drop_index("ix_publish_results_job_id", table_name="publish_results", schema="ops")
    op.drop_table("publish_results", schema="ops")
