"""Add generated_documents table

Revision ID: 003
Revises: 002
Create Date: 2026-07-22
"""

import sqlalchemy as sa
from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "generated_documents",
        sa.Column("id", sa.UUID, primary_key=True),
        sa.Column("job_id", sa.Text, nullable=False),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("filename", sa.Text, nullable=False),
        sa.Column("markdown_content", sa.Text, nullable=False),
        sa.Column("content_revision", sa.Text, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.UniqueConstraint("job_id", "kind", name="uq_generated_documents_job_kind"),
        schema="ops",
    )
    op.create_index("ix_generated_documents_job_id", "generated_documents", ["job_id"], schema="ops")


def downgrade() -> None:
    op.drop_index("ix_generated_documents_job_id", table_name="generated_documents", schema="ops")
    op.drop_table("generated_documents", schema="ops")
