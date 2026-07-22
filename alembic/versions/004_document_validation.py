"""Add validation columns to generated_documents

Revision ID: 004
Revises: 003
Create Date: 2026-07-22
"""

import sqlalchemy as sa
from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None

_VALIDATION_COLUMNS = [
    "validation_status",
    "validation_revision",
    "edited_content",
    "rejection_reason",
    "validated_by",
    "validated_at",
    "auto_approved",
    "approved_revision",
]


def upgrade() -> None:
    op.add_column(
        "generated_documents",
        sa.Column("validation_status", sa.Text, nullable=False, server_default="pending"),
        schema="ops",
    )
    op.add_column(
        "generated_documents",
        sa.Column("validation_revision", sa.Integer, nullable=False, server_default="0"),
        schema="ops",
    )
    op.add_column("generated_documents", sa.Column("edited_content", sa.Text, nullable=True), schema="ops")
    op.add_column("generated_documents", sa.Column("rejection_reason", sa.Text, nullable=True), schema="ops")
    op.add_column("generated_documents", sa.Column("validated_by", sa.Text, nullable=True), schema="ops")
    op.add_column(
        "generated_documents",
        sa.Column("validated_at", sa.TIMESTAMP(timezone=True), nullable=True),
        schema="ops",
    )
    op.add_column(
        "generated_documents",
        sa.Column("auto_approved", sa.Boolean, nullable=False, server_default=sa.false()),
        schema="ops",
    )
    op.add_column("generated_documents", sa.Column("approved_revision", sa.Text, nullable=True), schema="ops")


def downgrade() -> None:
    for column in reversed(_VALIDATION_COLUMNS):
        op.drop_column("generated_documents", column, schema="ops")
