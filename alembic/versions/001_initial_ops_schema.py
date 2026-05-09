"""Initial ops schema

Revision ID: 001
Revises:
Create Date: 2026-04-28
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS ops")
    op.create_table(
        "api_keys",
        sa.Column("key_hash", sa.Text, primary_key=True),
        sa.Column("user_id", sa.Text, nullable=False),
        sa.Column("role", sa.Text, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.TIMESTAMP(timezone=True)),
        schema="ops",
    )
    op.create_table(
        "jobs",
        sa.Column("job_id", sa.Text, primary_key=True),
        sa.Column("user_id", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("idempotency_key", sa.Text, unique=True),
        sa.Column("source_type", sa.Text, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("error_payload", JSONB),
        sa.Column("mlflow_run_id", sa.Text),
        schema="ops",
    )
    op.create_table(
        "init_runs",
        sa.Column("job_id", sa.Text, primary_key=True),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("source_path", sa.Text, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False),
        schema="ops",
    )
    op.create_table(
        "kb_nodes",
        sa.Column("node_id", sa.UUID, primary_key=True),
        sa.Column("schema_version", sa.Text, nullable=False, server_default="1.0"),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("source_quote", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("state", sa.Text, nullable=False, server_default="current"),
        sa.Column("chunk_index", sa.Integer, nullable=True),
        sa.Column("metadata", JSONB, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        schema="ops",
    )
    op.create_index("ix_kb_nodes_type", "kb_nodes", ["type"], schema="ops")
    op.create_index("ix_kb_nodes_state", "kb_nodes", ["state"], schema="ops")
    op.create_index(
        "ix_kb_nodes_metadata_job_id",
        "kb_nodes",
        [sa.text("(metadata->>'job_id')")],
        schema="ops",
    )
    op.create_index(
        "ix_kb_nodes_metadata_meeting_date",
        "kb_nodes",
        [sa.text("(metadata->>'meeting_date')")],
        schema="ops",
    )
    op.create_table(
        "kb_relationships",
        sa.Column("source_id", sa.UUID, sa.ForeignKey("ops.kb_nodes.node_id"), nullable=False),
        sa.Column("target_id", sa.UUID, sa.ForeignKey("ops.kb_nodes.node_id"), nullable=False),
        sa.Column("rel_type", sa.Text, nullable=False),
        sa.Column("job_id", sa.Text, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("source_id", "target_id", "rel_type"),
        schema="ops",
    )
    op.create_index("ix_kb_relationships_target_id", "kb_relationships", ["target_id"], schema="ops")


def downgrade() -> None:
    op.drop_index("ix_kb_relationships_target_id", table_name="kb_relationships", schema="ops")
    op.drop_table("kb_relationships", schema="ops")
    op.drop_index("ix_kb_nodes_metadata_meeting_date", table_name="kb_nodes", schema="ops")
    op.drop_index("ix_kb_nodes_metadata_job_id", table_name="kb_nodes", schema="ops")
    op.drop_index("ix_kb_nodes_state", table_name="kb_nodes", schema="ops")
    op.drop_index("ix_kb_nodes_type", table_name="kb_nodes", schema="ops")
    op.drop_table("kb_nodes", schema="ops")
    op.drop_table("init_runs", schema="ops")
    op.drop_table("jobs", schema="ops")
    op.drop_table("api_keys", schema="ops")
