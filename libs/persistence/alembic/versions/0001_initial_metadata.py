import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_initial_metadata"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    uuid_type = postgresql.UUID(as_uuid=True)
    json_type = postgresql.JSONB()
    op.create_table(
        "pipeline_runs",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column(
            "configuration",
            json_type,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("error_message", sa.Text()),
    )
    op.create_table(
        "artifacts",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("pipeline_run_id", uuid_type, sa.ForeignKey("pipeline_runs.id")),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("object_key", sa.String(512), unique=True, nullable=False),
        sa.Column("content_type", sa.String(128)),
        sa.Column("sha256", sa.String(64)),
        sa.Column("size_bytes", sa.Integer()),
        sa.Column(
            "metadata", json_type, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
    )
    op.create_table(
        "datasets",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "artifact_id",
            uuid_type,
            sa.ForeignKey("artifacts.id"),
            unique=True,
            nullable=False,
        ),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("row_count", sa.Integer()),
        sa.Column("source_dataset_id", uuid_type, sa.ForeignKey("datasets.id")),
    )
    op.create_table(
        "models",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "artifact_id",
            uuid_type,
            sa.ForeignKey("artifacts.id"),
            unique=True,
            nullable=False,
        ),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("version", sa.String(128), nullable=False),
        sa.Column(
            "training_dataset_id",
            uuid_type,
            sa.ForeignKey("datasets.id"),
            nullable=False,
        ),
        sa.Column(
            "parameters",
            json_type,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "metrics", json_type, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
    )
    op.create_table(
        "experiments",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "pipeline_run_id",
            uuid_type,
            sa.ForeignKey("pipeline_runs.id"),
            nullable=False,
        ),
        sa.Column("model_id", uuid_type, sa.ForeignKey("models.id")),
        sa.Column("evaluation_dataset_id", uuid_type, sa.ForeignKey("datasets.id")),
        sa.Column(
            "configuration",
            json_type,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "metrics", json_type, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
    )


def downgrade() -> None:
    op.drop_table("experiments")
    op.drop_table("models")
    op.drop_table("datasets")
    op.drop_table("artifacts")
    op.drop_table("pipeline_runs")
