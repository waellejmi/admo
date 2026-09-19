import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    configuration: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="pipeline_run")


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("pipeline_runs.id"))
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    object_key: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(128))
    sha256: Mapped[str | None] = mapped_column(String(64))
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    artifact_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    pipeline_run: Mapped[PipelineRun | None] = relationship(back_populates="artifacts")


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    artifact_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("artifacts.id"), unique=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    row_count: Mapped[int | None] = mapped_column(Integer)
    source_dataset_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("datasets.id"))


class ModelMetadata(Base):
    __tablename__ = "models"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    artifact_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("artifacts.id"), unique=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(128), nullable=False)
    training_dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipeline_runs.id"), nullable=False)
    model_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("models.id"))
    evaluation_dataset_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("datasets.id"))
    configuration: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
