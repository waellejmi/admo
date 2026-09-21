import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Artifact, Dataset, ModelMetadata, PipelineRun
from .object_store import ArtifactRef


@dataclass(frozen=True)
class ModelArtifactRefs:
    model_object_key: str
    preprocessor_object_key: str
    version: str
    pipeline_run_id: uuid.UUID


def resolve_model_artifacts(
    session: Session,
    name: str,
    version: str | None = None,
) -> ModelArtifactRefs:
    query = (
        select(ModelMetadata, Artifact, PipelineRun)
        .join(Artifact, ModelMetadata.artifact_id == Artifact.id)
        .join(PipelineRun, Artifact.pipeline_run_id == PipelineRun.id)
        .where(ModelMetadata.name == name, PipelineRun.status == "completed")
    )
    if version is not None:
        query = query.where(ModelMetadata.version == version)
    query = query.order_by(PipelineRun.finished_at.desc().nullslast())
    result = session.execute(query).first()
    if result is None:
        detail = f"model {name!r}"
        if version is not None:
            detail += f" version {version!r}"
        raise LookupError(f"No completed {detail} is registered.")

    model, model_artifact, run = result
    preprocessor_key = session.scalar(
        select(Artifact.object_key).where(
            Artifact.pipeline_run_id == run.id,
            Artifact.kind == "preprocessor",
        )
    )
    if preprocessor_key is None:
        raise LookupError(
            f"No preprocessor artifact is registered for pipeline run {run.id}."
        )
    return ModelArtifactRefs(
        model_object_key=model_artifact.object_key,
        preprocessor_object_key=preprocessor_key,
        version=model.version,
        pipeline_run_id=run.id,
    )


def create_pipeline_run(
    session: Session,
    configuration: dict[str, Any],
) -> PipelineRun:
    run = PipelineRun(
        id=uuid.uuid4(),
        status="running",
        configuration=configuration,
        started_at=datetime.now(UTC),
    )
    session.add(run)
    session.flush()
    return run


def register_artifact(
    session: Session,
    run: PipelineRun,
    artifact: ArtifactRef,
    kind: str,
    body: bytes,
    metadata: dict[str, Any] | None = None,
) -> Artifact:
    digest = hashlib.sha256(body).hexdigest()
    existing = session.scalar(
        select(Artifact).where(Artifact.object_key == artifact.object_key)
    )
    if existing is not None:
        if existing.sha256 != digest:
            raise ValueError(
                f"Artifact key already exists with different content: "
                f"{artifact.object_key}"
            )
        return existing

    record = Artifact(
        id=uuid.uuid4(),
        pipeline_run_id=run.id,
        kind=kind,
        object_key=artifact.object_key,
        content_type=artifact.content_type,
        sha256=digest,
        size_bytes=len(body),
        artifact_metadata=metadata or {},
    )
    session.add(record)
    session.flush()
    return record


def register_dataset(
    session: Session,
    artifact: Artifact,
    name: str,
    row_count: int,
    schema_version: str = "1",
    source_dataset_id: uuid.UUID | None = None,
) -> Dataset:
    existing = session.scalar(
        select(Dataset).where(Dataset.artifact_id == artifact.id)
    )
    if existing is not None:
        if existing.name != name or existing.row_count != row_count:
            raise ValueError(
                f"Dataset metadata conflicts with existing artifact: "
                f"{artifact.object_key}"
            )
        return existing

    dataset = Dataset(
        id=uuid.uuid4(),
        artifact_id=artifact.id,
        name=name,
        schema_version=schema_version,
        row_count=row_count,
        source_dataset_id=source_dataset_id,
    )
    session.add(dataset)
    session.flush()
    return dataset


def register_model(
    session: Session,
    artifact: Artifact,
    name: str,
    version: str,
    training_dataset_id: uuid.UUID,
    parameters: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> ModelMetadata:
    model = ModelMetadata(
        id=uuid.uuid4(),
        artifact_id=artifact.id,
        name=name,
        version=version,
        training_dataset_id=training_dataset_id,
        parameters=parameters,
        metrics=metrics or {},
    )
    session.add(model)
    session.flush()
    return model
