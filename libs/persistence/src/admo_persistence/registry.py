import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from .models import Artifact, Dataset, ModelMetadata, PipelineRun
from .object_store import ArtifactRef


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
    record = Artifact(
        id=uuid.uuid4(),
        pipeline_run_id=run.id,
        kind=kind,
        object_key=artifact.object_key,
        content_type=artifact.content_type,
        sha256=hashlib.sha256(body).hexdigest(),
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
