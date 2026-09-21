from io import BytesIO
from typing import Any

import pandas as pd
from admo_persistence import ArtifactRef, get_object, put_bytes
from admo_persistence.models import Dataset, PipelineRun
from admo_persistence.registry import register_artifact, register_dataset
from sqlalchemy.orm import Session


def read_parquet_artifact(artifact: ArtifactRef) -> pd.DataFrame:
    return pd.read_parquet(BytesIO(get_object(artifact.object_key)))


def publish_parquet(
    frame: pd.DataFrame,
    object_key: str,
    content_type: str = "application/vnd.apache.parquet",
) -> tuple[ArtifactRef, bytes]:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    body = buffer.getvalue()
    return put_bytes(body, object_key, content_type), body


def publish_dataset(
    session: Session,
    run: PipelineRun,
    frame: pd.DataFrame,
    object_key: str,
    name: str,
    source_dataset_id: Any = None,
) -> Dataset:
    artifact_ref, body = publish_parquet(frame, object_key)
    artifact = register_artifact(
        session,
        run,
        artifact_ref,
        kind="dataset",
        body=body,
        metadata={"columns": list(frame.columns)},
    )
    return register_dataset(
        session,
        artifact,
        name=name,
        row_count=len(frame),
        source_dataset_id=source_dataset_id,
    )
