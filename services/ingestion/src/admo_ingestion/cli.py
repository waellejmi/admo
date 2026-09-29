import argparse
import os
from io import BytesIO

import pyarrow.parquet as pq
from admo_persistence import get_object
from admo_persistence.database import session_scope
from admo_persistence.registry import (
    create_pipeline_run,
    register_artifact,
    register_dataset,
)

from .pipeline import ingest_csv_to_garage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a raw CSV object in Garage into a Parquet snapshot."
    )
    parser.add_argument("source_key")
    parser.add_argument("object_key")
    parser.add_argument("--separator", default=";")
    args = parser.parse_args()
    artifact_ref = ingest_csv_to_garage(
        args.source_key,
        args.object_key,
        sep=args.separator,
    )
    body = get_object(args.object_key)
    version = _extract_version(args.object_key)
    with session_scope() as session:
        run = create_pipeline_run(
            session,
            {
                "component": "ingestion",
                "operation": "ingest",
                "source": args.source_key,
                "object_key": args.object_key,
                "version": version,
                "updated_at": os.getenv("ADMO_ADEME_UPDATED_AT"),
            },
        )
        artifact = register_artifact(
            session,
            run,
            artifact_ref,
            kind="dataset",
            body=body,
            metadata={
                "dataset": "ademe",
                "version": version,
                "updated_at": os.getenv("ADMO_ADEME_UPDATED_AT"),
            },
        )
        register_dataset(
            session,
            artifact,
            name="ademe_raw",
            row_count=_count_rows(body),
        )
        run.status = "completed"


def _extract_version(object_key: str) -> str:
    marker = "version="
    if marker not in object_key:
        raise ValueError(
            "Raw object key must include a dataset version, "
            "for example raw/ademe/version=2026-08-31/source.parquet."
        )
    return object_key.split(marker, 1)[1].split("/", 1)[0]


def _count_rows(body: bytes) -> int:
    return pq.read_metadata(BytesIO(body)).num_rows
