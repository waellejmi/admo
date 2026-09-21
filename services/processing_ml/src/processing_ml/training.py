import argparse
import time
from datetime import UTC, datetime
from io import BytesIO
from uuid import UUID

import joblib
import pandas as pd
from admo_persistence import get_object, put_bytes
from admo_persistence.database import session_scope
from admo_persistence.models import Artifact, Dataset
from admo_persistence.registry import (
    create_pipeline_run,
    register_artifact,
    register_model,
)
from sklearn.ensemble import IsolationForest
from sqlalchemy import select

from .model_features import build_preprocessor, select_features


def train_isolation_forest(
    training_object_key: str,
    model_object_key: str,
    preprocessor_object_key: str,
    version: str,
    random_seed: int = 42,
) -> UUID:
    """Train and register the production Isolation Forest from a Garage dataset."""
    training = select_features(
        pd.read_parquet(BytesIO(get_object(training_object_key)))
    )
    preprocessor = build_preprocessor()
    train_values = preprocessor.fit_transform(training)
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.05,
        random_state=random_seed,
        n_jobs=-1,
    )
    started = time.perf_counter()
    model.fit(train_values)
    training_seconds = time.perf_counter() - started

    preprocessor_bytes = _dump_joblib(preprocessor)
    model_bytes = _dump_joblib(model)
    preprocessor_ref = put_bytes(
        preprocessor_bytes,
        preprocessor_object_key,
        "application/octet-stream",
    )
    model_ref = put_bytes(model_bytes, model_object_key, "application/octet-stream")

    with session_scope() as session:
        dataset = session.scalar(
            select(Dataset)
            .join(Artifact, Dataset.artifact_id == Artifact.id)
            .where(Artifact.object_key == training_object_key)
        )
        if dataset is None:
            raise LookupError(
                f"Training dataset is not registered: {training_object_key}"
            )

        run = create_pipeline_run(
            session,
            {
                "component": "processing_ml",
                "operation": "train",
                "model": "isolation_forest",
                "random_seed": random_seed,
                "training_object_key": training_object_key,
            },
        )
        model_artifact = register_artifact(
            session,
            run,
            model_ref,
            kind="model",
            body=model_bytes,
            metadata={"training_seconds": training_seconds},
        )
        register_artifact(
            session,
            run,
            preprocessor_ref,
            kind="preprocessor",
            body=preprocessor_bytes,
            metadata={"features": list(training.columns)},
        )
        register_model(
            session,
            model_artifact,
            name="isolation_forest",
            version=version,
            training_dataset_id=dataset.id,
            parameters={
                "n_estimators": 200,
                "max_samples": "auto",
                "contamination": 0.05,
                "random_state": random_seed,
            },
            metrics={"training_seconds": training_seconds},
        )
        run.status = "completed"
        run.finished_at = datetime.now(UTC)
        return run.id


def _dump_joblib(value: object) -> bytes:
    buffer = BytesIO()
    joblib.dump(value, buffer)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train production Isolation Forest.")
    parser.add_argument("--training-key", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--preprocessor-key", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    train_isolation_forest(
        training_object_key=args.training_key,
        model_object_key=args.model_key,
        preprocessor_object_key=args.preprocessor_key,
        version=args.version,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
