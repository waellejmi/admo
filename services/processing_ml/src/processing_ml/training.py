import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from io import BytesIO
from uuid import UUID

import joblib
import mlflow
import pandas as pd
from admo_persistence import get_object, put_bytes
from admo_persistence.database import session_scope
from admo_persistence.models import Artifact, Dataset, PipelineRun
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
        pipeline_run = create_pipeline_run(
            session,
            {
                "component": "processing_ml",
                "operation": "train",
                "model": "isolation_forest",
                "random_seed": random_seed,
                "training_object_key": training_object_key,
                "model_object_key": model_object_key,
                "preprocessor_object_key": preprocessor_object_key,
            },
        )
        pipeline_run_id = pipeline_run.id

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "admo-production-training",
    )

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"isolation_forest-{version}") as mlflow_run:
            mlflow_run_id = mlflow_run.info.run_id
            mlflow.set_tags(
                {
                    "admo_pipeline_run_id": str(pipeline_run_id),
                    "admo_model_version": version,
                    "admo_model_name": "isolation_forest",
                    "admo_training_dataset_key": training_object_key,
                    "admo_model_key": model_object_key,
                    "admo_preprocessor_key": preprocessor_object_key,
                }
            )
            mlflow.log_params(
                {
                    "n_estimators": 200,
                    "max_samples": "auto",
                    "contamination": 0.05,
                    "random_seed": random_seed,
                    "dataset_version": version,
                }
            )

            with session_scope() as session:
                run = session.get(PipelineRun, pipeline_run_id)
                if run is not None:
                    run.configuration = {
                        **run.configuration,
                        "mlflow_experiment": experiment_name,
                        "mlflow_run_id": mlflow_run_id,
                    }

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

            mlflow.log_metrics(
                {
                    "training_seconds": training_seconds,
                    "training_rows": len(training),
                    "transformed_feature_count": train_values.shape[1],
                }
            )

            preprocessor_bytes = _dump_joblib(preprocessor)
            model_bytes = _dump_joblib(model)
            preprocessor_ref = put_bytes(
                preprocessor_bytes,
                preprocessor_object_key,
                "application/octet-stream",
            )
            model_ref = put_bytes(
                model_bytes,
                model_object_key,
                "application/octet-stream",
            )
            mlflow.log_text(
                json.dumps(
                    {
                        "pipeline_run_id": str(pipeline_run_id),
                        "mlflow_run_id": mlflow_run_id,
                        "model_key": model_object_key,
                        "preprocessor_key": preprocessor_object_key,
                        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
                        "preprocessor_sha256": hashlib.sha256(
                            preprocessor_bytes
                        ).hexdigest(),
                    },
                    indent=2,
                ),
                "admo_artifact_references.json",
            )

            with session_scope() as session:
                dataset = session.scalar(
                    select(Dataset)
                    .join(Artifact, Dataset.artifact_id == Artifact.id)
                    .where(Artifact.object_key == training_object_key)
                )
                run = session.get(PipelineRun, pipeline_run_id)
                if dataset is None or run is None:
                    raise LookupError(
                        "Training dataset or pipeline run disappeared during training."
                    )
                model_artifact = register_artifact(
                    session,
                    run,
                    model_ref,
                    kind="model",
                    body=model_bytes,
                    metadata={
                        "training_seconds": training_seconds,
                        "mlflow_run_id": mlflow_run_id,
                    },
                )
                register_artifact(
                    session,
                    run,
                    preprocessor_ref,
                    kind="preprocessor",
                    body=preprocessor_bytes,
                    metadata={
                        "features": list(training.columns),
                        "mlflow_run_id": mlflow_run_id,
                    },
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
                    metrics={
                        "training_seconds": training_seconds,
                        "mlflow_run_id": mlflow_run_id,
                    },
                )
                run.status = "completed"
                run.finished_at = datetime.now(UTC)
                return run.id
    except Exception as exc:
        with session_scope() as session:
            run = session.get(PipelineRun, pipeline_run_id)
            if run is not None:
                run.status = "failed"
                run.error_message = str(exc)
                run.finished_at = datetime.now(UTC)
        raise


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
