from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from admo_persistence.config import load_settings
from admo_persistence.object_store import create_object_client
from airflow.exceptions import AirflowException
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.hitl import HITLBranchOperator, HITLOperator
from airflow.sdk import Param, dag, get_current_context, task
from airflow.task.trigger_rule import TriggerRule

PROJECT_ROOT = Path(os.getenv("ADMO_PROJECT_ROOT", Path.cwd()))


@task(task_id="create_development_dataset")
def create_development_dataset() -> None:
    params = get_current_context()["params"]
    version = params["version"]
    raw_key = f"raw/ademe/version={version}/source.parquet"
    clean_key = f"processed/ademe/version={version}/clean_100k.parquet"
    subprocess.run(
        [
            "uv",
            "run",
            "--package",
            "processing-ml",
            "python",
            "-m",
            "processing_ml.dev_dataset",
            "--raw-key",
            raw_key,
            "--clean-key",
            clean_key,
            "--n-rows",
            str(params["n_rows"]),
            "--clean-seed",
            str(params["clean_seed"]),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        env=os.environ.copy(),
    )


@task(
    task_id="select_training_dataset",
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
)
def select_training_dataset(selection: dict[str, object]) -> str:
    params = get_current_context()["params"]
    version = params["version"]
    chosen_options = selection.get("chosen_options")
    if not isinstance(chosen_options, list) or len(chosen_options) != 1:
        raise AirflowException(
            "Human dataset selection must contain exactly one option."
        )
    dataset_choice = chosen_options[0]
    if dataset_choice == "source":
        training_key = f"raw/ademe/version={version}/source.parquet"
    elif dataset_choice == "clean_100k":
        training_key = f"processed/ademe/version={version}/clean_100k.parquet"
    else:
        raise AirflowException(f"Unsupported training dataset: {dataset_choice}")

    settings = load_settings()
    create_object_client().head_object(
        Bucket=settings.object_storage_bucket,
        Key=training_key,
    )
    return training_key


@task(task_id="train_isolation_forest")
def train_model(training_key: str) -> str:
    params = get_current_context()["params"]
    if not training_key:
        raise AirflowException("Training dataset selection returned no object key.")

    version = params["version"]
    model_key = f"models/isolation_forest/version={version}/model.joblib"
    preprocessor_key = f"models/isolation_forest/version={version}/preprocessor.joblib"
    command = [
        "uv",
        "run",
        "--package",
        "processing-ml",
        "admo-train",
        "--training-key",
        training_key,
        "--model-key",
        model_key,
        "--preprocessor-key",
        preprocessor_key,
        "--version",
        version,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=os.environ.copy())
    return model_key


@dag(
    dag_id="training_pipeline",
    description="Human-controlled Isolation Forest training for an ADEME version.",
    schedule=None,
    start_date=datetime(  # noqa: DTZ001
        2026,
        1,
        1,
    ),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "admo",
        "retries": 0,
        "retry_delay": timedelta(minutes=1),
    },
    params={
        "version": Param(
            "",
            type="string",
            title="ADEME dataset version",
            description="Date version, for example 2026-08-31.",
            pattern=r"^\d{4}-\d{2}-\d{2}$",
        ),
        "n_rows": Param(
            100000,
            type="integer",
            minimum=1,
            title="Development dataset rows",
        ),
        "clean_seed": Param(
            42,
            type="integer",
            title="Development dataset seed",
        ),
    },
    tags=["admo", "training", "human-controlled"],
)
def training_pipeline():
    create_dev_decision = HITLBranchOperator(
        task_id="create_development_dataset_decision",
        subject="Create a development dataset?",
        body=(
            "Choose whether to create a clean development dataset for this "
            "ADEME version. The row count and seed come from the DAG parameters."
        ),
        options=["Create development dataset", "Use existing dataset"],
        options_mapping={
            "Create development dataset": "create_development_dataset",
            "Use existing dataset": "skip_development_dataset",
        },
    )
    create_dev = create_development_dataset()
    skip_dev = EmptyOperator(task_id="skip_development_dataset")
    dev_join = EmptyOperator(
        task_id="development_dataset_ready",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    create_dev_decision >> [create_dev, skip_dev]
    [create_dev, skip_dev] >> dev_join

    dataset_decision = HITLOperator(
        task_id="training_dataset_decision",
        subject="Select the training dataset",
        body="Choose the dataset artifact to use for Isolation Forest training.",
        options=["source", "clean_100k"],
        defaults=["clean_100k"],
    )
    select_dataset = select_training_dataset(dataset_decision.output)
    dev_join >> dataset_decision >> select_dataset
    train_model(select_dataset)


dag = training_pipeline()
