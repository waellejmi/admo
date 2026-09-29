from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.exceptions import AirflowException
from airflow.providers.standard.operators.hitl import (
    HITLBranchOperator,
    HITLEntryOperator,
)
from airflow.sdk import Param, dag, get_current_context, task
from airflow.task.trigger_rule import TriggerRule

from .lib.workload_launcher import Workload, run_workload

PROJECT_ROOT = Path(os.getenv("ADMO_PROJECT_ROOT", Path.cwd()))


@task(task_id="create_anomaly_evaluation_dataset")
def create_anomaly_evaluation_dataset(configuration: dict[str, object]) -> str:
    params = get_current_context()["params"]
    inputs = configuration.get("params_input")
    if not isinstance(inputs, dict):
        raise AirflowException("Anomaly configuration did not include form inputs.")

    weights = {
        anomaly_type: float(inputs[f"{anomaly_type}_weight"])
        for anomaly_type in (
            "contextual",
            "correlational",
            "structural",
            "energy_mix",
            "combination",
        )
    }
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise AirflowException("Anomaly distribution weights must sum to 1.")

    clean_key = params["clean_key"]
    evaluation_key = params["evaluation_key"]
    distribution = json.dumps(weights, sort_keys=True)
    run_workload(
        Workload(
            name="inject-anomaly-evaluation-dataset",
            image="admo-processing-ml:dev",
            command=(
                "admo-inject-anomalies",
                "--clean-key",
                clean_key,
                "--evaluation-key",
                evaluation_key,
                "--anomaly-rate",
                str(inputs["anomaly_rate"]),
                "--anomaly-seed",
                str(inputs["anomaly_seed"]),
                "--anomaly-distribution",
                distribution,
            ),
            host_command=(
                "uv",
                "run",
                "--package",
                "processing-ml",
                "admo-inject-anomalies",
                "--clean-key",
                clean_key,
                "--evaluation-key",
                evaluation_key,
                "--anomaly-rate",
                str(inputs["anomaly_rate"]),
                "--anomaly-seed",
                str(inputs["anomaly_seed"]),
                "--anomaly-distribution",
                distribution,
            ),
            environment={},
            cwd=PROJECT_ROOT,
        )
    )
    return evaluation_key


@task(task_id="use_existing_anomaly_evaluation_dataset")
def use_existing_anomaly_evaluation_dataset() -> str:
    evaluation_key = get_current_context()["params"]["evaluation_key"]
    if not evaluation_key:
        raise AirflowException(
            "evaluation_key is required when reusing an existing dataset."
        )
    return evaluation_key


@task(
    task_id="evaluation_dataset_ready",
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
)
def resolve_evaluation_dataset_key() -> str:
    task_instance = get_current_context()["ti"]
    evaluation_key = task_instance.xcom_pull(
        task_ids=[
            "create_anomaly_evaluation_dataset",
            "use_existing_anomaly_evaluation_dataset",
        ]
    )
    if isinstance(evaluation_key, list):
        evaluation_key = next(
            (value for value in evaluation_key if isinstance(value, str) and value),
            None,
        )
    if not isinstance(evaluation_key, str) or not evaluation_key:
        raise AirflowException(
            "Neither evaluation dataset branch produced an evaluation key."
        )
    return evaluation_key


@task(task_id="evaluate_persisted_model")
def evaluate_persisted_model(evaluation_key: str) -> None:
    params = get_current_context()["params"]
    version = params["version"]
    workload_command = (
        "admo-evaluate",
        "--model-key",
        f"models/isolation_forest/version={version}/model.joblib",
        "--preprocessor-key",
        f"models/isolation_forest/version={version}/preprocessor.joblib",
        "--evaluation-key",
        evaluation_key,
        "--version",
        version,
        "--k",
        str(params["evaluation_k"]),
    )
    run_workload(
        Workload(
            name=f"evaluate-isolation-forest-{version}",
            image="admo-processing-ml:dev",
            command=workload_command,
            host_command=(
                "uv",
                "run",
                "--package",
                "processing-ml",
                *workload_command,
            ),
            environment={},
            cwd=PROJECT_ROOT,
        )
    )


@dag(
    dag_id="evaluation_pipeline",
    description="Human-controlled evaluation of persisted Isolation Forest artifacts.",
    schedule=None,
    start_date=datetime(2026, 1, 1),  # noqa: DTZ001
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
            title="Model dataset version",
            pattern=r"^\d{4}-\d{2}-\d{2}$",
        ),
        "clean_key": Param(
            "",
            type="string",
            title="Clean evaluation source key",
        ),
        "evaluation_key": Param(
            "",
            type="string",
            title="Injected evaluation dataset key",
        ),
        "evaluation_k": Param(
            100,
            type="integer",
            minimum=1,
            title="Alert budget k",
        ),
    },
    tags=["admo", "evaluation", "human-controlled"],
)
def evaluation_pipeline():
    decision = HITLBranchOperator(
        task_id="create_anomaly_evaluation_dataset_decision",
        subject="Create an anomaly evaluation dataset?",
        body=(
            "Choose Create to inject labeled anomalies into clean_key, or "
            "Reuse to evaluate the existing evaluation_key."
        ),
        options=["Create", "Reuse existing"],
        options_mapping={
            "Create": "configure_anomaly_injection",
            "Reuse existing": "use_existing_anomaly_evaluation_dataset",
        },
    )
    configuration = HITLEntryOperator(
        task_id="configure_anomaly_injection",
        subject="Configure anomaly injection",
        body="Set the anomaly rate, seed, and distribution weights. Weights must sum to 1.",
        params={
            "anomaly_rate": Param(0.05, type="number", minimum=0.0, maximum=1.0),
            "anomaly_seed": Param(42, type="integer"),
            "contextual_weight": Param(0.20, type="number", minimum=0.0),
            "correlational_weight": Param(0.25, type="number", minimum=0.0),
            "structural_weight": Param(0.20, type="number", minimum=0.0),
            "energy_mix_weight": Param(0.25, type="number", minimum=0.0),
            "combination_weight": Param(0.10, type="number", minimum=0.0),
        },
    )
    create = create_anomaly_evaluation_dataset(configuration.output)
    reuse = use_existing_anomaly_evaluation_dataset()
    join = resolve_evaluation_dataset_key()
    decision >> [configuration, reuse]
    configuration >> create
    [create, reuse] >> join
    evaluate_persisted_model(join)


dag = evaluation_pipeline()
