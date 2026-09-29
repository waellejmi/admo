from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import requests
from airflow.exceptions import AirflowException
from airflow.sdk import Variable, dag, task

# pyrefly: ignore [missing-import]
from lib.workload_launcher import Workload, run_workload

ADEME_METADATA_URL = (
    "https://data.ademe.fr/data-fair/api/v1/datasets/operat03-ratio-conso-ajustee/"
)
ADEME_RAW_URL = (
    "https://data.ademe.fr/data-fair/api/v1/datasets/operat03-ratio-conso-ajustee/raw"
)
LAST_VERSION_VARIABLE = "admo_last_ademe_version"
PROJECT_ROOT = Path(os.getenv("ADMO_PROJECT_ROOT", Path.cwd()))
RAW_CSV_PATH = PROJECT_ROOT / "data/raw/OPERAT03_RATIO_CONSO_AJUSTEE.csv"


@task(task_id="check_ademe_metadata")
def check_ademe_metadata() -> dict[str, str]:
    response = requests.get(ADEME_METADATA_URL, timeout=60)
    response.raise_for_status()
    payload = response.json()
    updated_at = payload.get("updatedAt")
    if not isinstance(updated_at, str) or not updated_at:
        raise AirflowException("ADEME metadata response has no valid updatedAt field.")
    return {"updated_at": updated_at, "version": updated_at[:10]}


@task.short_circuit(task_id="detect_new_version", ignore_downstream_trigger_rules=False)
def detect_new_version(metadata: dict[str, str]) -> dict[str, str] | bool:
    current_version = metadata["version"]
    previous_version = Variable.get(
        LAST_VERSION_VARIABLE,
        default="",
    )
    return metadata if current_version != previous_version else False


# TODO: Deal with docker permsiion to allow the continair to download  data with airflow user
@task(task_id="download_raw")
def download_raw(metadata: dict[str, str]) -> str:
    version = metadata["version"]
    if not version:
        raise AirflowException("No dataset version available for raw download.")
    RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(ADEME_RAW_URL, stream=True, timeout=(60, 600)) as response:
        response.raise_for_status()
        with RAW_CSV_PATH.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    return str(RAW_CSV_PATH)


@task(task_id="convert_to_parquet_and_store")
def convert_to_parquet_and_store(metadata: dict[str, str], csv_path: str) -> str:
    version = metadata["version"]
    updated_at = metadata["updated_at"]

    object_key = f"raw/ademe/version={version}/source.parquet"
    run_workload(
        Workload(
            name=f"ingest-ademe-{version}",
            image="admo-ingestion:dev",
            command=(csv_path, object_key),
            host_command=(
                "uv",
                "run",
                "--package",
                "ingestion",
                "admo-ingest",
                csv_path,
                object_key,
            ),
            environment={"ADMO_ADEME_UPDATED_AT": updated_at or ""},
            cwd=PROJECT_ROOT,
        )
    )
    Variable.set(LAST_VERSION_VARIABLE, version)
    return object_key


@dag(
    dag_id="data_pipeline",
    description="Check, download, and store new ADEME dataset versions.",
    schedule="0 3 1 * *",
    start_date=datetime(year=2026, month=8, day=1),  # noqa: DTZ001
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "admo",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["admo", "data", "ademe"],
)
def data_pipeline():
    metadata = check_ademe_metadata()
    new_metadata = detect_new_version(metadata)
    csv_path = download_raw(new_metadata)
    convert_to_parquet_and_store(new_metadata, csv_path)


dag = data_pipeline()
