from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from airflow.exceptions import AirflowException


@dataclass(frozen=True)
class Workload:
    name: str
    image: str
    command: tuple[str, ...]
    host_command: tuple[str, ...]
    environment: dict[str, str]
    cwd: Path | None = None


def _backend() -> str:
    backend = os.getenv("ADMO_EXECUTION_BACKEND", "host").lower()
    if backend not in {"host", "kubernetes"}:
        raise AirflowException(
            f"Unsupported ADMO_EXECUTION_BACKEND: {backend}. Use host or kubernetes."
        )
    return backend


def _safe_job_name(name: str) -> str:
    value = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return (value or "admo-workload")[:63]


def _kubernetes_namespace() -> str:
    namespace = os.getenv("ADMO_NAMESPACE")
    if namespace:
        return namespace
    try:
        return (
            Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
            .read_text()
            .strip()
        )
    except OSError:
        return "default"


def _job_environment(
    workload: Workload,
    environment: dict[str, str],
) -> dict[str, str]:
    job_environment = {
        key: value
        for key, value in environment.items()
        if key.startswith(("ADMO_", "MLFLOW_", "AWS_"))
    }
    for key, value in workload.environment.items():
        job_environment[key] = value
    return job_environment


def _print_job_logs(core, namespace: str, job_name: str) -> None:
    try:
        pods = core.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"job-name={job_name}",
        )
        for pod in pods.items:
            log = core.read_namespaced_pod_log(
                name=pod.metadata.name,
                namespace=namespace,
            )
            if log:
                print(log, end="" if log.endswith("\n") else "\n")
    except Exception as e:
        print("ERROR 2", e)


def _run_kubernetes(workload: Workload, environment: dict[str, str]) -> None:
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        try:
            config.load_kube_config()
        except config.ConfigException as error:
            raise AirflowException(
                "No Kubernetes configuration available. Run inside the cluster "
                "or provide a kubeconfig."
            ) from error

    batch = client.BatchV1Api()
    core = client.CoreV1Api()
    namespace = _kubernetes_namespace()
    job_name = _safe_job_name(workload.name)
    timeout = int(os.getenv("ADMO_WORKLOAD_TIMEOUT", "1800"))

    manifest = client.V1Job(
        metadata=client.V1ObjectMeta(
            name=job_name,
            labels={"app.kubernetes.io/managed-by": "admo-airflow"},
        ),
        spec=client.V1JobSpec(
            backoff_limit=0,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(name=job_name),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="workload",
                            image=workload.image,
                            command=list(workload.command),
                            env=[
                                client.V1EnvVar(name=key, value=value)
                                for key, value in sorted(
                                    _job_environment(workload, environment).items()
                                )
                            ],
                        )
                    ],
                ),
            ),
        ),
    )
    batch.create_namespaced_job(namespace=namespace, body=manifest)
    print(f"Started Kubernetes job {namespace}/{job_name} (image {workload.image}).")

    deadline = time.monotonic() + timeout
    try:
        while True:
            status = batch.read_namespaced_job_status(
                name=job_name,
                namespace=namespace,
            ).status
            if status.succeeded:
                _print_job_logs(core, namespace, job_name)
                return
            if status.failed:
                _print_job_logs(core, namespace, job_name)
                raise AirflowException(f"Kubernetes job {namespace}/{job_name} failed.")
            if time.monotonic() > deadline:
                raise AirflowException(
                    f"Timed out after {timeout}s waiting for Kubernetes job "
                    f"{namespace}/{job_name}."
                )
            time.sleep(5)
    finally:
        try:
            batch.delete_namespaced_job(
                name=job_name,
                namespace=namespace,
                propagation_policy="Background",
            )
        except Exception as e:
            print("ERROR 2", e)


def run_workload(workload: Workload) -> None:
    backend = _backend()
    environment = {**os.environ, **workload.environment}

    if backend == "host":
        subprocess.run(
            list(workload.host_command),
            cwd=workload.cwd,
            check=True,
            env=environment,
        )
        return

    _run_kubernetes(workload, environment)
