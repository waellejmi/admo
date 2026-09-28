from __future__ import annotations

import os
import re
import subprocess
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
    if backend not in {"host", "docker", "kubernetes"}:
        raise AirflowException(
            f"Unsupported ADMO_EXECUTION_BACKEND: {backend}. "
            "Use host, docker, or kubernetes."
        )
    return backend


def _safe_job_name(name: str) -> str:
    value = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return (value or "admo-workload")[:63]


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

    if backend == "docker":
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            os.getenv("ADMO_DOCKER_NETWORK", "admo_default"),
        ]
        container_environment = {
            key: value
            for key, value in environment.items()
            if key.startswith(("ADMO_", "MLFLOW_", "AWS_"))
        }
        project_host_path = os.getenv("ADMO_PROJECT_HOST_PATH")
        if project_host_path:
            command.extend(
                [
                    "--volume",
                    f"{project_host_path}:{os.getenv('ADMO_PROJECT_ROOT', '/opt/admo')}:ro",
                ]
            )
        for key, value in container_environment.items():
            command.extend(["--env", f"{key}={value}"])
        command.extend([workload.image, *workload.command])
        subprocess.run(command, cwd=workload.cwd, check=True, env=environment)
        return

    job_name = _safe_job_name(workload.name)
    create_command = [
        "kubectl",
        "create",
        "job",
        job_name,
        "--image",
        workload.image,
    ]
    for key, value in workload.environment.items():
        create_command.extend(["--env", f"{key}={value}"])
    create_command.extend(
        [
            "--",
            *workload.command,
        ]
    )
    subprocess.run(create_command, cwd=workload.cwd, check=True, env=environment)
    try:
        subprocess.run(
            [
                "kubectl",
                "wait",
                "--for=condition=complete",
                f"job/{job_name}",
                "--timeout=30m",
            ],
            cwd=workload.cwd,
            check=True,
            env=environment,
        )
        subprocess.run(
            ["kubectl", "logs", f"job/{job_name}"],
            cwd=workload.cwd,
            check=True,
            env=environment,
        )
    finally:
        subprocess.run(
            ["kubectl", "delete", "job", job_name, "--ignore-not-found"],
            cwd=workload.cwd,
            check=False,
            env=environment,
        )
