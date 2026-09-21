from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

import boto3
from botocore.config import Config

from .config import load_settings


@dataclass(frozen=True)
class ArtifactRef:
    bucket: str
    object_key: str
    content_type: str


def artifact_ref(object_key: str, content_type: str) -> ArtifactRef:
    return ArtifactRef(load_settings().object_storage_bucket, object_key, content_type)


def create_object_client():
    settings = load_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.object_storage_endpoint,
        aws_access_key_id=settings.object_storage_access_key,
        aws_secret_access_key=settings.object_storage_secret_key,
        region_name=settings.object_storage_region,
        config=Config(s3={"addressing_style": "path"}),
    )


def put_object(
    object_key: str,
    body: bytes | BinaryIO,
    content_type: str,
) -> ArtifactRef:
    settings = load_settings()
    create_object_client().put_object(
        Bucket=settings.object_storage_bucket,
        Key=object_key,
        Body=body,
        ContentType=content_type,
    )
    return ArtifactRef(settings.object_storage_bucket, object_key, content_type)


def get_object(object_key: str) -> bytes:
    settings = load_settings()
    response = create_object_client().get_object(
        Bucket=settings.object_storage_bucket,
        Key=object_key,
    )
    return response["Body"].read()


def put_file(
    file_path: str,
    object_key: str,
    content_type: str,
) -> ArtifactRef:
    with open(file_path, "rb") as file:
        return put_object(object_key, file, content_type)


def put_bytes(
    body: bytes,
    object_key: str,
    content_type: str,
) -> ArtifactRef:
    return put_object(object_key, BytesIO(body), content_type)
