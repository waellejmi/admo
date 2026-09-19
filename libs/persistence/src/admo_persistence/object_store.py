import boto3

from .config import load_settings


def create_object_client():
    settings = load_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.object_storage_endpoint,
        aws_access_key_id=settings.object_storage_access_key,
        aws_secret_access_key=settings.object_storage_secret_key,
        region_name=settings.object_storage_region,
    )


def put_object(
    object_key: str,
    body: bytes,
    content_type: str,
) -> None:
    settings = load_settings()
    create_object_client().put_object(
        Bucket=settings.object_storage_bucket,
        Key=object_key,
        Body=body,
        ContentType=content_type,
    )
