import os
from dataclasses import dataclass


@dataclass(frozen=True)
class StorageSettings:
    database_url: str
    object_storage_endpoint: str
    object_storage_access_key: str
    object_storage_secret_key: str
    object_storage_bucket: str
    object_storage_region: str


def load_settings() -> StorageSettings:
    return StorageSettings(
        database_url=os.getenv(
            "ADMO_DATABASE_URL",
            "postgresql+psycopg://admo:admo123@localhost:5432/admo",
        ),
        object_storage_endpoint=os.getenv(
            "ADMO_OBJECT_STORAGE_ENDPOINT",
            "http://localhost:3900",
        ),
        object_storage_access_key=os.getenv(
            "ADMO_OBJECT_STORAGE_ACCESS_KEY",
            "GKADMOLOCAL",
        ),
        object_storage_secret_key=os.getenv(
            "ADMO_OBJECT_STORAGE_SECRET_KEY",
            "admo_dev_secret_key_change_me",
        ),
        object_storage_bucket=os.getenv("ADMO_OBJECT_STORAGE_BUCKET", "admo"),
        object_storage_region=os.getenv("ADMO_OBJECT_STORAGE_REGION", "garage"),
    )
