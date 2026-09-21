from io import BytesIO
from pathlib import Path

import pandas as pd
from admo_persistence import ArtifactRef, put_bytes


def load_data(file_path: str | Path, sep: str = ";") -> pd.DataFrame:
    return pd.read_csv(file_path, sep=sep, decimal=",")


def ingest_csv_to_garage(
    csv_path: str | Path,
    object_key: str,
    sep: str = ";",
) -> ArtifactRef:
    frame = load_data(csv_path, sep=sep)
    frame.insert(0, "source_row_id", pd.Series(range(len(frame)), dtype="int64"))
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    return put_bytes(
        buffer.getvalue(),
        object_key,
        "application/vnd.apache.parquet",
    )
