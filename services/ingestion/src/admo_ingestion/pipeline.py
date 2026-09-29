from io import BytesIO

import pandas as pd
from admo_persistence import ArtifactRef, get_object, put_bytes


def load_data(source: bytes, sep: str = ";") -> pd.DataFrame:
    return pd.read_csv(BytesIO(source), sep=sep, decimal=",")


def ingest_csv_to_garage(
    source_key: str,
    object_key: str,
    sep: str = ";",
) -> ArtifactRef:
    frame = load_data(get_object(source_key), sep=sep)
    frame.insert(0, "source_row_id", pd.Series(range(len(frame)), dtype="int64"))
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    return put_bytes(
        buffer.getvalue(),
        object_key,
        "application/vnd.apache.parquet",
    )
