from io import BytesIO

import pandas as pd
from admo_ingestion import pipeline


def test_ingest_csv_to_garage_uploads_raw_parquet(monkeypatch):
    source_bytes = b"value\n1,5\n2,5\n"
    captured: dict[str, object] = {}

    def fake_get_object(object_key: str) -> bytes:
        assert object_key == "raw/run-1/source.csv"
        return source_bytes

    def fake_put_bytes(body: bytes, object_key: str, content_type: str):
        captured.update(
            body=body,
            object_key=object_key,
            content_type=content_type,
        )
        return object_key

    monkeypatch.setattr(pipeline, "get_object", fake_get_object)
    monkeypatch.setattr(pipeline, "put_bytes", fake_put_bytes)

    result = pipeline.ingest_csv_to_garage(
        "raw/run-1/source.csv",
        "raw/run-1/source.parquet",
    )

    uploaded = pd.read_parquet(BytesIO(captured["body"]))
    assert result == "raw/run-1/source.parquet"
    assert captured["content_type"] == "application/vnd.apache.parquet"
    assert uploaded["source_row_id"].tolist() == [0, 1]
    assert uploaded["value"].tolist() == [1.5, 2.5]
