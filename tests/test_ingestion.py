from pathlib import Path

import pandas as pd
from admo_ingestion import pipeline


def test_ingest_csv_to_garage_uploads_raw_parquet(monkeypatch, tmp_path: Path):
    source = tmp_path / "source.csv"
    pd.DataFrame({"value": ["1,5", "2,5"]}).to_csv(
        source,
        sep=";",
        index=False,
    )
    captured: dict[str, object] = {}

    def fake_put_bytes(body: bytes, object_key: str, content_type: str):
        captured.update(
            body=body,
            object_key=object_key,
            content_type=content_type,
        )
        return object_key

    monkeypatch.setattr(pipeline, "put_bytes", fake_put_bytes)

    result = pipeline.ingest_csv_to_garage(source, "raw/run-1/source.parquet")

    uploaded = pd.read_parquet(__import__("io").BytesIO(captured["body"]))
    assert result == "raw/run-1/source.parquet"
    assert captured["content_type"] == "application/vnd.apache.parquet"
    assert uploaded["source_row_id"].tolist() == [0, 1]
    assert uploaded["value"].tolist() == [1.5, 2.5]
