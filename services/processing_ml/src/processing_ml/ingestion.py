from pathlib import Path

import pandas as pd

from .preprocessing import load_data, preprocess


def ingest_csv_to_parquet(
    csv_path: str | Path,
    parquet_path: str | Path,
    sep: str = ";",
) -> pd.DataFrame:

    df = load_data(csv_path, sep=sep)
    df.insert(0, "source_row_id", pd.Series(range(len(df)), dtype="int64"))
    df = preprocess(df)

    output_path = Path(parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def load_processed_parquet(parquet_path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)
