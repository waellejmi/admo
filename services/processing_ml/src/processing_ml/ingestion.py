from pathlib import Path

import pandas as pd


def load_data(file_path: str | Path, sep: str = ";") -> pd.DataFrame:
    return pd.read_csv(file_path, sep=sep, decimal=",")


def ingest_csv_to_parquet(
    csv_path: str | Path,
    parquet_path: str | Path,
    sep: str = ";",
) -> pd.DataFrame:
    df = load_data(csv_path, sep=sep)
    df.insert(0, "source_row_id", pd.Series(range(len(df)), dtype="int64"))

    output_path = Path(parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def load_parquet(file_path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(file_path)
