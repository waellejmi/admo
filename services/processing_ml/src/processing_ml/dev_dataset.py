import pandas as pd

from .anomaly_injection import create_experiment_dataset
from .ingestion import ingest_csv_to_parquet, load_processed_parquet


def create_dev_dataset(
    processed_path: str, n_rows: int = 100000, seed: int = 42
) -> pd.DataFrame:
    df = load_processed_parquet(processed_path)

    sample_size = min(n_rows, len(df))
    df = df.sample(n=sample_size, random_state=seed)
    df.reset_index(drop=True, inplace=True)

    return df


if __name__ == "__main__":
    processed_path = "data/processed/ademe_processed.parquet"
    clean_path = "data/processed/clean_100k.parquet"
    ingest_csv_to_parquet(
        "data/OPERAT03_RATIO_CONSO_AJUSTEE.csv",
        processed_path,
    )
    df = create_dev_dataset(processed_path)
    df.to_parquet(clean_path, index=False)
    create_experiment_dataset(
        clean_path,
        "data/processed/experiment_100k.parquet",
    )
