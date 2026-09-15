import argparse
from pathlib import Path

import pandas as pd

from .anomaly_injection import create_experiment_dataset
from .ingestion import ingest_csv_to_parquet, load_parquet
from .preprocessing import preprocess_parquet


def create_dev_dataset(
    processed_path: str,
    n_rows: int = 100000,
    seed: int = 42,
    exclude_source_ids: set[int] | None = None,
) -> pd.DataFrame:
    df = load_parquet(processed_path)
    if exclude_source_ids is not None:
        df = df[~df["source_row_id"].isin(exclude_source_ids)]

    sample_size = min(n_rows, len(df))
    df = df.sample(n=sample_size, random_state=seed)
    df.reset_index(drop=True, inplace=True)

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ingest-csv", action="store_true")
    parser.add_argument("--dev-dataset", action="store_true")
    parser.add_argument("--experiment-dataset", action="store_true")
    parser.add_argument("--n-rows", type=int, default=100000)
    parser.add_argument("--clean-seed", type=int, default=42)
    parser.add_argument("--experiment-seed", type=int, default=43)
    parser.add_argument(
        "--anomaly-seed",
        type=int,
        default=44,
        help="Seed used when injecting anomalies into the evaluation sample",
    )

    parser.add_argument(
        "--raw-file",
        default="ademe_raw.parquet",
        help="Raw parquet filename",
    )

    parser.add_argument(
        "--processed-file",
        default="ademe_processed.parquet",
        help="Processed parquet filename",
    )
    parser.add_argument(
        "--clean-file",
        default="clean_100k.parquet",
        help="Clean dataset filename",
    )
    parser.add_argument(
        "--experiment-file",
        default="experiment_100k.parquet",
        help="Experiment dataset filename",
    )
    parser.add_argument(
        "--csv-file",
        default="data/OPERAT03_RATIO_CONSO_AJUSTEE.csv",
        help="Input CSV filename",
    )

    args = parser.parse_args()

    data_dir = Path("data/processed")

    raw_path = data_dir / args.raw_file
    processed_path = data_dir / args.processed_file
    clean_path = data_dir / args.clean_file
    experiment_path = data_dir / args.experiment_file
    csv_path = Path(args.csv_file)

    if args.ingest_csv:
        ingest_csv_to_parquet(
            csv_path,
            raw_path,
        )

        preprocess_parquet(
            raw_path,
            processed_path,
        )

    if args.dev_dataset:
        df = create_dev_dataset(
            processed_path,
            n_rows=args.n_rows,
            seed=args.clean_seed,
        )
        df.to_parquet(clean_path, index=False)

    if args.experiment_dataset:
        clean_df = pd.read_parquet(clean_path)
        evaluation_df = create_dev_dataset(
            processed_path,
            n_rows=args.n_rows,
            seed=args.experiment_seed,
            exclude_source_ids=set(clean_df["source_row_id"]),
        )
        evaluation_clean_path = clean_path.with_name(
            f"{clean_path.stem}_evaluation_clean.parquet"
        )
        evaluation_df.to_parquet(evaluation_clean_path, index=False)
        create_experiment_dataset(
            evaluation_clean_path,
            experiment_path,
            random_seed=args.anomaly_seed,
        )


if __name__ == "__main__":
    main()
