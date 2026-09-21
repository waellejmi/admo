import argparse

import pandas as pd
from admo_persistence import artifact_ref

from .artifacts import publish_dataset, read_parquet_artifact
from .preprocessing import preprocess_raw_frame


def create_dev_dataset(
    processed_frame: pd.DataFrame,
    n_rows: int = 100000,
    seed: int = 42,
    exclude_source_ids: set[int] | None = None,
) -> pd.DataFrame:
    df = processed_frame
    if exclude_source_ids is not None:
        df = df[~df["source_row_id"].isin(exclude_source_ids)]

    sample_size = min(n_rows, len(df))
    df = df.sample(n=sample_size, random_state=seed)
    df.reset_index(drop=True, inplace=True)

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw-key", required=True)
    parser.add_argument("--clean-key", default="processed/ademe/clean_100k.parquet")
    parser.add_argument("--n-rows", type=int, default=100000)
    parser.add_argument("--clean-seed", type=int, default=42)
    args = parser.parse_args()
    raw_artifact = artifact_ref(args.raw_key, "application/vnd.apache.parquet")
    processed = preprocess_raw_frame(read_parquet_artifact(raw_artifact))
    clean = create_dev_dataset(processed, n_rows=args.n_rows, seed=args.clean_seed)
    from admo_persistence.database import session_scope
    from admo_persistence.registry import create_pipeline_run

    with session_scope() as session:
        run = create_pipeline_run(
            session,
            {
                "component": "processing_ml",
                "seed": args.clean_seed,
                "source_object_key": args.raw_key,
            },
        )
        publish_dataset(session, run, clean, args.clean_key, "clean_100k")
        run.status = "completed"


if __name__ == "__main__":
    main()
