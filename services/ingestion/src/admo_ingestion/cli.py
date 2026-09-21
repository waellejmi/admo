import argparse
from pathlib import Path

from .pipeline import ingest_csv_to_garage


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a raw Parquet snapshot to Garage.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("object_key")
    parser.add_argument("--separator", default=";")
    args = parser.parse_args()
    ingest_csv_to_garage(args.csv_path, args.object_key, sep=args.separator)
