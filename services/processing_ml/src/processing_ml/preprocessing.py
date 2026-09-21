from pathlib import Path

import pandas as pd

from .columns import NUMERIC_COLUMNS
from .features import add_engineered_features, add_occupation_features
from .validation import validate_building_split, validate_energy_mix


def filter_accumulated_years(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["annee_de_consommation"] != "2010-2019"].copy()


def rename_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: col.replace("_%", "_pct") for col in df.columns if "_%" in col}
    return df.rename(columns=rename_map)


def cast_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_validation_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_valid_energy_mix"] = validate_energy_mix(df)
    df["is_valid_building_split"] = validate_building_split(df)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = filter_accumulated_years(df)
    df = rename_percent_columns(df)
    df = cast_to_numeric(df)
    df = add_validation_flags(df)
    df = add_occupation_features(df)
    df = add_engineered_features(df)
    df.reset_index(drop=True, inplace=True)
    return df


def preprocess_parquet(
    raw_path: str | Path,
    processed_path: str | Path,
) -> pd.DataFrame:
    df = preprocess(pd.read_parquet(raw_path))
    output_path = Path(processed_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def preprocess_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    return preprocess(df)
