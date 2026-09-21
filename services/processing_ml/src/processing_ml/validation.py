import pandas as pd

from .columns import BUILDING_SPLIT_COLUMNS, ENERGY_MIX_COLUMNS


def validate_energy_mix(df: pd.DataFrame, tolerance: float = 0.1) -> pd.Series:
    mix_sum = df[ENERGY_MIX_COLUMNS].sum(axis=1).round(4)
    return (mix_sum - 100.0).abs() <= tolerance


def validate_building_split(df: pd.DataFrame, tolerance: float = 0.1) -> pd.Series:
    split_sum = df[BUILDING_SPLIT_COLUMNS].sum(axis=1).round(4)
    return (split_sum - 100.0).abs() <= tolerance
