import pandas as pd

ENERGY_MIX_COLUMNS = [
    "electricite_%",
    "gaz_naturel_reseau_%",
    "gaz_naturel_liquefie_%",
    "gaz_propane_%",
    "gaz_butane_%",
    "fioul_domestique_%",
    "charbon_%",
    "houille_%",
    "bois_%",
    "reseau_de_chaleur_%",
    "reseau_de_froid_%",
    "gazole_non_routier_%",
]

BUILDING_SPLIT_COLUMNS = [
    "consommation_individuelle_%",
    "consommation_espaces_communs_%",
    "consommation_repartie_%",
]


def validate_energy_mix(df: pd.DataFrame, tolerance: float = 0.1) -> pd.Series:
    mix_sum = df[ENERGY_MIX_COLUMNS].sum(axis=1)
    return (mix_sum - 100.0).abs() <= tolerance


def validate_building_split(df: pd.DataFrame, tolerance: float = 0.1) -> pd.Series:
    split_sum = df[BUILDING_SPLIT_COLUMNS].sum(axis=1)
    return (split_sum - 100.0).abs() <= tolerance
