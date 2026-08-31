import pandas as pd

ENERGY_MIX_COLUMNS = [
    "electricite_pct",
    "gaz_naturel_reseau_pct",
    "gaz_naturel_liquefie_pct",
    "gaz_propane_pct",
    "gaz_butane_pct",
    "fioul_domestique_pct",
    "charbon_pct",
    "houille_pct",
    "bois_pct",
    "reseau_de_chaleur_pct",
    "reseau_de_froid_pct",
    "gazole_non_routier_pct",
]

BUILDING_SPLIT_COLUMNS = [
    "consommation_individuelle_pct",
    "consommation_espaces_communs_pct",
    "consommation_repartie_pct",
]


def validate_energy_mix(df: pd.DataFrame, tolerance: float = 0.1) -> pd.Series:
    mix_sum = df[ENERGY_MIX_COLUMNS].sum(axis=1).round(4)
    return (mix_sum - 100.0).abs() <= tolerance


def validate_building_split(df: pd.DataFrame, tolerance: float = 0.1) -> pd.Series:
    split_sum = df[BUILDING_SPLIT_COLUMNS].sum(axis=1).round(4)
    return (split_sum - 100.0).abs() <= tolerance
