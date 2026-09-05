import pandas as pd

from .features import add_engineered_features, add_occupation_features
from .validation import validate_building_split, validate_energy_mix

NUMERIC_COLUMNS = [
    "ratio_de_consommation_ajustee_du_climat_kwh_par_m2",
    "ratio_de_consommation_brut_kwh_par_m2",
    "consommation_individuelle_pct",
    "consommation_espaces_communs_pct",
    "consommation_repartie_pct",
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
    "nombre_de_categories_activite_distinctes",
    "nombre_de_sous_categories_activite_distinctes",
]


def load_data(file_path: str, sep: str = ";") -> pd.DataFrame:
    return pd.read_csv(file_path, sep=sep, decimal=",")


def filter_accumulated_years(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["annee_de_consommation"] != "2010-2019"].copy()


def rename_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: col.replace("_%", "_pct") for col in df.columns if "_%" in col}
    return df.rename(columns=rename_map)


def cast_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_validation_flags(df: pd.DataFrame) -> pd.DataFrame:
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
