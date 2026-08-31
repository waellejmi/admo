import pandas as pd

from .features import add_occupation_features
from .validation import validate_building_split, validate_energy_mix

NUMERIC_COLUMNS = [
    "ratio_de_consommation_ajustee_du_climat_kwh_par_m2",
    "ratio_de_consommation_brut_kwh_par_m2",
    "consommation_individuelle_%",
    "consommation_espaces_communs_%",
    "consommation_repartie_%",
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


def load_data(file_path: str, sep: str = ";") -> pd.DataFrame:
    return pd.read_csv(file_path, sep=sep)


def filter_accumulated_years(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["annee_de_consommation"] != "2010-2019"].copy()


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
    df = cast_to_numeric(df)
    df = add_validation_flags(df)
    df = add_occupation_features(df)
    df.reset_index(drop=True, inplace=True)
    return df
