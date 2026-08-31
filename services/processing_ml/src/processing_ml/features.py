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


def add_occupation_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Boolean derived from cas_assujettissement_efa
    df["is_mono_occupation"] = (df["cas_assujettissement_efa"] == "1A").astype(bool)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # delta_climat_kwh_m2: Difference between brut and ajusté ratios
    # Indicates climate sensitivity
    if (
        "ratio_de_consommation_brut_kwh_par_m2" in df.columns
        and "ratio_de_consommation_ajustee_du_climat_kwh_par_m2" in df.columns
    ):
        df["delta_climat_kwh_m2"] = (
            df["ratio_de_consommation_brut_kwh_par_m2"]
            - df["ratio_de_consommation_ajustee_du_climat_kwh_par_m2"]
        )
    
    # sum_energy_mix_percent: Sum of all fuel type percentages
    available_energy_cols = [col for col in ENERGY_MIX_COLUMNS if col in df.columns]
    if available_energy_cols:
        df["sum_energy_mix_percent"] = df[available_energy_cols].sum(axis=1)
    
    # sum_building_split_percent: Sum of individuelle, espaces_communs, and repartie percentages
    available_split_cols = [col for col in BUILDING_SPLIT_COLUMNS if col in df.columns]
    if available_split_cols:
        df["sum_building_split_percent"] = df[available_split_cols].sum(axis=1)
    
    # activity_complexity: Integer derived from nombre_de_sous_categories_activite_distinctes
    if "nombre_de_sous_categories_activite_distinctes" in df.columns:
        df["activity_complexity"] = df[
            "nombre_de_sous_categories_activite_distinctes"
        ].astype("Int64")
    
    return df
