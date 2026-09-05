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

FEATURE_COLUMNS = [
    "annee_de_consommation",
    "cas_assujettissement_efa",
    "categorie_activite_majoritaire_efa",
    "sous_categorie_activite_majoritaire_efa",
    "nombre_de_categories_activite_distinctes",
    "nombre_de_sous_categories_activite_distinctes",
    "ratio_de_consommation_ajustee_du_climat_kwh_par_m2",
    "ratio_de_consommation_brut_kwh_par_m2",
    "consommation_individuelle_pct",
    "consommation_espaces_communs_pct",
    "consommation_repartie_pct",
    *ENERGY_MIX_COLUMNS,
    "is_mono_occupation",
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
        "ratio_de_consommation_ajustee_du_climat_kwh_par_m2" in df.columns
        and "ratio_de_consommation_brut_kwh_par_m2" in df.columns
    ):
        df["delta_climat_kwh_m2"] = (
            df["ratio_de_consommation_ajustee_du_climat_kwh_par_m2"]
            - df["ratio_de_consommation_brut_kwh_par_m2"]
        )

    # sum_energy_mix_percent: Sum of all fuel type percentages
    available_energy_cols = [col for col in ENERGY_MIX_COLUMNS if col in df.columns]
    if available_energy_cols:
        df["sum_energy_mix_percent"] = df[available_energy_cols].sum(axis=1)

    # sum_building_split_percent: Sum of individuelle, espaces_communs, and repartie percentages
    available_split_cols = [col for col in BUILDING_SPLIT_COLUMNS if col in df.columns]
    if available_split_cols:
        df["sum_building_split_percent"] = df[available_split_cols].sum(axis=1)

    return df
