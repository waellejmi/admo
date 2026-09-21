"""Information about columns in dataset"""

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

CONSUMPTION_RATIO_COLUMNS = [
    "ratio_de_consommation_ajustee_du_climat_kwh_par_m2",
    "ratio_de_consommation_brut_kwh_par_m2",
]

ACTIVITY_COUNT_COLUMNS = [
    "nombre_de_categories_activite_distinctes",
    "nombre_de_sous_categories_activite_distinctes",
]

NUMERIC_COLUMNS = [
    *CONSUMPTION_RATIO_COLUMNS,
    *BUILDING_SPLIT_COLUMNS,
    *ENERGY_MIX_COLUMNS,
    *ACTIVITY_COUNT_COLUMNS,
]


CATEGORICAL_COLUMNS = [
    "annee_de_consommation",
    "cas_assujettissement_efa",
    "categorie_activite_majoritaire_efa",
    "sous_categorie_activite_majoritaire_efa",
    "is_mono_occupation",
]

FEATURE_COLUMNS = [
    "annee_de_consommation",
    "cas_assujettissement_efa",
    "categorie_activite_majoritaire_efa",
    "sous_categorie_activite_majoritaire_efa",
    *ACTIVITY_COUNT_COLUMNS,
    *CONSUMPTION_RATIO_COLUMNS,
    *BUILDING_SPLIT_COLUMNS,
    *ENERGY_MIX_COLUMNS,
    "is_mono_occupation",
]
