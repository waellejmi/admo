import pandas as pd
import numpy as np
from processing_ml.anomaly_injection import inject_anomalies
from processing_ml.columns import (
    BUILDING_SPLIT_COLUMNS,
    ENERGY_MIX_COLUMNS,
)
from processing_ml.features import add_engineered_features, add_occupation_features
from processing_ml.model_features import build_preprocessor
from processing_ml.evaluation import calculate_metrics, precision_at_k, recall_at_k


def _clean_frame(rows: int = 20) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "source_row_id": range(rows),
            "categorie_activite_majoritaire_efa": ["office"] * rows,
            "sous_categorie_activite_majoritaire_efa": ["admin"] * rows,
            "cas_assujettissement_efa": ["1A"] * rows,
            "annee_de_consommation": ["2022"] * rows,
            "ratio_de_consommation_ajustee_du_climat_kwh_par_m2": [
                100.0 + i for i in range(rows)
            ],
            "ratio_de_consommation_brut_kwh_par_m2": [90.0 + i for i in range(rows)],
            "nombre_de_categories_activite_distinctes": [1] * rows,
            "nombre_de_sous_categories_activite_distinctes": [1] * rows,
            "is_mono_occupation": [True] * rows,
        }
    )
    for column in BUILDING_SPLIT_COLUMNS:
        frame[column] = 0.0
    frame[BUILDING_SPLIT_COLUMNS[0]] = [80.0 + (i % 10) for i in range(rows)]
    frame[BUILDING_SPLIT_COLUMNS[2]] = [20.0 - (i % 10) for i in range(rows)]
    for column in ENERGY_MIX_COLUMNS:
        frame[column] = 0.0
    frame[ENERGY_MIX_COLUMNS[0]] = [50.0 + (i % 10) for i in range(rows)]
    frame[ENERGY_MIX_COLUMNS[1]] = [50.0 - (i % 10) for i in range(rows)]
    return frame


def test_engineered_features_copy_input_and_add_aggregates():
    source = _clean_frame(1)
    source_before = source.copy(deep=True)
    result = add_engineered_features(add_occupation_features(source))

    assert "delta_climat_kwh_m2" in result
    assert result["delta_climat_kwh_m2"].iloc[0] == 10.0
    assert result["sum_energy_mix_percent"].iloc[0] == 100.0
    assert result["sum_building_split_percent"].iloc[0] == 100.0
    pd.testing.assert_frame_equal(source, source_before)


def test_anomaly_injection_is_reproducible_and_preserves_clean_input():
    clean = _clean_frame()
    first = inject_anomalies(clean, anomaly_rate=0.2, random_seed=7)
    second = inject_anomalies(clean, anomaly_rate=0.2, random_seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert int(first["is_anomaly"].sum()) == 4
    assert "is_anomaly" not in clean
    assert first["source_row_id"].equals(clean["source_row_id"])


def test_model_preprocessor_fits_explicit_feature_contract():
    frame = add_occupation_features(_clean_frame())
    transformed = build_preprocessor().fit_transform(frame)

    assert transformed.shape == (len(frame), 24)


def test_ranking_metrics_measure_top_k_alert_budget():
    y_true = np.array([0, 1, 0, 1, 0])
    scores = np.array([0.1, 0.9, 0.6, 0.7, 0.2])

    assert precision_at_k(y_true, scores, 2) == 1.0
    assert recall_at_k(y_true, scores, 2) == 1.0
    metrics = calculate_metrics(y_true, scores, ks=(2,))
    assert metrics["pr_auc"] > 0.0
    assert metrics["precision_at_2"] == 1.0
