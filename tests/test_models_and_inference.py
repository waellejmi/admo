import numpy as np
import pandas as pd
from admo_inference import app as inference_module

from ml.experiments.models import build_models


def test_research_benchmark_keeps_all_models():
    assert [model.name for model in build_models()] == [
        "isolation_forest",
        "lof",
        "one_class_svm",
    ]


def test_inference_uses_persisted_objects_and_model_labels(monkeypatch):
    class FakePreprocessor:
        def transform(self, frame):
            return np.zeros((len(frame), 1))

    class FakeModel:
        def score_samples(self, values):
            return np.array([0.8] * len(values))

        def predict(self, values):
            return np.array([-1] * len(values))

    monkeypatch.setattr(inference_module, "get_object", lambda key: b"artifact")
    loaded = iter([FakePreprocessor(), FakeModel()])
    monkeypatch.setattr(inference_module.joblib, "load", lambda _: next(loaded))
    service = inference_module.InferenceService(
        "models/isolation_forest/model.joblib",
        "models/isolation_forest/preprocessor.joblib",
    )

    record = {
        "annee_de_consommation": "2022",
        "cas_assujettissement_efa": "1A",
        "categorie_activite_majoritaire_efa": "office",
        "sous_categorie_activite_majoritaire_efa": "admin",
    }
    for column in (
        "nombre_de_categories_activite_distinctes",
        "nombre_de_sous_categories_activite_distinctes",
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
        "is_mono_occupation",
    ):
        record[column] = 0

    predictions = service.predict([record], "isolation_forest")
    assert predictions[0].anomaly_score == 0.8
    assert predictions[0].is_anomaly is True
