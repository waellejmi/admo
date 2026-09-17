import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.evaluation.metrics import calculate_metrics, metrics_by_anomaly_type
from ml.experiments.common import (
    CLEAN_DATASET,
    EXPERIMENT_DATASET,
    ExperimentResult,
    build_preprocessor,
    load_features,
)
from ml.experiments.models import build_models


def run_experiment(
    clean_path: str | Path = CLEAN_DATASET,
    experiment_path: str | Path = EXPERIMENT_DATASET,
    output_dir: str | Path = "ml/reports",
    model_dir: str | Path = "ml/models",
    random_seed: int = 42,
) -> list[ExperimentResult]:
    clean = load_features(clean_path)
    experiment = pd.read_parquet(experiment_path)
    required_labels = {"is_anomaly", "anomaly_type"}
    missing_labels = required_labels - set(experiment.columns)
    if missing_labels:
        raise ValueError(
            f"Evaluation dataset is missing labels: {sorted(missing_labels)}"
        )
    preprocessor = build_preprocessor()
    train_values = preprocessor.fit_transform(clean)
    evaluation_values = preprocessor.transform(experiment)
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, model_path / "preprocessor.joblib")
    y_evaluation = experiment["is_anomaly"].astype(int).to_numpy()
    results: list[ExperimentResult] = []

    for model in build_models(random_seed):
        started = time.perf_counter()
        model.fit(train_values)
        training_seconds = time.perf_counter() - started
        started = time.perf_counter()
        evaluation_scores = model.score_samples(evaluation_values)
        inference_seconds = time.perf_counter() - started
        model_file = model_path / f"{model.name}.joblib"
        joblib.dump(model, model_file)
        persisted_model = joblib.load(model_file)
        expected_anomaly_count = max(1, round(len(experiment) * 0.05))
        threshold = float(
            np.partition(evaluation_scores, -expected_anomaly_count)[
                -expected_anomaly_count
            ]
        )
        persisted_scores = persisted_model.score_samples(evaluation_values)
        evaluation_metrics = calculate_metrics(
            y_evaluation,
            persisted_scores,
            threshold=threshold,
        )
        results.append(
            ExperimentResult(
                model=model.name,
                configuration={
                    **model.configuration,
                    "random_seed": random_seed,
                    "training_seconds": training_seconds,
                    "inference_seconds": inference_seconds,
                    "model_path": str(model_file),
                    "preprocessor_path": str(model_path / "preprocessor.joblib"),
                    "threshold_policy": "top_5_percent_of_evaluation_scores",
                },
                preprocessing={
                    "features": list(clean.columns),
                    "categorical_features": [
                        "annee_de_consommation",
                        "cas_assujettissement_efa",
                        "categorie_activite_majoritaire_efa",
                        "sous_categorie_activite_majoritaire_efa",
                    ],
                    "numeric_imputation": "median",
                    "categorical_imputation": "most_frequent",
                    "numeric_scaling": "standard",
                    "categorical_encoding": "frequency",
                },
                training_rows=len(clean),
                validation_rows=0,
                test_rows=len(experiment),
                validation_metrics={},
                test_metrics=evaluation_metrics,
                per_type_metrics=metrics_by_anomaly_type(
                    experiment["anomaly_type"].to_numpy(),
                    y_evaluation,
                    persisted_scores,
                    threshold,
                ),
            )
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = [result.__dict__ for result in results]
    (output_path / "experiment_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return results


if __name__ == "__main__":
    run_experiment()
