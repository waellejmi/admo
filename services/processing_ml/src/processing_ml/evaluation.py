from __future__ import annotations

import argparse
import json
import os
from io import BytesIO
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from admo_persistence import get_object
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .model_features import select_features


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    _validate_k(k, len(y_true))
    return float(np.mean(y_true[np.argsort(scores)[-k:]]))


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    _validate_k(k, len(y_true))
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    return float(y_true[np.argsort(scores)[-k:]].sum() / positives)


def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return float(np.inf)
    f1 = (2 * precision[:-1] * recall[:-1]) / np.maximum(
        precision[:-1] + recall[:-1],
        np.finfo(float).eps,
    )
    return float(thresholds[int(np.nanargmax(f1))])


def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
    ks: tuple[int, ...] = (),
) -> dict[str, float]:
    threshold = select_threshold(y_true, scores) if threshold is None else threshold
    predictions = scores >= threshold
    metrics = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "threshold": float(threshold),
    }
    for k in ks:
        metrics[f"precision_at_{k}"] = precision_at_k(y_true, scores, k)
        metrics[f"recall_at_{k}"] = recall_at_k(y_true, scores, k)
    return metrics


def metrics_by_anomaly_type(
    anomaly_types: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    ks: tuple[int, ...] = (),
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for anomaly_type in (
        "contextual",
        "correlational",
        "structural",
        "energy_mix",
        "combination",
    ):
        mask = (anomaly_types == anomaly_type) | (y_true == 0)
        if not mask.any() or not y_true[mask].any():
            continue
        results[anomaly_type] = calculate_metrics(
            y_true[mask], scores[mask], threshold, ks
        )
    return results


def evaluate_labeled_model(
    model: Any,
    preprocessor: Any,
    labeled_evaluation: pd.DataFrame,
    ks: tuple[int, ...] = (),
) -> dict[str, Any]:
    required = {"is_anomaly", "anomaly_type"}
    missing = required - set(labeled_evaluation.columns)
    if missing:
        raise ValueError(f"Evaluation dataset is missing labels: {sorted(missing)}")
    values = preprocessor.transform(select_features(labeled_evaluation))
    scores = -model.score_samples(values)
    y_true = labeled_evaluation["is_anomaly"].astype(int).to_numpy()
    threshold = select_threshold(y_true, scores)
    return {
        "metrics": calculate_metrics(y_true, scores, threshold, ks),
        "per_type_metrics": metrics_by_anomaly_type(
            labeled_evaluation["anomaly_type"].to_numpy(),
            y_true,
            scores,
            threshold,
            ks,
        ),
        "evaluation_rows": len(labeled_evaluation),
        "anomaly_count": int(y_true.sum()),
        "anomaly_rate": float(y_true.mean()),
        "score_direction": "higher_is_more_anomalous",
    }


def evaluate_persisted_model(
    model_object_key: str,
    preprocessor_object_key: str,
    evaluation_object_key: str,
    ks: tuple[int, ...] = (),
) -> dict[str, Any]:
    model = joblib.load(BytesIO(get_object(model_object_key)))
    preprocessor = joblib.load(BytesIO(get_object(preprocessor_object_key)))
    labeled_evaluation = pd.read_parquet(
        BytesIO(get_object(evaluation_object_key))
    )
    return evaluate_labeled_model(
        model,
        preprocessor,
        labeled_evaluation,
        ks,
    )


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


def _validate_k(k: int, row_count: int) -> None:
    if not 1 <= k <= row_count:
        raise ValueError(f"k must be between 1 and {row_count}, got {k}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate persisted ADMO model artifacts on injected labels."
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--preprocessor-key", required=True)
    parser.add_argument("--evaluation-key", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--k", type=int, action="append", default=[100])
    args = parser.parse_args()
    report = evaluate_persisted_model(
        args.model_key,
        args.preprocessor_key,
        args.evaluation_key,
        tuple(sorted(set(args.k))),
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(
        os.getenv("MLFLOW_EXPERIMENT_NAME", "admo-production-training")
    )
    with mlflow.start_run(run_name=f"evaluation-{args.version}"):
        mlflow.set_tags(
            {
                "admo_model_version": args.version,
                "admo_model_key": args.model_key,
                "admo_preprocessor_key": args.preprocessor_key,
                "admo_evaluation_dataset_key": args.evaluation_key,
                "admo_score_direction": report["score_direction"],
            }
        )
        mlflow.log_params(
            {
                "evaluation_rows": report["evaluation_rows"],
                "evaluation_anomaly_count": report["anomaly_count"],
            }
        )
        mlflow.log_metrics(
            {f"evaluation_{name}": value for name, value in report["metrics"].items()}
        )
        for anomaly_type, metrics in report["per_type_metrics"].items():
            mlflow.log_metrics(
                {
                    f"evaluation_{name}_{anomaly_type}": value
                    for name, value in metrics.items()
                }
            )
        mlflow.log_text(report_json(report), "evaluation_metrics.json")


if __name__ == "__main__":
    main()
