import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return float(np.inf)
    f1 = (2 * precision[:-1] * recall[:-1]) / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    return float(thresholds[int(np.nanargmax(f1))])


def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    threshold = select_threshold(y_true, scores) if threshold is None else threshold
    predictions = scores >= threshold
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "threshold": float(threshold),
    }


def metrics_by_anomaly_type(
    anomaly_types: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
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
            y_true[mask],
            scores[mask],
            threshold,
        )
    return results
