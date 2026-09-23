import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.experiments.common import EXPERIMENT_DATASET, load_features

ANOMALY_TYPES = (
    "contextual",
    "correlational",
    "structural",
    "energy_mix",
    "combination",
)


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _safe_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    if not y_true.any():
        return float("nan")
    return float(average_precision_score(y_true, scores))


def _score_summary(scores: np.ndarray, prefix: str) -> dict[str, float]:
    quantiles = np.quantile(scores, (0.05, 0.50, 0.95))
    return {
        f"{prefix}_p05": float(quantiles[0]),
        f"{prefix}_median": float(quantiles[1]),
        f"{prefix}_p95": float(quantiles[2]),
        f"{prefix}_mean": float(np.mean(scores)),
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def build_diagnostics(
    experiment_path: str | Path = EXPERIMENT_DATASET,
    model_dir: str | Path = "ml/models",
    output_path: str | Path = "ml/reports/anomaly_diagnostics.md",
    anomaly_rate: float = 0.05,
) -> pd.DataFrame:
    experiment = pd.read_parquet(experiment_path)
    required = {"is_anomaly", "anomaly_type"}
    missing = required - set(experiment.columns)
    if missing:
        raise ValueError(f"Evaluation dataset is missing labels: {sorted(missing)}")
    if not 0 < anomaly_rate < 1:
        raise ValueError("anomaly_rate must be between 0 and 1.")

    model_path = Path(model_dir)
    preprocessor = joblib.load(model_path / "preprocessor.joblib")
    values = preprocessor.transform(load_features(experiment_path))
    y_true = experiment["is_anomaly"].astype(bool).to_numpy()
    anomaly_types = experiment["anomaly_type"].astype(str).to_numpy()

    overall_rows: list[dict[str, float | str]] = []
    type_rows: list[dict[str, float | str]] = []

    for model_file in sorted(model_path.glob("*.joblib")):
        if model_file.name == "preprocessor.joblib":
            continue

        model = joblib.load(model_file)
        model_name = model_file.stem
        scores = np.asarray(model.score_samples(values), dtype=float)
        normal_scores = scores[~y_true]
        anomaly_scores = scores[y_true]
        expected_count = max(1, round(len(scores) * anomaly_rate))
        threshold = float(np.partition(scores, -expected_count)[-expected_count])

        overall_rows.append(
            {
                "model": model_name,
                "anomaly_count": int(y_true.sum()),
                "anomaly_rate": float(y_true.mean()),
                "pr_auc": _safe_average_precision(y_true, scores),
                "roc_auc": _safe_auc(y_true, scores),
                "top_rate_precision": float(y_true[scores >= threshold].mean()),
                "top_rate_recall": float((anomaly_scores >= threshold).mean()),
                "threshold": threshold,
                **_score_summary(normal_scores, "normal_score"),
                **_score_summary(anomaly_scores, "anomaly_score"),
            }
        )

        for anomaly_type in ANOMALY_TYPES:
            type_mask = anomaly_types == anomaly_type
            if not type_mask.any():
                continue
            subset_mask = type_mask | ~y_true
            type_scores = scores[subset_mask]
            type_labels = type_mask[subset_mask]
            type_anomaly_scores = scores[type_mask]
            baseline_pr_auc = float(type_mask.sum() / subset_mask.sum())
            type_pr_auc = _safe_average_precision(type_labels, type_scores)
            type_rows.append(
                {
                    "model": model_name,
                    "anomaly_type": anomaly_type,
                    "anomaly_count": int(type_mask.sum()),
                    "baseline_pr_auc": baseline_pr_auc,
                    "pr_auc": type_pr_auc,
                    "pr_auc_lift": float(type_pr_auc / baseline_pr_auc),
                    "roc_auc": _safe_auc(type_labels, type_scores),
                    "score_lift_vs_normals": float(
                        np.mean(type_anomaly_scores) - np.mean(normal_scores)
                    ),
                    "median_score_lift_vs_normals": float(
                        np.median(type_anomaly_scores) - np.median(normal_scores)
                    ),
                    "top_rate_recall": float(
                        (type_anomaly_scores >= threshold).mean()
                    ),
                }
            )

    overall = pd.DataFrame(overall_rows).sort_values("pr_auc", ascending=False)
    by_type = pd.DataFrame(type_rows)
    hardest = (
        by_type.sort_values(["model", "roc_auc", "score_lift_vs_normals"])
        .groupby("model", as_index=False)
        .first()
    )

    lines = [
        "# Anomaly diagnostics",
        "",
        "This report evaluates persisted unsupervised models against labels that are used only after scoring.",
        "",
        "## Overall results",
        "",
        _markdown_table(overall),
        "",
        "## Hardest anomaly family by model",
        "",
        "Lower ROC-AUC and lower score lift indicate a harder anomaly family.",
        "PR-AUC should be compared with the corresponding family baseline.",
        "",
        _markdown_table(hardest),
        "",
        "## Per-family results",
        "",
        _markdown_table(
            by_type.sort_values(["anomaly_type", "model"])
        ),
        "",
        "## Interpretation",
        "",
        "- ROC-AUC near 0.50 means the model ranks that anomaly family close to random.",
        "- PR-AUC lift below 1.0 means the model performs below the family prevalence baseline.",
        "- A low or negative score lift means injected anomalies do not consistently receive higher anomaly scores than normal rows.",
        "- Low top-rate recall indicates that the model's highest anomaly scores miss most rows from that family.",
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-path", default=str(EXPERIMENT_DATASET))
    parser.add_argument("--model-dir", default="ml/models")
    parser.add_argument(
        "--output-path",
        default="ml/reports/anomaly_diagnostics.md",
    )
    parser.add_argument("--anomaly-rate", type=float, default=0.05)
    args = parser.parse_args()
    build_diagnostics(
        experiment_path=args.experiment_path,
        model_dir=args.model_dir,
        output_path=args.output_path,
        anomaly_rate=args.anomaly_rate,
    )
