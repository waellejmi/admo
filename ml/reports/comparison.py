import json
from pathlib import Path

import pandas as pd


def build_comparison_report(
    results_path: str | Path = "ml/reports/experiment_results.json",
    output_path: str | Path = "ml/reports/comparison.csv",
) -> pd.DataFrame:
    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    rows = []
    for result in payload:
        metrics = result["test_metrics"]
        rows.append(
            {
                "model": result["model"],
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "training_seconds": result["configuration"]["training_seconds"],
                "inference_seconds": result["configuration"]["inference_seconds"],
            }
        )
    report = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    report.to_csv(output_path, index=False)
    return report


if __name__ == "__main__":
    build_comparison_report()
