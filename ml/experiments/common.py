from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from processing_ml.model_features import (
    select_features,
)

DATA_DIR = Path("data/processed")
CLEAN_DATASET = DATA_DIR / "clean_100k.parquet"
EXPERIMENT_DATASET = DATA_DIR / "experiment_100k.parquet"


def load_features(path: str | Path) -> pd.DataFrame:
    return select_features(pd.read_parquet(path))


def split_evaluation_daka(
    df: pd.DataFrame,
    validation_fraction: float = 0.5,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    rng = np.random.default_rng(random_seed)
    validation_mask = np.zeros(len(df), dtype=bool)
    if "is_anomaly" in df.columns:
        for _, group in df.groupby("is_anomaly", sort=False):
            indices = group.index.to_numpy()
            count = round(len(indices) * validation_fraction)
            selected = rng.choice(indices, size=count, replace=False)
            validation_mask[df.index.get_indexer(selected)] = True
    else:
        validation_indices = rng.choice(
            len(df), size=round(len(df) * validation_fraction), replace=False
        )
        validation_mask[validation_indices] = True
    return df.loc[validation_mask].copy(), df.loc[~validation_mask].copy()


class AnomalyModel(Protocol):
    name: str
    configuration: dict[str, Any]

    def fit(self, values: np.ndarray) -> "AnomalyModel": ...

    def score_samples(self, values: np.ndarray) -> np.ndarray: ...


@dataclass
class ExperimentResult:
    model: str
    configuration: dict[str, Any]
    preprocessing: dict[str, Any]
    training_rows: int
    validation_rows: int
    test_rows: int
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    per_type_metrics: dict[str, dict[str, float]]
