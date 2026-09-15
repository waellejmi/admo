from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from processing_ml.features import FEATURE_COLUMNS
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path("data/processed")
CLEAN_DATASET = DATA_DIR / "clean_100k.parquet"
EXPERIMENT_DATASET = DATA_DIR / "experiment_100k.parquet"

CATEGORICAL_FEATURES = [
    "annee_de_consommation",
    "cas_assujettissement_efa",
    "categorie_activite_majoritaire_efa",
    "sous_categorie_activite_majoritaire_efa",
]
NUMERIC_FEATURES = [
    column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES
]


def build_preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("numeric", numeric, NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def load_features(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing configured features: {sorted(missing)}")
    return df[FEATURE_COLUMNS].copy()


def split_evaluation_data(
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
            count = int(round(len(indices) * validation_fraction))
            selected = rng.choice(indices, size=count, replace=False)
            validation_mask[df.index.get_indexer(selected)] = True
    else:
        validation_indices = rng.choice(
            len(df), size=int(round(len(df) * validation_fraction)), replace=False
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
