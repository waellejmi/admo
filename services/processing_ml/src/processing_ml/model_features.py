from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .columns import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, values: np.ndarray, y: Any = None) -> "FrequencyEncoder":
        frame = pd.DataFrame(values)
        self.frequencies_ = [
            frame[column].value_counts(normalize=True, dropna=False).to_dict()
            for column in frame.columns
        ]
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values)
        encoded = np.column_stack(
            [
                frame[column].map(self.frequencies_[column]).fillna(0.0)
                for column in frame.columns
            ]
        )
        return encoded.astype(float)


def build_preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("frequency", FrequencyEncoder()),
            ("scaler", StandardScaler()),
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
            ("categorical", categorical, CATEGORICAL_COLUMNS),
            ("numeric", numeric, NUMERIC_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def select_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing configured features: {sorted(missing)}")
    return frame[FEATURE_COLUMNS].copy()
