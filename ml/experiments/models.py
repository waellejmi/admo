from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


@dataclass
class SklearnAnomalyModel:
    name: str
    estimator: Any
    configuration: dict[str, Any] = field(default_factory=dict)

    def fit(self, values: np.ndarray) -> "SklearnAnomalyModel":
        self.estimator.fit(values)
        return self

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        if isinstance(self.estimator, LocalOutlierFactor):
            return -self.estimator.score_samples(values)
        return -self.estimator.decision_function(values)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.estimator.predict(values)


def build_models(random_seed: int = 42) -> list[SklearnAnomalyModel]:
    return [
        SklearnAnomalyModel(
            name="isolation_forest",
            estimator=IsolationForest(
                n_estimators=200,
                max_samples="auto",
                contamination="auto",
                random_state=random_seed,
                n_jobs=-1,
            ),
            configuration={
                "n_estimators": 200,
                "max_samples": "auto",
                "contamination": "auto",
                "random_state": random_seed,
            },
        ),
        SklearnAnomalyModel(
            name="lof",
            estimator=LocalOutlierFactor(
                n_neighbors=35,
                metric="minkowski",
                novelty=True,
                n_jobs=-1,
            ),
            configuration={
                "n_neighbors": 35,
                "metric": "minkowski",
                "novelty": True,
            },
        ),
        SklearnAnomalyModel(
            name="one_class_svm",
            estimator=OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=0.05,
            ),
            configuration={
                "kernel": "rbf",
                "gamma": "scale",
                "nu": 0.05,
            },
        ),
    ]
