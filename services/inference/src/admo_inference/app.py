from io import BytesIO
import os
from typing import Any

import joblib
import pandas as pd
from admo_persistence import get_object
from admo_persistence.database import session_scope
from admo_persistence.registry import resolve_model_artifacts
from fastapi import FastAPI, HTTPException
from processing_ml.model_features import select_features
from pydantic import BaseModel, Field

SUPPORTED_MODELS = {"isolation_forest"}
DEFAULT_MODEL_OBJECT_KEY = os.getenv(
    "ADMO_MODEL_OBJECT_KEY",
    "",
)
DEFAULT_PREPROCESSOR_OBJECT_KEY = os.getenv(
    "ADMO_PREPROCESSOR_OBJECT_KEY",
    "",
)


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1)
    model: str = "isolation_forest"
    model_version: str | None = None


class Prediction(BaseModel):
    anomaly_score: float
    is_anomaly: bool


class PredictionResponse(BaseModel):
    model: str
    model_version: str | None = None
    predictions: list[Prediction]


class InferenceService:
    def __init__(
        self,
        model_object_key: str | None = DEFAULT_MODEL_OBJECT_KEY,
        preprocessor_object_key: str | None = DEFAULT_PREPROCESSOR_OBJECT_KEY,
        model_version: str | None = None,
    ) -> None:
        self.model_object_key = model_object_key
        self.preprocessor_object_key = preprocessor_object_key
        self.model_version = model_version
        self._resolved_version: str | None = None
        self._preprocessor = None
        self._models: dict[str, Any] = {}

    def predict(
        self,
        records: list[dict[str, Any]],
        model_name: str,
        model_version: str | None = None,
    ) -> list[Prediction]:
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")
        self._resolve_artifacts(model_name, model_version)
        frame = select_features(pd.DataFrame.from_records(records))
        values = self._load_preprocessor().transform(frame)
        model = self._load_model(model_name)
        scores = model.score_samples(values)
        labels = model.predict(values)
        return [
            Prediction(
                anomaly_score=float(score),
                is_anomaly=bool(label == -1),
            )
            for score, label in zip(scores, labels, strict=True)
        ]

    def _resolve_artifacts(
        self,
        model_name: str,
        model_version: str | None,
    ) -> None:
        if self.model_object_key and self.preprocessor_object_key:
            return
        requested_version = model_version or self.model_version
        if self._resolved_version is not None and self._resolved_version == requested_version:
            return
        with session_scope() as session:
            refs = resolve_model_artifacts(
                session,
                model_name,
                requested_version,
            )
        self.model_object_key = refs.model_object_key
        self.preprocessor_object_key = refs.preprocessor_object_key
        self._resolved_version = requested_version
        self._preprocessor = None
        self._models.clear()

    def _load_preprocessor(self) -> Any:
        if self._preprocessor is None:
            self._preprocessor = joblib.load(
                BytesIO(get_object(self.preprocessor_object_key))
            )
        return self._preprocessor

    def _load_model(self, model_name: str) -> Any:
        if model_name not in self._models:
            if model_name != "isolation_forest":
                raise ValueError(f"Unsupported model: {model_name}")
            self._models[model_name] = joblib.load(
                BytesIO(get_object(self.model_object_key))
            )
        return self._models[model_name]


def create_app(
    model_object_key: str | None = None,
    preprocessor_object_key: str | None = None,
) -> FastAPI:
    service = InferenceService(model_object_key, preprocessor_object_key)
    app = FastAPI(title="ADMO Inference API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        try:
            predictions = service.predict(
                request.records,
                request.model,
                request.model_version,
            )
        except (FileNotFoundError, LookupError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return PredictionResponse(
            model=request.model,
            model_version=request.model_version,
            predictions=predictions,
        )

    return app


app = create_app()
