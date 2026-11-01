
import json
from pathlib import Path

from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from careflow.features import build_features


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
MODEL_PATH = ARTIFACT_DIR / "admission_xgboost.joblib"

app = FastAPI(
    title="CareFlow AI API",
    description=(
        "Research prototype for emergency department "
        "hospital admission risk prediction. "
        "Not intended for clinical decision-making."
    ),
    version="1.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "careflow-ai",
        "model_available": MODEL_PATH.is_file(),
    }


@app.get("/model/info")
def model_info():
    if not METADATA_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Model metadata is unavailable.",
        )

    try:
        metadata = json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Model metadata could not be read.",
        ) from exc

    return {
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "features": metadata["features"],
        "threshold": metadata["threshold"],
        "research_only": metadata["research_only"],
        "model_available": MODEL_PATH.is_file(),
    }


class AdmissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=0, le=120)
    sex: int = Field(ge=1, le=2)
    residence: int | None = Field(default=None, ge=1, le=4)
    triage_immediacy: int | None = Field(default=None, ge=1, le=5)

    temperature_f: float | None = Field(default=None, ge=85, le=110)
    pulse: int | None = Field(default=None, ge=0, le=240)
    respiratory_rate: int | None = Field(default=None, ge=0, le=150)
    systolic_bp: int | None = Field(default=None, ge=0, le=289)
    diastolic_bp: int | None = Field(default=None, ge=0, le=190)
    oxygen_saturation: int | None = Field(default=None, ge=0, le=100)
    pain_scale: int | None = Field(default=None, ge=0, le=10)

    visit_month: int = Field(ge=1, le=12)
    visit_day: int = Field(ge=1, le=7)
    arrival_time: str = Field(pattern=r"^\d{4}$")

    @field_validator("arrival_time")
    @classmethod
    def validate_arrival_time(cls, value: str) -> str:
        hour = int(value[:2])
        minute = int(value[2:])

        if hour > 23 or minute > 59:
            raise ValueError("Arrival time must be a valid HHMM time")

        return value


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Model artifact is unavailable.",
        )

    return joblib.load(MODEL_PATH)


@app.post("/predict")
def predict(request: AdmissionRequest):
    raw = {
        "AGE": request.age,
        "SEX": request.sex,
        "RESIDNCE": request.residence,
        "IMMEDR": request.triage_immediacy,
        "TEMPF": (
            request.temperature_f * 10
            if request.temperature_f is not None
            else None
        ),
        "PULSE": request.pulse,
        "RESPR": request.respiratory_rate,
        "BPSYS": request.systolic_bp,
        "BPDIAS": request.diastolic_bp,
        "POPCT": request.oxygen_saturation,
        "PAINSCALE": request.pain_scale,
        "VMONTH": request.visit_month,
        "VDAYR": request.visit_day,
        "ARRTIME": request.arrival_time,
    }

    features = build_features(pd.DataFrame([raw]))
    model = load_model()

    probability = float(model.predict_proba(features)[0, 1])

    return {
        "admission_probability": round(probability, 4),
        "predicted_admission": probability >= 0.20,
        "threshold": 0.20,
        "model_version": "1.0.0",
        "research_only": True,
        "disclaimer": (
            "Research prototype only. Not validated for "
            "clinical decision-making."
        ),
    }
